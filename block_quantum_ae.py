#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Architecture #2A + Option B (Mahalanobis):
- Background-only training objective: "clean the trash qubits" per particle block
- No classical decoder, no KL.
- Per-event block scores s = [s_MET, s_e, s_mu, s_j] in R^4
- Fit (mu, Sigma) on background scores; anomaly score = 0.5 * (s-mu)^T Sigma^{-1} (s-mu)

This is a drop-in style rewrite using your earlier PennyLane/PyTorch patterns:
angle encoding (RX/RY/RZ) + CNOT ladder + depth trainable RX per wire.

Notes:
- Uses float64 for PennyLane backprop compatibility.
- Presence masking: for standardized data you should NOT use (x != 0) as a padding mask.
  This implementation accepts optional presence masks; if you don't pass them, it assumes "all present".
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import pennylane as qml


def _trash_excitation_from_z(z_expvals: torch.Tensor) -> torch.Tensor:
    """
    Convert PauliZ expectation values into "trash excitation probability proxy":
      p(|1>) = (1 - <Z>)/2
    Works elementwise.

    z_expvals: (...,) in [-1, 1]
    returns: (...,) in [0, 1]
    """
    return 0.5 * (1.0 - z_expvals)


# Feature index mapping for 56-feature input:
# PT:  MET=0, e0-e3=2-5, μ0-μ3=14-17, j0-j9=26-35
# Eta: MET=N/A, e0-e3=6-9, μ0-μ3=18-21, j0-j9=36-45
# Phi: MET=1, e0-e3=10-13, μ0-μ3=22-25, j0-j9=46-55

PARTICLE_INDICES = {
    # (pt_idx, eta_idx, phi_idx) - eta_idx is None for MET
    'MET': (0, None, 1),
    'e0': (2, 6, 10),
    'e1': (3, 7, 11),
    'e2': (4, 8, 12),
    'e3': (5, 9, 13),
    'mu0': (14, 18, 22),
    'mu1': (15, 19, 23),
    'mu2': (16, 20, 24),
    'mu3': (17, 21, 25),
    'j0': (26, 36, 46),
    'j1': (27, 37, 47),
    'j2': (28, 38, 48),
    'j3': (29, 39, 49),
    'j4': (30, 40, 50),
    'j5': (31, 41, 51),
    'j6': (32, 42, 52),
    'j7': (33, 43, 53),
    'j8': (34, 44, 54),
    'j9': (35, 45, 55),
}


class ParticleQAEEncoder(nn.Module):
    """
    Particle-wise QAE-style encoder:
      - MET: 1 qubit
      - Electrons: 4 qubits
      - Muons: 4 qubits
      - Jets: 10 qubits

    Each block returns PauliZ expectations ONLY on trash wires.
    """

    def __init__(self, depth: int = 2, device: str = "default.qubit"):
        super().__init__()
        self.depth = int(depth)
        self.device_name = device

        # Particle assignments for each block - standard layout
        self.met_particles = ['MET']
        self.ele_particles = ['e0', 'e1', 'e2', 'e3']
        self.mu_particles = ['mu0', 'mu1', 'mu2', 'mu3']
        self.jet_particles = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'j8', 'j9']

        # Latent/trash wire assignments
        self.met_latent = []
        self.met_trash = [0]
        
        self.ele_latent = [0, 1]
        self.ele_trash = [2, 3]
        
        self.mu_latent = [0, 1]
        self.mu_trash = [2, 3]
        
        self.jet_latent = [0, 1, 2, 3, 4, 5]
        self.jet_trash = [6, 7, 8, 9]

        self._qnodes = {}
        self._build_circuits()

        # Trainable parameters
        self.met_weights = nn.Parameter(0.01 * torch.randn(1 * 1, dtype=torch.float64))    # depth=1
        self.ele_weights = nn.Parameter(0.01 * torch.randn(1 * 4, dtype=torch.float64))    # depth=1, 4 qubits
        self.mu_weights  = nn.Parameter(0.01 * torch.randn(1 * 4, dtype=torch.float64))    # depth=1, 4 qubits
        self.jet_weights = nn.Parameter(0.01 * torch.randn(4 * 10, dtype=torch.float64))   # depth=4, 10 qubits

    def _build_circuits(self):
        """Build quantum circuits for each particle block."""
        ele_latent = self.ele_latent
        ele_trash = self.ele_trash
        mu_latent = self.mu_latent
        mu_trash = self.mu_trash
        jet_latent = self.jet_latent
        jet_trash = self.jet_trash

        # MET circuit (1 qubit)
        met_dev = qml.device(self.device_name, wires=1)

        @qml.qnode(met_dev, interface="torch", diff_method="backprop")
        def qnode_met(pt, phi, weights):
            qml.RY(pt, wires=0)
            qml.RZ(phi, wires=0)
            for d in range(1):  # depth = 1
                qml.RY(weights[d], wires=0)
            return qml.expval(qml.PauliZ(0))

        self._qnodes["met"] = qnode_met

        # Electron circuit (4 qubits)
        ele_dev = qml.device(self.device_name, wires=4)

        @qml.qnode(ele_dev, interface="torch", diff_method="backprop")
        def qnode_ele(pt_vec, eta_vec, phi_vec, weights):
            for k in range(4):
                qml.RX(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RZ(phi_vec[:, k], wires=k)
            for d in range(1):  # depth = 1
                for t in ele_trash:
                    for l in ele_latent:
                        qml.CNOT(wires=[t, l])
                base = d * 4
                for k in range(4):
                    qml.RY(weights[base + k], wires=k)
                for l in ele_latent:
                    for t in ele_trash:
                        qml.CNOT(wires=[l, t])
            return [qml.expval(qml.PauliZ(w)) for w in ele_trash]

        self._qnodes["ele"] = qnode_ele

        # Muon circuit (4 qubits)
        mu_dev = qml.device(self.device_name, wires=4)

        @qml.qnode(mu_dev, interface="torch", diff_method="backprop")
        def qnode_mu(pt_vec, eta_vec, phi_vec, weights):
            for k in range(4):
                qml.RX(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RZ(phi_vec[:, k], wires=k)
            for d in range(1):  # depth = 1
                for t in mu_trash:
                    for l in mu_latent:
                        qml.CNOT(wires=[t, l])
                base = d * 4
                for k in range(4):
                    qml.RY(weights[base + k], wires=k)
                for l in mu_latent:
                    for t in mu_trash:
                        qml.CNOT(wires=[l, t])
            return [qml.expval(qml.PauliZ(w)) for w in mu_trash]

        self._qnodes["mu"] = qnode_mu

        # Jet circuit (10 qubits)
        jet_dev = qml.device(self.device_name, wires=10)

        @qml.qnode(jet_dev, interface="torch", diff_method="backprop")
        def qnode_jet(pt_vec, eta_vec, phi_vec, weights):
            for k in range(10):
                qml.RX(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RZ(phi_vec[:, k], wires=k)
            for d in range(4):  # depth = 4
                for t in jet_trash:
                    for l in jet_latent:
                        qml.CNOT(wires=[t, l])
                base = d * 10
                for k in range(10):
                    qml.RY(weights[base + k], wires=k)
                for l in jet_latent:
                    for t in jet_trash:
                        qml.CNOT(wires=[l, t])
            return [qml.expval(qml.PauliZ(w)) for w in jet_trash]

        self._qnodes["jet"] = qnode_jet

    def _extract_block_inputs(self, x: torch.Tensor, particles: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract pt, eta, phi for a list of particles from the 56-feature input."""
        batch = x.shape[0]
        
        pt_list = []
        eta_list = []
        phi_list = []
        
        for p in particles:
            pt_idx, eta_idx, phi_idx = PARTICLE_INDICES[p]
            pt_list.append(x[:, pt_idx])
            if eta_idx is not None:
                eta_list.append(x[:, eta_idx])
            else:
                # MET has no eta, use zeros
                eta_list.append(torch.zeros(batch, dtype=x.dtype, device=x.device))
            phi_list.append(x[:, phi_idx])
        
        pt = torch.stack(pt_list, dim=1)    # (batch, n_particles)
        eta = torch.stack(eta_list, dim=1)  # (batch, n_particles)
        phi = torch.stack(phi_list, dim=1)  # (batch, n_particles)
        
        return pt, eta, phi

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns dict of trash Z expectations per block.
          - met: (batch, 1)
          - ele: (batch, 2)
          - mu:  (batch, 2)
          - jet: (batch, 4)
        """
        orig_device = x.device
        x = x.to(torch.float64)

        out: Dict[str, torch.Tensor] = {}

        # MET
        pt_met, eta_met, phi_met = self._extract_block_inputs(x, self.met_particles)
        met_z = self._qnodes["met"](pt_met[:, 0], phi_met[:, 0], self.met_weights)
        if met_z.dim() == 1:
            met_z = met_z.unsqueeze(1)
        out["met"] = (met_z.real if met_z.is_complex() else met_z).to(orig_device)

        # Electrons
        pt_ele, eta_ele, phi_ele = self._extract_block_inputs(x, self.ele_particles)
        ele_z_list = self._qnodes["ele"](pt_ele, eta_ele, phi_ele, self.ele_weights)
        ele_z = torch.stack(ele_z_list, dim=1)
        out["ele"] = (ele_z.real if ele_z.is_complex() else ele_z).to(orig_device)

        # Muons
        pt_mu, eta_mu, phi_mu = self._extract_block_inputs(x, self.mu_particles)
        mu_z_list = self._qnodes["mu"](pt_mu, eta_mu, phi_mu, self.mu_weights)
        mu_z = torch.stack(mu_z_list, dim=1)
        out["mu"] = (mu_z.real if mu_z.is_complex() else mu_z).to(orig_device)

        # Jets
        pt_jet, eta_jet, phi_jet = self._extract_block_inputs(x, self.jet_particles)
        jet_z_list = self._qnodes["jet"](pt_jet, eta_jet, phi_jet, self.jet_weights)
        jet_z = torch.stack(jet_z_list, dim=1)
        out["jet"] = (jet_z.real if jet_z.is_complex() else jet_z).to(orig_device)

        return out


class ParticleQAEAnomalyModel(nn.Module):
    """
    Wraps the particle-wise QAE encoder and provides:
      - training loss: background-only trash cleaning
      - block scores s(x) in R^4
      - Option B anomaly score: Mahalanobis distance in block-score space

    You must call `fit_background_stats(...)` after training (or using a held-out background set)
    to set mu and precision (Sigma^{-1}).
    """

    def __init__(self, depth: int = 2, eps: float = 1e-6):
        super().__init__()
        self.encoder = ParticleQAEEncoder(depth=depth)
        self.eps = float(eps)

        # Buffers for Option B scoring
        self.register_buffer("mu", torch.zeros(4, dtype=torch.float64))
        self.register_buffer("precision", torch.eye(4, dtype=torch.float64))
        self._stats_fitted = False

    def block_scores(
        self,
        x: torch.Tensor,
        presence_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute per-block scalar scores s(x) in R^4:
          s_b = mean_{trash wires} (1 - <Z>)/2

        presence_mask (optional): shape (batch, 4) with 0/1 to gate block contribution
          order: [met, ele, mu, jet]
        If None: assumes all present (ones).
        """
        z = self.encoder(x)

        met_s = _trash_excitation_from_z(z["met"]).mean(dim=1)
        ele_s = _trash_excitation_from_z(z["ele"]).mean(dim=1)
        mu_s  = _trash_excitation_from_z(z["mu"]).mean(dim=1)
        jet_s = _trash_excitation_from_z(z["jet"]).mean(dim=1)

        s = torch.stack([met_s, ele_s, mu_s, jet_s], dim=1).to(torch.float64)

        if presence_mask is not None:
            s = s * presence_mask.to(s.dtype)

        return s

    def loss_background_only(
        self,
        x: torch.Tensor,
        presence_mask: Optional[torch.Tensor] = None,
        reduce: str = "mean",
    ) -> torch.Tensor:
        """
        Background-only QAE-style loss:
          L = mean_b s_b  (averaged over PRESENT blocks only)
        """
        s = self.block_scores(x, presence_mask=None)

        if presence_mask is not None:
            presence_mask = presence_mask.to(s.dtype).to(s.device)
            s_masked = s * presence_mask
            n_present = presence_mask.sum(dim=1).clamp(min=1.0)
            per_event = s_masked.sum(dim=1) / n_present
        else:
            per_event = s.mean(dim=1)

        if reduce == "mean":
            return per_event.mean()
        if reduce == "sum":
            return per_event.sum()
        if reduce == "none":
            return per_event
        raise ValueError(f"Unknown reduce={reduce!r}")

    @torch.no_grad()
    def fit_background_stats(
        self,
        scores_bg: torch.Tensor,
        use_full_cov: bool = True,
        ridge: float = 1e-3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit mu and precision = Sigma^{-1} from background block scores.
        """
        s = scores_bg.to(torch.float64)
        mu = s.mean(dim=0)

        xc = s - mu
        if use_full_cov:
            cov = (xc.T @ xc) / max(s.shape[0] - 1, 1)
            cov = cov + ridge * torch.eye(4, dtype=torch.float64, device=cov.device)
            precision = torch.linalg.pinv(cov)
        else:
            var = xc.pow(2).mean(dim=0) + ridge
            precision = torch.diag(1.0 / var)

        self.mu.copy_(mu)
        self.precision.copy_(precision)
        self._stats_fitted = True
        return mu, precision

    def anomaly_score(
        self,
        x: torch.Tensor,
        presence_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Option B: Mahalanobis score in R^4 block-score space:
          S(x) = 0.5 * (s - mu)^T Precision (s - mu)
        """
        if not self._stats_fitted:
            raise RuntimeError("Background stats not fitted. Call fit_background_stats(...) first.")

        s = self.block_scores(x, presence_mask=presence_mask).to(torch.float64)
        d = s - self.mu.unsqueeze(0)
        q = torch.einsum("bi,ij,bj->b", d, self.precision, d)
        return 0.5 * q
