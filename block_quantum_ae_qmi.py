#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QMI-style QAE Architecture with 4/4/4/7 qubit groupings:

Block A (4 qubits): e3, e2, j9, j8
Block B (4 qubits): e1, j7, j5, e0
Block C (4 qubits): j2, j6, μ1, μ2
Block D (7 qubits): MET, μ0, j4, j0, j3, j1, μ3

Latent/Trash splits:
  - Block A: latent=[0,1], trash=[2,3]
  - Block B: latent=[0,1], trash=[2,3]
  - Block C: latent=[0,1], trash=[2,3]
  - Block D: latent=[0,1,2,3], trash=[4,5,6]

Training objective: minimize trash qubit excitation on background.
Anomaly score: Mahalanobis distance in 4D block-score space.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import pennylane as qml


def _trash_excitation_from_z(z_expvals: torch.Tensor) -> torch.Tensor:
    """
    Convert PauliZ expectation values into trash excitation probability:
      p(|1>) = (1 - <Z>)/2
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


class QMIParticleQAEEncoder(nn.Module):
    """
    QMI-style particle QAE encoder with 4/4/4/7 qubit groupings.
    
    Blocks:
      - blockA (4 qubits): e3, e2, j9, j8
      - blockB (4 qubits): e1, j7, j5, e0
      - blockC (4 qubits): j2, j6, μ1, μ2
      - blockD (7 qubits): MET, μ0, j4, j0, j3, j1, μ3
    """
    
    def __init__(self, depth: int = 2, device: str = "default.qubit"):
        super().__init__()
        self.depth = int(depth)
        self.device_name = device
        
        # Particle assignments for each block - easily rearrangeable
        self.blockA_particles = ['e3', 'e2', 'j9', 'j8']
        self.blockB_particles = ['e1', 'j7', 'j5', 'e0']
        self.blockC_particles = ['j2', 'j6', 'mu1', 'mu2']
        self.blockD_particles = ['MET', 'mu0', 'j4', 'j0', 'j3', 'j1', 'mu3']
        
        # Latent/trash wire assignments
        self.blockA_latent = [0, 1]
        self.blockA_trash = [2, 3]
        
        self.blockB_latent = [0, 1]
        self.blockB_trash = [2, 3]
        
        self.blockC_latent = [0, 1]
        self.blockC_trash = [2, 3]
        
        self.blockD_latent = [0, 1, 2, 3]
        self.blockD_trash = [4, 5, 6]
        
        self._qnodes = {}
        self._build_circuits()
        
        # Trainable parameters (depth=1 for 4-qubit blocks, depth=4 for 7-qubit block)
        self.blockA_weights = nn.Parameter(0.01 * torch.randn(1 * 4, dtype=torch.float64))
        self.blockB_weights = nn.Parameter(0.01 * torch.randn(1 * 4, dtype=torch.float64))
        self.blockC_weights = nn.Parameter(0.01 * torch.randn(1 * 4, dtype=torch.float64))
        self.blockD_weights = nn.Parameter(0.01 * torch.randn(4 * 7, dtype=torch.float64))
    
    def _build_circuits(self):
        """Build quantum circuits for each block with all-to-all CNOTs."""
        blockA_latent = self.blockA_latent
        blockA_trash = self.blockA_trash
        blockB_latent = self.blockB_latent
        blockB_trash = self.blockB_trash
        blockC_latent = self.blockC_latent
        blockC_trash = self.blockC_trash
        blockD_latent = self.blockD_latent
        blockD_trash = self.blockD_trash
        
        import math
        pi_half = math.pi / 2
        
        # Block A circuit (4 qubits)
        devA = qml.device(self.device_name, wires=4)
        
        @qml.qnode(devA, interface="torch", diff_method="backprop")
        def qnode_blockA(pt_vec, eta_vec, phi_vec, weights):
            # First rotation layer: RZ(pi/2) RY(pt) RZ(phi)
            for k in range(4):
                qml.RX(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RZ(phi_vec[:, k], wires=k)
            # Nearest-neighbor entangling
            for k in range(3):
                qml.CNOT(wires=[k, k + 1])
            # Second rotation layer: RZ(eta) RY(pt) RZ(0)
            for k in range(4):
                qml.RZ(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RX(phi_vec[:, k], wires=k)
            # Ansatz layers
            for d in range(1):  # depth = 1
                for t in blockA_trash:
                    for l in blockA_latent:
                        qml.CNOT(wires=[t, l])
                base = d * 4
                for k in range(4):
                    qml.RY(weights[base + k], wires=k)
                for l in blockA_latent:
                    for t in blockA_trash:
                        qml.CNOT(wires=[l, t])
            return [qml.expval(qml.PauliZ(w)) for w in blockA_trash]
        
        self._qnodes["blockA"] = qnode_blockA
        
        # Block B circuit (4 qubits)
        devB = qml.device(self.device_name, wires=4)
        
        @qml.qnode(devB, interface="torch", diff_method="backprop")
        def qnode_blockB(pt_vec, eta_vec, phi_vec, weights):
            # First rotation layer: RZ(pi/2) RY(pt) RZ(phi)
            for k in range(4):
                qml.RX(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RZ(phi_vec[:, k], wires=k)
            # Nearest-neighbor entangling
            for k in range(3):
                qml.CNOT(wires=[k, k + 1])
            # Second rotation layer: RZ(eta) RY(pt) RZ(0)
            for k in range(4):
                qml.RZ(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RX(phi_vec[:, k], wires=k)
            # Ansatz layers
            for d in range(1):  # depth = 1
                for t in blockB_trash:
                    for l in blockB_latent:
                        qml.CNOT(wires=[t, l])
                base = d * 4
                for k in range(4):
                    qml.RY(weights[base + k], wires=k)
                for l in blockB_latent:
                    for t in blockB_trash:
                        qml.CNOT(wires=[l, t])
            return [qml.expval(qml.PauliZ(w)) for w in blockB_trash]
        
        self._qnodes["blockB"] = qnode_blockB
        
        # Block C circuit (4 qubits)
        devC = qml.device(self.device_name, wires=4)
        
        @qml.qnode(devC, interface="torch", diff_method="backprop")
        def qnode_blockC(pt_vec, eta_vec, phi_vec, weights):
            # First rotation layer: RZ(pi/2) RY(pt) RZ(phi)
            for k in range(4):
                qml.RX(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RZ(phi_vec[:, k], wires=k)
            # Nearest-neighbor entangling
            for k in range(3):
                qml.CNOT(wires=[k, k + 1])
            # Second rotation layer: RZ(eta) RY(pt) RZ(0)
            for k in range(4):
                qml.RZ(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RX(phi_vec[:, k], wires=k)
            # Ansatz layers
            for d in range(1):  # depth = 1
                for t in blockC_trash:
                    for l in blockC_latent:
                        qml.CNOT(wires=[t, l])
                base = d * 4
                for k in range(4):
                    qml.RY(weights[base + k], wires=k)
                for l in blockC_latent:
                    for t in blockC_trash:
                        qml.CNOT(wires=[l, t])
            return [qml.expval(qml.PauliZ(w)) for w in blockC_trash]
        
        self._qnodes["blockC"] = qnode_blockC
        
        # Block D circuit (7 qubits)
        devD = qml.device(self.device_name, wires=7)
        
        @qml.qnode(devD, interface="torch", diff_method="backprop")
        def qnode_blockD(pt_vec, eta_vec, phi_vec, weights):
            # First rotation layer: RZ(pi/2) RY(pt) RZ(phi)
            for k in range(7):
                qml.RX(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RZ(phi_vec[:, k], wires=k)
            # Nearest-neighbor entangling
            for k in range(6):
                qml.CNOT(wires=[k, k + 1])
            # Second rotation layer: RZ(eta) RY(pt) RZ(0)
            for k in range(7):
                qml.RZ(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RX(phi_vec[:, k], wires=k)
            # Ansatz layers
            for d in range(4):  # depth = 4
                for t in blockD_trash:
                    for l in blockD_latent:
                        qml.CNOT(wires=[t, l])
                base = d * 7
                for k in range(7):
                    qml.RY(weights[base + k], wires=k)
                for l in blockD_latent:
                    for t in blockD_trash:
                        qml.CNOT(wires=[l, t])
            return [qml.expval(qml.PauliZ(w)) for w in blockD_trash]
        
        self._qnodes["blockD"] = qnode_blockD
    
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
          - blockA: (batch, 2)
          - blockB: (batch, 2)
          - blockC: (batch, 2)
          - blockD: (batch, 3)
        """
        orig_device = x.device
        x = x.to(torch.float64)
        
        out: Dict[str, torch.Tensor] = {}
        
        # Block A
        pt_A, eta_A, phi_A = self._extract_block_inputs(x, self.blockA_particles)
        zA_list = self._qnodes["blockA"](pt_A, eta_A, phi_A, self.blockA_weights)
        zA = torch.stack(zA_list, dim=1)
        out["blockA"] = (zA.real if zA.is_complex() else zA).to(orig_device)
        
        # Block B
        pt_B, eta_B, phi_B = self._extract_block_inputs(x, self.blockB_particles)
        zB_list = self._qnodes["blockB"](pt_B, eta_B, phi_B, self.blockB_weights)
        zB = torch.stack(zB_list, dim=1)
        out["blockB"] = (zB.real if zB.is_complex() else zB).to(orig_device)
        
        # Block C
        pt_C, eta_C, phi_C = self._extract_block_inputs(x, self.blockC_particles)
        zC_list = self._qnodes["blockC"](pt_C, eta_C, phi_C, self.blockC_weights)
        zC = torch.stack(zC_list, dim=1)
        out["blockC"] = (zC.real if zC.is_complex() else zC).to(orig_device)
        
        # Block D
        pt_D, eta_D, phi_D = self._extract_block_inputs(x, self.blockD_particles)
        zD_list = self._qnodes["blockD"](pt_D, eta_D, phi_D, self.blockD_weights)
        zD = torch.stack(zD_list, dim=1)
        out["blockD"] = (zD.real if zD.is_complex() else zD).to(orig_device)
        
        return out


class QMIParticleQAEAnomalyModel(nn.Module):
    """
    QMI-style QAE anomaly model with 4/4/4/7 grouping.
    
    Block scores are 4D: [blockA_score, blockB_score, blockC_score, blockD_score]
    Each score is the mean trash excitation for that block.
    Anomaly score: Mahalanobis distance in 4D block-score space.
    """
    
    def __init__(self, depth: int = 2, eps: float = 1e-6):
        super().__init__()
        self.encoder = QMIParticleQAEEncoder(depth=depth)
        self.eps = float(eps)
        
        # Buffers for Mahalanobis scoring (4D)
        self.register_buffer("mu", torch.zeros(4, dtype=torch.float64))
        self.register_buffer("precision", torch.eye(4, dtype=torch.float64))
        self._stats_fitted = False
    
    def block_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-block scalar scores s(x) in R^4:
          s_b = mean_{trash wires} (1 - <Z>)/2
        """
        z = self.encoder(x)
        
        blockA_s = _trash_excitation_from_z(z["blockA"]).mean(dim=1)
        blockB_s = _trash_excitation_from_z(z["blockB"]).mean(dim=1)
        blockC_s = _trash_excitation_from_z(z["blockC"]).mean(dim=1)
        blockD_s = _trash_excitation_from_z(z["blockD"]).mean(dim=1)
        
        s = torch.stack([blockA_s, blockB_s, blockC_s, blockD_s], dim=1).to(torch.float64)
        return s
    
    def loss_background_only(
        self,
        x: torch.Tensor,
        reduce: str = "mean",
    ) -> torch.Tensor:
        """
        Background-only QAE-style loss:
          L = mean_b s_b
        """
        s = self.block_scores(x)
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
    
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mahalanobis score in R^4 block-score space:
          S(x) = 0.5 * (s - mu)^T Precision (s - mu)
        """
        if not self._stats_fitted:
            raise RuntimeError("Background stats not fitted. Call fit_background_stats(...) first.")
        
        s = self.block_scores(x).to(torch.float64)
        d = s - self.mu.unsqueeze(0)
        q = torch.einsum("bi,ij,bj->b", d, self.precision, d)
        return 0.5 * q
