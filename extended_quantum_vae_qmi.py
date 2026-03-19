#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Quantum VAE model with QMI-style qubit ordering.

Uses the same 1/4/4/10 grouping as block_quantum_ae_qmi.py:
  - Block 1 (1 qubit):  μ3
  - Block 2 (4 qubits): e3, e2, j9, j8
  - Block 3 (4 qubits): j2, j6, μ1, μ2
  - Block 4 (10 qubits): e1, j7, j5, e0, MET, μ0, j4, j0, j3, j1

Architecture:
- Block 1: 1 qubit -> PauliZ, PauliX, PauliY -> 3 outputs
- Block 2: 4 qubits -> PauliZ (4) -> 4 outputs  
- Block 3: 4 qubits -> PauliZ (4) + PauliX (1) -> 5 outputs
- Block 4: 10 qubits -> PauliZ (10) + PauliX (10) -> 20 outputs
Total: 3 + 4 + 5 + 20 = 32 outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from typing import Tuple


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


class QMIExtendedParticleQuantumEncoder(nn.Module):
    """
    Extended quantum encoder with QMI-style particle ordering.
    
    Uses multiple Pauli measurements (X, Y, Z) to extract more features from each qubit.
    
    Architecture:
    - Block 1 (μ3): 1 qubit -> PauliZ + PauliX + PauliY -> 3 values
    - Block 2 (e3, e2, j9, j8): 4 qubits -> PauliZ -> 4 values  
    - Block 3 (j2, j6, μ1, μ2): 4 qubits -> PauliZ + PauliX (1) -> 5 values
    - Block 4 (e1, j7, j5, e0, MET, μ0, j4, j0, j3, j1): 10 qubits -> PauliZ + PauliX -> 20 values
    Total: 32 outputs
    """
    
    def __init__(self, depth=2, device='default.qubit'):
        super().__init__()
        self.depth = depth
        self.device_name = device
        
        # Block particle assignments (same as QMI QAE)
        self.block1_particles = ['mu3']
        self.block2_particles = ['e3', 'e2', 'j9', 'j8']
        self.block3_particles = ['j2', 'j6', 'mu1', 'mu2']
        self.block4_particles = ['e1', 'j7', 'j5', 'e0', 'MET', 'mu0', 'j4', 'j0', 'j3', 'j1']
        
        # Create devices
        self.dev1 = qml.device('default.qubit', wires=1)
        self.dev2 = qml.device('default.qubit', wires=4)
        self.dev3 = qml.device('default.qubit', wires=4)
        self.dev4 = qml.device('default.qubit', wires=10)
        
        # Build circuits
        self._build_circuits()
        
        # Trainable parameters
        # Block 1: depth * 3 rotations (RX, RY, RZ per depth layer)
        self.block1_weights = nn.Parameter(0.01 * torch.randn(depth * 3, dtype=torch.float64))
        # Block 2: depth * 8 rotations (RX + RY for each of 4 qubits per layer)
        self.block2_weights = nn.Parameter(0.01 * torch.randn(depth * 8, dtype=torch.float64))
        # Block 3: depth * 8 rotations
        self.block3_weights = nn.Parameter(0.01 * torch.randn(depth * 8, dtype=torch.float64))
        # Block 4: depth * 20 rotations (10 RX + 10 RY per layer)
        self.block4_weights = nn.Parameter(0.01 * torch.randn(depth * 20, dtype=torch.float64))
    
    def _build_circuits(self):
        depth = self.depth
        
        @qml.qnode(self.dev1, interface='torch', diff_method='backprop')
        def circuit_block1(pt, eta, phi, weights):
            """Block 1: 1 qubit (μ3), returns 3 expectation values (Z, X, Y)."""
            # Feature encoding
            qml.RX(eta[:, 0], wires=0)
            qml.RY(pt[:, 0], wires=0)
            qml.RZ(phi[:, 0], wires=0)
            # Variational layers
            for d in range(depth):
                qml.RX(weights[d * 3], wires=0)
                qml.RY(weights[d * 3 + 1], wires=0)
                qml.RZ(weights[d * 3 + 2], wires=0)
            return [
                qml.expval(qml.PauliZ(0)),
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliY(0))
            ]
        
        @qml.qnode(self.dev2, interface='torch', diff_method='backprop')
        def circuit_block2(pt, eta, phi, weights):
            """Block 2: 4 qubits (e3, e2, j9, j8), returns 4 values (PauliZ only)."""
            # Feature encoding
            for k in range(4):
                qml.RX(eta[:, k], wires=k)
                qml.RY(pt[:, k], wires=k)
                qml.RZ(phi[:, k], wires=k)
            # CNOT ladder
            for k in range(3):
                qml.CNOT(wires=[k, k+1])
            # Variational layers
            for d in range(depth):
                for k in range(4):
                    qml.RX(weights[d * 8 + k], wires=k)
                    qml.RY(weights[d * 8 + 4 + k], wires=k)
                for k in range(3):
                    qml.CNOT(wires=[k, k+1])
            return [qml.expval(qml.PauliZ(k)) for k in range(4)]
        
        @qml.qnode(self.dev3, interface='torch', diff_method='backprop')
        def circuit_block3(pt, eta, phi, weights):
            """Block 3: 4 qubits (j2, j6, μ1, μ2), returns 5 values (PauliZ + 1 PauliX)."""
            # Feature encoding
            for k in range(4):
                qml.RX(eta[:, k], wires=k)
                qml.RY(pt[:, k], wires=k)
                qml.RZ(phi[:, k], wires=k)
            # CNOT ladder
            for k in range(3):
                qml.CNOT(wires=[k, k+1])
            # Variational layers
            for d in range(depth):
                for k in range(4):
                    qml.RX(weights[d * 8 + k], wires=k)
                    qml.RY(weights[d * 8 + 4 + k], wires=k)
                for k in range(3):
                    qml.CNOT(wires=[k, k+1])
            return [qml.expval(qml.PauliZ(k)) for k in range(4)] + \
                   [qml.expval(qml.PauliX(0))]
        
        @qml.qnode(self.dev4, interface='torch', diff_method='backprop')
        def circuit_block4(pt, eta, phi, weights):
            """
            Block 4: 10 qubits (e1, j7, j5, e0, MET, μ0, j4, j0, j3, j1).
            Returns 20 values: PauliZ for all 10 + PauliX for all 10.
            """
            # Feature encoding
            for k in range(10):
                qml.RX(eta[:, k], wires=k)
                qml.RY(pt[:, k], wires=k)
                qml.RZ(phi[:, k], wires=k)
            # CNOT ladder
            for k in range(9):
                qml.CNOT(wires=[k, k+1])
            # Variational layers
            for d in range(depth):
                for k in range(10):
                    qml.RX(weights[d * 20 + k], wires=k)
                for k in range(10):
                    qml.RY(weights[d * 20 + 10 + k], wires=k)
                for k in range(9):
                    qml.CNOT(wires=[k, k+1])
            return [qml.expval(qml.PauliZ(k)) for k in range(10)] + \
                   [qml.expval(qml.PauliX(k)) for k in range(10)]
        
        self.circuit_block1 = circuit_block1
        self.circuit_block2 = circuit_block2
        self.circuit_block3 = circuit_block3
        self.circuit_block4 = circuit_block4
    
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
    
    def forward(self, inputs):
        """
        Process input of shape (batch, 56) using parameter broadcasting.
        Returns (batch, 32).
        """
        device = inputs.device
        inputs = inputs.to(torch.float64)
        
        # Block 1: μ3 -> (batch, 3)
        pt1, eta1, phi1 = self._extract_block_inputs(inputs, self.block1_particles)
        block1_result = self.circuit_block1(pt1, eta1, phi1, self.block1_weights)
        block1_vals = torch.stack(block1_result, dim=1)  # (batch, 3)
        
        # Block 2: e3, e2, j9, j8 -> (batch, 4)
        pt2, eta2, phi2 = self._extract_block_inputs(inputs, self.block2_particles)
        block2_result = self.circuit_block2(pt2, eta2, phi2, self.block2_weights)
        block2_vals = torch.stack(block2_result, dim=1)  # (batch, 4)
        
        # Block 3: j2, j6, μ1, μ2 -> (batch, 5)
        pt3, eta3, phi3 = self._extract_block_inputs(inputs, self.block3_particles)
        block3_result = self.circuit_block3(pt3, eta3, phi3, self.block3_weights)
        block3_vals = torch.stack(block3_result, dim=1)  # (batch, 5)
        
        # Block 4: e1, j7, j5, e0, MET, μ0, j4, j0, j3, j1 -> (batch, 20)
        pt4, eta4, phi4 = self._extract_block_inputs(inputs, self.block4_particles)
        block4_result = self.circuit_block4(pt4, eta4, phi4, self.block4_weights)
        block4_vals = torch.stack(block4_result, dim=1)  # (batch, 20)
        
        # Ensure real values
        block1_vals = block1_vals.real if block1_vals.is_complex() else block1_vals
        block2_vals = block2_vals.real if block2_vals.is_complex() else block2_vals
        block3_vals = block3_vals.real if block3_vals.is_complex() else block3_vals
        block4_vals = block4_vals.real if block4_vals.is_complex() else block4_vals
        
        # Concatenate: (batch, 32)
        return torch.cat([block1_vals, block2_vals, block3_vals, block4_vals], dim=1).to(device)


class Sampling(nn.Module):
    """Sampling layer for VAE reparameterization trick."""
    
    def forward(self, z_mean, z_log_var):
        batch_size = z_mean.shape[0]
        dim = z_mean.shape[1]
        epsilon = torch.randn(batch_size, dim, dtype=z_mean.dtype, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class QMIExtendedQuantumEncoderFull(nn.Module):
    """Full encoder with QMI-style extended quantum layer (56->32) and dense layers."""
    
    def __init__(self, input_dim=56, hidden_dim=16, latent_dim=3, 
                 quantum_depth=2, l2_factor=1e-3, device='default.qubit'):
        super().__init__()
        self.quantum_encoder = QMIExtendedParticleQuantumEncoder(depth=quantum_depth, device=device)
        # 32 quantum outputs -> hidden_dim
        self.dense_hidden = nn.Linear(32, hidden_dim)
        self.z_mean = nn.Linear(hidden_dim, latent_dim)
        self.z_log_var = nn.Linear(hidden_dim, latent_dim)
        self.sampling = Sampling()
        
        # Initialize weights
        nn.init.kaiming_normal_(self.dense_hidden.weight)
        nn.init.kaiming_normal_(self.z_mean.weight)
        nn.init.zeros_(self.z_mean.bias)
        nn.init.zeros_(self.z_log_var.weight)
        nn.init.zeros_(self.z_log_var.bias)
        
        # Convert to float64 for PennyLane compatibility
        self.dense_hidden = self.dense_hidden.double()
        self.z_mean = self.z_mean.double()
        self.z_log_var = self.z_log_var.double()
    
    def forward(self, x):
        x = self.quantum_encoder(x)
        x = F.relu(self.dense_hidden(x))
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    """Decoder for VAE with extended hidden dimension."""
    
    def __init__(self, input_dim=56, h_dim_1=32, h_dim_2=16, latent_dim=3):
        super().__init__()
        self.dense1 = nn.Linear(latent_dim, h_dim_2)
        self.dense2 = nn.Linear(h_dim_2, h_dim_1)
        self.output = nn.Linear(h_dim_1, input_dim)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.dense1.weight)
        nn.init.kaiming_normal_(self.dense2.weight)
        nn.init.kaiming_normal_(self.output.weight)
        nn.init.zeros_(self.dense1.bias)
        nn.init.zeros_(self.dense2.bias)
        nn.init.zeros_(self.output.bias)
        
        # Convert to float64
        self.dense1 = self.dense1.double()
        self.dense2 = self.dense2.double()
        self.output = self.output.double()
    
    def forward(self, z):
        x = F.relu(self.dense1(z))
        x = F.relu(self.dense2(x))
        return self.output(x)


class QMIExtendedQuantumVAE(nn.Module):
    """Variational Autoencoder with QMI-style extended quantum layers (56->32)."""
    
    def __init__(self, encoder, decoder, steps_per_epoch=3125, 
                 cycle_length=10, min_beta=0.1, max_beta=0.8):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.steps_per_epoch = steps_per_epoch
        self.cycle_length = float(cycle_length)
        self.min_beta = float(min_beta)
        self.max_beta = float(max_beta)
        self.register_buffer('beta', torch.tensor(min_beta, dtype=torch.float64))
        self.step = 0
    
    def cyclical_annealing_beta(self):
        """Cyclical beta annealing schedule."""
        epoch = float(self.step) / self.steps_per_epoch
        cycle = np.floor(1.0 + epoch / self.cycle_length)
        x = np.abs(epoch / self.cycle_length - cycle + 1)
        beta = self.min_beta + (self.max_beta - self.min_beta) * min(x, 1.0)
        self.beta.fill_(beta)
    
    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var
    
    def loss_function(self, x, reconstruction, z_mean, z_log_var):
        """Compute VAE loss with cyclical beta annealing."""
        self.cyclical_annealing_beta()
        
        # Masked reconstruction loss
        mask = (x != 0).float()
        reconstruction_loss = F.mse_loss(mask * x, mask * reconstruction, reduction='mean')
        reconstruction_loss = reconstruction_loss * (1 - self.beta)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
        kl_loss = kl_loss.mean() * self.beta
        
        total_loss = reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss


def create_qmi_extended_quantum_encoder(input_dim=56, hidden_dim=16, latent_dim=3, 
                                         quantum_depth=2, l2_factor=1e-3, device='default.qubit'):
    """Create encoder with QMI-style extended quantum encoding (56->32)."""
    return QMIExtendedQuantumEncoderFull(
        input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
        quantum_depth=quantum_depth, l2_factor=l2_factor, device=device
    )


def create_decoder(input_dim=56, h_dim_1=32, h_dim_2=16, latent_dim=3):
    """Create decoder for VAE with extended hidden dimension."""
    return Decoder(
        input_dim=input_dim, h_dim_1=h_dim_1, h_dim_2=h_dim_2, latent_dim=latent_dim
    )


def create_qmi_extended_quantum_vae(input_dim=56, hidden_dim=16, latent_dim=3,
                                     quantum_depth=2, steps_per_epoch=3125, 
                                     cycle_length=10, min_beta=0.1, max_beta=0.8,
                                     device='default.qubit'):
    """
    Create complete VAE with QMI-style extended quantum encoding (56->32).
    
    Uses the same particle grouping as QMI QAE:
    - Block 1 (μ3): PauliZ + PauliX + PauliY (3 outputs)
    - Block 2 (e3, e2, j9, j8): PauliZ + PauliX (8 outputs)
    - Block 3 (j2, j6, μ1, μ2): PauliZ + PauliX (8 outputs)
    - Block 4 (e1, j7, j5, e0, MET, μ0, j4, j0, j3, j1): PauliZ + PauliX for leading 3 (13 outputs)
    Total: 32 outputs
    """
    encoder = create_qmi_extended_quantum_encoder(
        input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
        quantum_depth=quantum_depth, device=device
    )
    
    decoder = create_decoder(
        input_dim=input_dim, h_dim_1=32, h_dim_2=hidden_dim, latent_dim=latent_dim
    )
    
    vae = QMIExtendedQuantumVAE(
        encoder, decoder, steps_per_epoch=steps_per_epoch,
        cycle_length=cycle_length, min_beta=min_beta, max_beta=max_beta
    )
    
    return vae
