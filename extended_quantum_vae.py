#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Quantum VAE model - 56->32 quantum layer using PauliX, PauliY, PauliZ measurements.

Architecture:
- MeT: 1 qubit -> PauliZ, PauliX, PauliY -> 3 outputs
- Electrons: 4 qubits -> PauliZ (4) + PauliX (4) -> 8 outputs  
- Muons: 4 qubits -> PauliZ (4) + PauliX (4) -> 8 outputs
- Jets: 10 qubits -> PauliZ (10) + PauliX (first 3) -> 13 outputs
Total: 3 + 8 + 8 + 13 = 32 outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np


class ExtendedParticleQuantumEncoder(nn.Module):
    """
    Extended quantum encoder layer for particle physics data.
    Uses multiple Pauli measurements (X, Y, Z) to extract more features from each qubit.
    
    Architecture overview:
    - MeT (2 features) -> 1 qubit -> PauliZ, PauliX, PauliY -> 3 values
    - Electrons (12 features: 4 * 3) -> 4 qubits -> PauliZ + PauliX -> 8 values  
    - Muons (12 features: 4 * 3) -> 4 qubits -> PauliZ + PauliX -> 8 values
    - Jets (30 features: 10 * 3) -> 10 qubits -> PauliZ + PauliX (first 3) -> 13 values
    Total: 32 outputs
    """
    
    def __init__(self, depth=2, device='default.qubit'):
        super().__init__()
        self.depth = depth
        self.device_name = device
        
        # Use default.qubit for parameter broadcasting support
        self.met_dev = qml.device('default.qubit', wires=1)
        self.ele_dev = qml.device('default.qubit', wires=4)
        self.mu_dev = qml.device('default.qubit', wires=4)
        self.jet_dev = qml.device('default.qubit', wires=10)

        @qml.qnode(self.met_dev, interface='torch', diff_method='backprop')
        def met_circuit(pt, phi, weights):
            """MeT circuit: 1 qubit, returns 3 expectation values (Z, X, Y)."""
            # pt, phi have shape (batch,)
            qml.RY(pt, wires=0)
            qml.RZ(phi, wires=0)
            for d in range(depth):
                qml.RX(weights[d * 3], wires=0)
                qml.RY(weights[d * 3 + 1], wires=0)
                qml.RZ(weights[d * 3 + 2], wires=0)
            # Return all three Pauli expectations
            return [
                qml.expval(qml.PauliZ(0)),
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliY(0))
            ]
        
        @qml.qnode(self.ele_dev, interface='torch', diff_method='backprop')
        def ele_circuit(pt_vec, eta_vec, phi_vec, weights):
            """Electron circuit: 4 qubits, returns 8 values (PauliZ + PauliX for each)."""
            # pt_vec, eta_vec, phi_vec have shape (batch, 4)
            for k in range(4):
                qml.RX(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RZ(phi_vec[:, k], wires=k)
            for k in range(3):
                qml.CNOT(wires=[k, k+1])
            for d in range(depth):
                for k in range(4):
                    qml.RX(weights[d * 8 + k], wires=k)
                    qml.RY(weights[d * 8 + 4 + k], wires=k)
                for k in range(3):
                    qml.CNOT(wires=[k, k+1])
            # Return PauliZ and PauliX for all 4 qubits
            return [qml.expval(qml.PauliZ(k)) for k in range(4)] + \
                   [qml.expval(qml.PauliX(k)) for k in range(4)]
        
        @qml.qnode(self.mu_dev, interface='torch', diff_method='backprop')
        def mu_circuit(pt_vec, eta_vec, phi_vec, weights):
            """Muon circuit: 4 qubits, returns 8 values (PauliZ + PauliX for each)."""
            # pt_vec, eta_vec, phi_vec have shape (batch, 4)
            for k in range(4):
                qml.RX(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RZ(phi_vec[:, k], wires=k)
            for k in range(3):
                qml.CNOT(wires=[k, k+1])
            for d in range(depth):
                for k in range(4):
                    qml.RX(weights[d * 8 + k], wires=k)
                    qml.RY(weights[d * 8 + 4 + k], wires=k)
                for k in range(3):
                    qml.CNOT(wires=[k, k+1])
            # Return PauliZ and PauliX for all 4 qubits
            return [qml.expval(qml.PauliZ(k)) for k in range(4)] + \
                   [qml.expval(qml.PauliX(k)) for k in range(4)]
        
        @qml.qnode(self.jet_dev, interface='torch', diff_method='backprop')
        def jet_circuit(pt_vec, eta_vec, phi_vec, weights):
            """
            Jet circuit: 10 qubits, returns 13 values.
            - PauliZ for all 10 qubits
            - PauliX for first 3 qubits (leading jets are most important)
            """
            # pt_vec, eta_vec, phi_vec have shape (batch, 10)
            for k in range(10):
                qml.RX(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RZ(phi_vec[:, k], wires=k)
            for k in range(9):
                qml.CNOT(wires=[k, k+1])
            for d in range(depth):
                for k in range(10):
                    qml.RX(weights[d * 13 + k], wires=k)
                # Additional rotations for first 3 qubits (more expressivity)
                for k in range(3):
                    qml.RY(weights[d * 13 + 10 + k], wires=k)
                for k in range(9):
                    qml.CNOT(wires=[k, k+1])
            # Return PauliZ for all + PauliX for first 3
            return [qml.expval(qml.PauliZ(k)) for k in range(10)] + \
                   [qml.expval(qml.PauliX(k)) for k in range(3)]
        
        self.met_circuit = met_circuit
        self.ele_circuit = ele_circuit
        self.mu_circuit = mu_circuit
        self.jet_circuit = jet_circuit
        
        # Trainable parameters for the rotation weights
        # MeT: depth * 3 rotations (RX, RY, RZ per depth layer)
        self.met_weights = nn.Parameter(0.01 * torch.randn(depth * 3, dtype=torch.float64))
        # Electrons: depth * 8 rotations (RX + RY for each of 4 qubits per layer)
        self.ele_weights = nn.Parameter(0.01 * torch.randn(depth * 8, dtype=torch.float64))
        # Muons: depth * 8 rotations
        self.mu_weights = nn.Parameter(0.01 * torch.randn(depth * 8, dtype=torch.float64))
        # Jets: depth * 13 rotations (10 RX + 3 RY per layer)
        self.jet_weights = nn.Parameter(0.01 * torch.randn(depth * 13, dtype=torch.float64))
    
    def forward(self, inputs):
        """
        Process input of shape (batch, 56) using parameter broadcasting.
        Returns (batch, 32).
        
        Input layout:
        - [0:2]: MeT (pt, phi)
        - [2:6]: Electron pt, [6:10]: eta, [10:14]: phi
        - [14:18]: Muon pt, [18:22]: eta, [22:26]: phi
        - [26:36]: Jet pt, [36:46]: eta, [46:56]: phi
        """
        device = inputs.device
        inputs = inputs.to(torch.float64)
        
        # MeT: (batch,) inputs -> (batch, 3) output
        met_result = self.met_circuit(inputs[:, 0], inputs[:, 1], self.met_weights)
        met_vals = torch.stack(met_result, dim=1)  # (batch, 3)
        
        # Electrons: (batch, 4) inputs -> (batch, 8) output
        ele_result = self.ele_circuit(
            inputs[:, 2:6], inputs[:, 6:10], inputs[:, 10:14], self.ele_weights
        )
        ele_vals = torch.stack(ele_result, dim=1)  # (batch, 8)
        
        # Muons: (batch, 4) inputs -> (batch, 8) output
        mu_result = self.mu_circuit(
            inputs[:, 14:18], inputs[:, 18:22], inputs[:, 22:26], self.mu_weights
        )
        mu_vals = torch.stack(mu_result, dim=1)  # (batch, 8)
        
        # Jets: (batch, 10) inputs -> (batch, 13) output
        jet_result = self.jet_circuit(
            inputs[:, 26:36], inputs[:, 36:46], inputs[:, 46:56], self.jet_weights
        )
        jet_vals = torch.stack(jet_result, dim=1)  # (batch, 13)
        
        # Ensure real values
        met_vals = met_vals.real if met_vals.is_complex() else met_vals
        ele_vals = ele_vals.real if ele_vals.is_complex() else ele_vals
        mu_vals = mu_vals.real if mu_vals.is_complex() else mu_vals
        jet_vals = jet_vals.real if jet_vals.is_complex() else jet_vals
        
        # Concatenate: (batch, 32)
        return torch.cat([met_vals, ele_vals, mu_vals, jet_vals], dim=1).to(device)


class Sampling(nn.Module):
    """Sampling layer for VAE reparameterization trick."""
    
    def forward(self, z_mean, z_log_var):
        batch_size = z_mean.shape[0]
        dim = z_mean.shape[1]
        epsilon = torch.randn(batch_size, dim, dtype=z_mean.dtype, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class ExtendedQuantumEncoderFull(nn.Module):
    """Full encoder with extended quantum layer (56->32) and dense layers."""
    
    def __init__(self, input_dim=56, hidden_dim=16, latent_dim=3, 
                 quantum_depth=2, l2_factor=1e-3, device='default.qubit'):
        super().__init__()
        self.quantum_encoder = ExtendedParticleQuantumEncoder(depth=quantum_depth, device=device)
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


class ExtendedQuantumVAE(nn.Module):
    """Variational Autoencoder with extended quantum layers (56->32)."""
    
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


def create_extended_quantum_encoder(input_dim=56, hidden_dim=16, latent_dim=3, 
                                     quantum_depth=2, l2_factor=1e-3, device='default.qubit'):
    """Create encoder with extended quantum encoding (56->32)."""
    return ExtendedQuantumEncoderFull(
        input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
        quantum_depth=quantum_depth, l2_factor=l2_factor, device=device
    )


def create_decoder(input_dim=56, h_dim_1=32, h_dim_2=16, latent_dim=3):
    """Create decoder for VAE with extended hidden dimension."""
    return Decoder(
        input_dim=input_dim, h_dim_1=h_dim_1, h_dim_2=h_dim_2, latent_dim=latent_dim
    )


def create_extended_quantum_vae(input_dim=56, hidden_dim=16, latent_dim=3,
                                 quantum_depth=2, steps_per_epoch=3125, 
                                 cycle_length=10, min_beta=0.1, max_beta=0.8,
                                 device='default.qubit'):
    """
    Create complete VAE with extended quantum encoding (56->32).
    
    Uses multiple Pauli measurements (X, Y, Z) to extract richer quantum features:
    - MeT: PauliZ + PauliX + PauliY (3 outputs)
    - Electrons: PauliZ + PauliX (8 outputs)
    - Muons: PauliZ + PauliX (8 outputs)
    - Jets: PauliZ + PauliX for leading 3 jets (13 outputs)
    Total: 32 outputs (vs 19 in the original model)
    """
    encoder = create_extended_quantum_encoder(
        input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
        quantum_depth=quantum_depth, device=device
    )
    
    decoder = create_decoder(
        input_dim=input_dim, h_dim_1=32, h_dim_2=hidden_dim, latent_dim=latent_dim
    )
    
    vae = ExtendedQuantumVAE(
        encoder, decoder, steps_per_epoch=steps_per_epoch,
        cycle_length=cycle_length, min_beta=min_beta, max_beta=max_beta
    )
    
    return vae
