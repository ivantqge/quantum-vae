#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Quantum VAE model - Classical encoder layer replacing the quantum 56->19 layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalParticleEncoder(nn.Module):
    """
    Classical encoder layer that replaces the quantum layer.
    Maps 56 input features to 19 output features using dense layers.
    
    This provides a baseline comparison against the quantum encoder,
    matching the same dimensionality reduction (56 -> 19).
    """
    
    def __init__(self, input_dim=56, output_dim=19, hidden_dim=32):
        super().__init__()
        
        # Two-layer MLP to match the expressive power of the quantum circuits
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Output bounded in [-1, 1] like quantum expectation values
        )
        
        # Initialize weights
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Convert to float64 for consistency with quantum model
        self.encoder = self.encoder.double()
    
    def forward(self, inputs):
        """
        Process input of shape (batch, 56) -> (batch, 19)
        """
        return self.encoder(inputs.to(torch.float64))


class Sampling(nn.Module):
    """Sampling layer for VAE reparameterization trick."""
    
    def forward(self, z_mean, z_log_var):
        batch_size = z_mean.shape[0]
        dim = z_mean.shape[1]
        epsilon = torch.randn(batch_size, dim, dtype=z_mean.dtype, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class HybridEncoderFull(nn.Module):
    """Full encoder with classical layer and dense layers."""
    
    def __init__(self, input_dim=56, hidden_dim=16, latent_dim=3, 
                 encoder_hidden_dim=32, l2_factor=1e-3):
        super().__init__()
        self.classical_encoder = ClassicalParticleEncoder(
            input_dim=input_dim, output_dim=19, hidden_dim=encoder_hidden_dim
        )
        self.dense_hidden = nn.Linear(19, hidden_dim)
        self.z_mean = nn.Linear(hidden_dim, latent_dim)
        self.z_log_var = nn.Linear(hidden_dim, latent_dim)
        self.sampling = Sampling()
        
        # Initialize weights
        nn.init.kaiming_normal_(self.dense_hidden.weight)
        nn.init.kaiming_normal_(self.z_mean.weight)
        nn.init.zeros_(self.z_mean.bias)
        nn.init.zeros_(self.z_log_var.weight)
        nn.init.zeros_(self.z_log_var.bias)
        
        # Convert to float64
        self.dense_hidden = self.dense_hidden.double()
        self.z_mean = self.z_mean.double()
        self.z_log_var = self.z_log_var.double()
    
    def forward(self, x):
        x = self.classical_encoder(x)
        x = F.relu(self.dense_hidden(x))
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    """Decoder for VAE."""
    
    def __init__(self, input_dim=56, h_dim_1=19, h_dim_2=16, latent_dim=3):
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


class HybridVAE(nn.Module):
    """Variational Autoencoder with classical encoder (replacing quantum layer)."""
    
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


def create_hybrid_encoder(input_dim=56, hidden_dim=16, latent_dim=3, 
                          encoder_hidden_dim=32, l2_factor=1e-3):
    """Create encoder with classical (non-quantum) encoding layer."""
    return HybridEncoderFull(
        input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
        encoder_hidden_dim=encoder_hidden_dim, l2_factor=l2_factor
    )


def create_decoder(input_dim=56, h_dim_1=19, h_dim_2=16, latent_dim=3):
    """Create decoder for VAE."""
    return Decoder(
        input_dim=input_dim, h_dim_1=h_dim_1, h_dim_2=h_dim_2, latent_dim=latent_dim
    )


def create_hybrid_vae(input_dim=56, hidden_dim=16, latent_dim=3,
                      encoder_hidden_dim=32, steps_per_epoch=3125, 
                      cycle_length=10, min_beta=0.1, max_beta=0.8):
    """
    Create complete VAE with classical encoder layer.
    
    This model replaces the quantum 56->19 layer with a classical MLP,
    providing a baseline for comparison against the full quantum VAE.
    """
    encoder = create_hybrid_encoder(
        input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
        encoder_hidden_dim=encoder_hidden_dim
    )
    
    decoder = create_decoder(
        input_dim=input_dim, h_dim_1=19, h_dim_2=hidden_dim, latent_dim=latent_dim
    )
    
    vae = HybridVAE(
        encoder, decoder, steps_per_epoch=steps_per_epoch,
        cycle_length=cycle_length, min_beta=min_beta, max_beta=max_beta
    )
    
    return vae
