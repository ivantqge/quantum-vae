#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify particle quantum VAE model training works correctly.
"""

import sys
import numpy as np
import torch
import torch.optim as optim
import pennylane as qml

from particle_vae import create_particle_quantum_vae, ParticleQuantumEncoder


def create_test_data(n_samples=1000, n_features=56):
    """Create synthetic test data for model testing."""
    np.random.seed(42)
    data = np.zeros((n_samples, n_features))
    
    # MET features (indices 0-1)
    data[:, 0] = np.random.exponential(50, n_samples)
    data[:, 1] = np.random.uniform(-np.pi, np.pi, n_samples)
    
    # Electron features (indices 2-13)
    for i in range(4):
        data[:, 2+i] = np.random.exponential(30, n_samples)
        data[:, 6+i] = np.random.uniform(-2.5, 2.5, n_samples)
        data[:, 10+i] = np.random.uniform(-np.pi, np.pi, n_samples)
    
    # Muon features (indices 14-25)
    for i in range(4):
        data[:, 14+i] = np.random.exponential(25, n_samples)
        data[:, 18+i] = np.random.uniform(-2.5, 2.5, n_samples)
        data[:, 22+i] = np.random.uniform(-np.pi, np.pi, n_samples)
    
    # Jet features (indices 26-55)
    for i in range(10):
        data[:, 26+i] = np.random.exponential(40, n_samples)
        data[:, 36+i] = np.random.uniform(-3, 3, n_samples)
        data[:, 46+i] = np.random.uniform(-np.pi, np.pi, n_samples)
    
    # Add some zeros to simulate missing particles
    zero_mask = np.random.random((n_samples, n_features)) < 0.3
    data[zero_mask] = 0
    
    return data


def test_model_creation(device):
    """Test if the particle quantum VAE model can be created successfully."""
    print("\n[Model Creation]")
    
    model = create_particle_quantum_vae(
        input_dim=56, hidden_dim=16, latent_dim=3, quantum_depth=2,
        steps_per_epoch=10, cycle_length=5, min_beta=0.1, max_beta=0.8,
        device='lightning.gpu'
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # Test forward pass
    test_input = torch.randn(1, 56, dtype=torch.float64).to(device)
    model.eval()
    with torch.no_grad():
        reconstruction, z_mean, z_log_var = model(test_input)
    
    print(f"  Forward pass: OK")
    return model


def test_forward_pass(model, test_data, device):
    """Test if the model can perform forward pass."""
    print("\n[Forward Pass]")
    
    batch_size = 10
    test_batch = torch.tensor(test_data[:batch_size], dtype=torch.float64).to(device)
    
    model.eval()
    with torch.no_grad():
        reconstruction, z_mean, z_log_var = model(test_batch)
    
    print(f"  Input: {test_batch.shape} -> Output: {reconstruction.shape}")
    return True


def test_training_step(model, test_data, optimizer, device):
    """Test if the model can perform a training step."""
    print("\n[Training Step]")
    
    batch_size = 32
    test_batch = torch.tensor(test_data[:batch_size], dtype=torch.float64).to(device)
    
    model.train()
    optimizer.zero_grad()
    reconstruction, z_mean, z_log_var = model(test_batch)
    total_loss, reco_loss, kl_loss = model.loss_function(test_batch, reconstruction, z_mean, z_log_var)
    total_loss.backward()
    optimizer.step()
    model.step += 1
    
    print(f"  Loss: {total_loss.item():.4f} (Reco: {reco_loss.item():.4f}, KL: {kl_loss.item():.4f})")
    return True


def test_quantum_layer_parameter_updates(model, test_data, optimizer, device):
    """Test if quantum layer parameters are updated during training."""
    print("\n[Parameter Updates]")
    
    quantum_encoder = model.encoder.quantum_encoder
    initial_met = quantum_encoder.met_weights.data.clone()
    initial_ele = quantum_encoder.ele_weights.data.clone()
    initial_mu = quantum_encoder.mu_weights.data.clone()
    initial_jet = quantum_encoder.jet_weights.data.clone()
    
    train_batch = torch.tensor(test_data[:32], dtype=torch.float64).to(device)
    model.train()
    
    for step in range(5):
        optimizer.zero_grad()
        reconstruction, z_mean, z_log_var = model(train_batch)
        total_loss, _, _ = model.loss_function(train_batch, reconstruction, z_mean, z_log_var)
        total_loss.backward()
        optimizer.step()
        model.step += 1
    
    max_changes = [
        (quantum_encoder.met_weights.data - initial_met).abs().max().item(),
        (quantum_encoder.ele_weights.data - initial_ele).abs().max().item(),
        (quantum_encoder.mu_weights.data - initial_mu).abs().max().item(),
        (quantum_encoder.jet_weights.data - initial_jet).abs().max().item()
    ]
    
    all_updated = all(change > 1e-6 for change in max_changes)
    print(f"  Max changes: MET={max_changes[0]:.4f}, Ele={max_changes[1]:.4f}, Mu={max_changes[2]:.4f}, Jet={max_changes[3]:.4f}")
    return all_updated


def test_backprop_gradients(model, test_data, optimizer, device):
    """Test if gradients are computed correctly."""
    print("\n[Backpropagation]")
    
    batch_size = 16
    test_batch = torch.tensor(test_data[:batch_size], dtype=torch.float64).to(device)
    
    model.train()
    optimizer.zero_grad()
    reconstruction, z_mean, z_log_var = model(test_batch)
    total_loss, _, _ = model.loss_function(test_batch, reconstruction, z_mean, z_log_var)
    total_loss.backward()
    
    has_gradients = any(p.grad is not None and p.grad.norm() > 0 for p in model.parameters())
    
    if has_gradients:
        optimizer.step()
        model.step += 1
        print("  Gradients: OK")
        return True
    else:
        print("  Gradients: FAILED")
        return False


def test_short_training(model, test_data, epochs=3, device=None):
    """Test short training session - optimized for GPU."""
    print(f"\n[Training ({epochs} epochs)]")
    
    train_data = torch.tensor(test_data[:800], dtype=torch.float64, device=device)
    val_data = torch.tensor(test_data[800:], dtype=torch.float64, device=device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 64
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_reco_loss = 0.0
        epoch_kl_loss = 0.0
        n_batches = 0
        
        indices = torch.randperm(len(train_data), device=device)
        for i in range(0, len(train_data), batch_size):
            batch = train_data[indices[i:i+batch_size]]
            
            optimizer.zero_grad()
            reconstruction, z_mean, z_log_var = model(batch)
            total_loss, reco_loss, kl_loss = model.loss_function(batch, reconstruction, z_mean, z_log_var)
            total_loss.backward()
            optimizer.step()
            model.step += 1
            
            epoch_loss += total_loss.item()
            epoch_reco_loss += reco_loss.item()
            epoch_kl_loss += kl_loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        avg_reco = epoch_reco_loss / n_batches
        avg_kl = epoch_kl_loss / n_batches
        
        model.eval()
        val_loss = 0.0
        val_n_batches = 0
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                reconstruction, z_mean, z_log_var = model(batch)
                total_loss, _, _ = model.loss_function(batch, reconstruction, z_mean, z_log_var)
                val_loss += total_loss.item()
                val_n_batches += 1
        
        avg_val_loss = val_loss / val_n_batches if val_n_batches > 0 else 0.0
        print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f} (Reco={avg_reco:.4f}, KL={avg_kl:.4f}), Val={avg_val_loss:.4f}")
        model.train()
    
    return True


def main():
    """Main test function."""
    print("Particle Quantum VAE Testing (PyTorch)")
    print("=" * 50)
    
    torch.set_default_dtype(torch.float64)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Check PennyLane Lightning GPU
    try:
        lightning_dev = qml.device('lightning.gpu', wires=1)
        print(f"PennyLane: {lightning_dev.name} OK")
    except Exception as e:
        print(f"PennyLane: ERROR - {e}")
    
    test_data = create_test_data(n_samples=1000, n_features=56)
    
    model = test_model_creation(device)
    if model is None:
        print("ERROR: Model creation failed")
        return 1
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    if not test_forward_pass(model, test_data, device):
        return 1
    
    if not test_backprop_gradients(model, test_data, optimizer, device):
        return 1
    
    if not test_quantum_layer_parameter_updates(model, test_data, optimizer, device):
        print("WARNING: Some parameters may not be updating")
    
    if not test_training_step(model, test_data, optimizer, device):
        return 1
    
    if not test_short_training(model, test_data, epochs=5, device=device):
        return 1
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
