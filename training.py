#!/usr/bin/env python3
"""
Training script for Quantum VAE model - PyTorch implementation.
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from quantum_encoding import LazyH5Array
from full_quantum_vae import create_particle_quantum_vae
from hybrid_quantum_vae import create_hybrid_vae
from extended_quantum_vae import create_extended_quantum_vae
from extended_quantum_vae_qmi import create_qmi_extended_quantum_vae
from block_quantum_ae import ParticleQAEAnomalyModel#, compute_presence_mask
from block_quantum_ae_qmi import QMIParticleQAEAnomalyModel


class H5Dataset(Dataset):
    """PyTorch Dataset wrapper that loads data"""
    
    def __init__(self, h5_file_path, dataset_key="Particles", max_samples=None, 
                 return_raw_for_mask=False):
        """
        Args:
            h5_file_path: Path to HDF5 file
            dataset_key: Key in HDF5 file
            max_samples: Maximum samples to load
            return_raw_for_mask: If True, also return raw (unnormalized) data for 
                                 computing presence masks
        """
        self.return_raw_for_mask = return_raw_for_mask
        
        # Use LazyH5Array to load the normalized data
        data_loader = LazyH5Array(h5_file_path, dataset_key, norm=True)
        total_samples = len(data_loader)
        
        if max_samples is not None:
            self.length = min(max_samples, total_samples)
        else:
            self.length = total_samples
        
        print(f"Loading {self.length:,} samples into memory...")
        
        if self.length == total_samples:
            data_array = np.array(data_loader[:], dtype=np.float64)
        else:
            data_array = np.array(data_loader[:self.length], dtype=np.float64)
        
        self.data = torch.tensor(data_array, dtype=torch.float64)
        
        # Also load raw data if needed for presence mask computation
        if return_raw_for_mask:
            print("  Also loading raw (unnormalized) data for presence masks...")
            raw_loader = LazyH5Array(h5_file_path, dataset_key, norm=False)
            if self.length == total_samples:
                raw_array = np.array(raw_loader[:], dtype=np.float64)
            else:
                raw_array = np.array(raw_loader[:self.length], dtype=np.float64)
            self.raw_data = torch.tensor(raw_array, dtype=torch.float64)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.length}")
        if self.return_raw_for_mask:
            return self.data[idx], self.raw_data[idx]
        return self.data[idx]


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience=15, restore_best_weights=True):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
        return self.early_stop


def plot_training_history(history, save_path="training_history.png"):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    epochs = range(1, len(history['loss']) + 1)
    
    axes[0, 0].plot(epochs, history['loss'], label='Training')
    axes[0, 0].plot(epochs, history['val_loss'], label='Validation')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs, history['reco_loss'], label='Training')
    axes[0, 1].plot(epochs, history['val_reco_loss'], label='Validation')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(epochs, history['kl_loss'], label='Training')
    axes[1, 0].plot(epochs, history['val_kl_loss'], label='Validation')
    axes[1, 0].set_title('KL Divergence Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(epochs, history['beta'], label='Beta')
    axes[1, 1].set_title('Beta Annealing')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Beta')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def train_model(dataset, model_config, training_config, output_dir="outputs", 
                model_name="best_model", device=None, model_type='vae'):
    """Train the quantum VAE model (or hybrid/extended variants)."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    
    # Split dataset
    val_split = training_config['validation_split']
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], 
                             shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], 
                           shuffle=False, num_workers=0, pin_memory=False)
    
    # Calculate steps per epoch for beta annealing
    steps_per_epoch = len(train_loader)
    model_config['steps_per_epoch'] = steps_per_epoch
    
    # Create model based on type
    if model_type == 'hybrid':
        print("Creating hybrid VAE model (classical 56->19 encoder)...")
        # Remove quantum-specific config for hybrid model
        hybrid_config = {k: v for k, v in model_config.items() 
                        if k not in ['quantum_depth', 'device']}
        model = create_hybrid_vae(**hybrid_config)
    elif model_type == 'extended':
        print("Creating extended quantum VAE model (quantum 56->32 encoder)...")
        model = create_extended_quantum_vae(**model_config)
    elif model_type == 'extended_qmi':
        print("Creating QMI extended quantum VAE model (quantum 56->32 encoder, QMI ordering)...")
        model = create_qmi_extended_quantum_vae(**model_config)
    else:
        print("Creating quantum VAE model (quantum 56->19 encoder)...")
        model = create_particle_quantum_vae(**model_config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=training_config['lr_patience'],
        verbose=True, min_lr=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=training_config['stop_patience'])
    
    history = {
        'loss': [],
        'reco_loss': [],
        'kl_loss': [],
        'val_loss': [],
        'val_reco_loss': [],
        'val_kl_loss': [],
        'beta': []
    }
    
    print(f"\nTraining Configuration:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Epochs: {training_config['epochs']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    
    # Save training metadata
    training_metadata = {
        'model_type': model_type,
        'total_samples': len(dataset),
        'training_samples': len(train_dataset),
        'validation_samples': len(val_dataset),
        'validation_split': val_split,
        'model_config': model_config,
        'training_config': training_config
    }
    metadata_path = os.path.join(output_dir, 'models', f'{model_name}_training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(training_metadata, f, indent=2)
    
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    epoch_pbar = tqdm(range(training_config['epochs']), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        # Training
        model.train()
        epoch_loss = 0.0
        epoch_reco_loss = 0.0
        epoch_kl_loss = 0.0
        n_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_config['epochs']} [Train]", 
                         leave=False, unit="batch")
        for batch in train_pbar:
            batch = batch.to(device)
            
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
            
            # Update progress bar with current loss
            train_pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'reco': f'{reco_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })
        
        avg_loss = epoch_loss / n_batches
        avg_reco = epoch_reco_loss / n_batches
        avg_kl = epoch_kl_loss / n_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_reco_loss = 0.0
        val_kl_loss = 0.0
        val_n_batches = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{training_config['epochs']} [Val]", 
                       leave=False, unit="batch")
        with torch.no_grad():
            for batch in val_pbar:
                batch = batch.to(device)
                reconstruction, z_mean, z_log_var = model(batch)
                total_loss, reco_loss, kl_loss = model.loss_function(batch, reconstruction, z_mean, z_log_var)
                val_loss += total_loss.item()
                val_reco_loss += reco_loss.item()
                val_kl_loss += kl_loss.item()
                val_n_batches += 1
                
                # Update progress bar with current loss
                val_pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'reco': f'{reco_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}'
                })
        
        avg_val_loss = val_loss / val_n_batches if val_n_batches > 0 else 0.0
        avg_val_reco = val_reco_loss / val_n_batches if val_n_batches > 0 else 0.0
        avg_val_kl = val_kl_loss / val_n_batches if val_n_batches > 0 else 0.0
        
        # Get current beta
        current_beta = model.beta.item()
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Record history
        history['loss'].append(avg_loss)
        history['reco_loss'].append(avg_reco)
        history['kl_loss'].append(avg_kl)
        history['val_loss'].append(avg_val_loss)
        history['val_reco_loss'].append(avg_val_reco)
        history['val_kl_loss'].append(avg_val_kl)
        history['beta'].append(current_beta)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_loss:.4f}',
            'val_loss': f'{avg_val_loss:.4f}',
            'beta': f'{current_beta:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Progress printing
        print(f"Epoch {epoch+1}/{training_config['epochs']}: "
              f"Loss={avg_loss:.4f} (Reco={avg_reco:.4f}, KL={avg_kl:.4f}), "
              f"Val={avg_val_loss:.4f} (Reco={avg_val_reco:.4f}, KL={avg_val_kl:.4f}), "
              f"Beta={current_beta:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if early_stopping(avg_val_loss, model):
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(output_dir, 'models', f'{model_name}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'model_config': model_config,
                'model_type': model_type
            }, best_model_path)
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'models', f'{model_name}_final.pt')
    torch.save({
        'epoch': training_config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'model_config': model_config,
        'model_type': model_type
    }, final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Plot training history
    plot_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    return model, history


def plot_qae_training_history(history, save_path="training_history.png"):
    """Plot QAE training history (simpler than VAE - just loss)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    epochs = range(1, len(history['loss']) + 1)
    
    ax.plot(epochs, history['loss'], label='Training')
    ax.plot(epochs, history['val_loss'], label='Validation')
    ax.set_title('QAE Loss (Trash Excitation)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def train_qae_model(dataset, model_config, training_config, output_dir="outputs",
                    model_name="best_model", device=None, trash_dim=4, use_presence_mask=True,
                    qae_variant='4block'):
    """Train the quantum autoencoder (QAE) model for anomaly detection.
    
    Args:
        trash_dim: 4 for averaged block scores, 9 for all individual trash wires
        use_presence_mask: If True, filter out missing particles (pt=0) from loss
        qae_variant: '4block' (original) or '2circuit' (mixed particle circuits)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Using presence mask: {use_presence_mask}")
    print(f"QAE variant: {qae_variant}")
    
    # Split dataset
    val_split = training_config['validation_split']
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'],
                             shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'],
                           shuffle=False, num_workers=0, pin_memory=False)
    
    # Create QAE model based on variant
    if qae_variant == '2circuit':
        print(f"Creating 2-circuit QAE model (10+9 qubits, 8D block scores)...")
        model = TwoCircuitQAEAnomalyModel(depth=model_config['quantum_depth'])
        # 2-circuit model doesn't use presence mask
        use_presence_mask = False
    elif qae_variant == 'qmi':
        print(f"Creating QMI QAE model (4/4/4/7 qubits, 4D block scores)...")
        model = QMIParticleQAEAnomalyModel(depth=model_config['quantum_depth'])
        # QMI model doesn't use presence mask
        use_presence_mask = False
    else:
        print(f"Creating quantum autoencoder (QAE) model with {trash_dim}D trash vectors...")
        if trash_dim == 9:
            model = ParticleQAEAnomalyModel9D(depth=model_config['quantum_depth'])
        else:
            model = ParticleQAEAnomalyModel(depth=model_config['quantum_depth'])
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=training_config['lr_patience'],
        verbose=True, min_lr=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=training_config['stop_patience'])
    
    history = {
        'loss': [],
        'val_loss': [],
    }
    
    print(f"\nTraining Configuration:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Steps per epoch: {len(train_loader)}")
    print(f"  Epochs: {training_config['epochs']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    
    # Save training metadata
    training_metadata = {
        'model_type': 'qae',
        'qae_variant': qae_variant,
        'total_samples': len(dataset),
        'training_samples': len(train_dataset),
        'validation_samples': len(val_dataset),
        'validation_split': val_split,
        'model_config': model_config,
        'training_config': training_config
    }
    metadata_path = os.path.join(output_dir, 'models', f'{model_name}_training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(training_metadata, f, indent=2)
    
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    epoch_pbar = tqdm(range(training_config['epochs']), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        # Training
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_config['epochs']} [Train]",
                         leave=False, unit="batch")
        for batch_data in train_pbar:
            # Handle presence mask if enabled
            if use_presence_mask:
                batch, raw_batch = batch_data
                batch = batch.to(device)
                raw_batch = raw_batch.to(device)
                presence_mask = compute_presence_mask(raw_batch)
            else:
                batch = batch_data.to(device)
                presence_mask = None
            
            optimizer.zero_grad()
            if qae_variant in ('2circuit', 'qmi'):
                loss = model.loss_background_only(batch, reduce="mean")
            else:
                loss = model.loss_background_only(batch, reduce="mean", presence_mask=presence_mask)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / n_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_n_batches = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{training_config['epochs']} [Val]",
                       leave=False, unit="batch")
        with torch.no_grad():
            for batch_data in val_pbar:
                # Handle presence mask if enabled
                if use_presence_mask:
                    batch, raw_batch = batch_data
                    batch = batch.to(device)
                    raw_batch = raw_batch.to(device)
                    presence_mask = compute_presence_mask(raw_batch)
                else:
                    batch = batch_data.to(device)
                    presence_mask = None
                
                if qae_variant in ('2circuit', 'qmi'):
                    loss = model.loss_background_only(batch, reduce="mean")
                else:
                    loss = model.loss_background_only(batch, reduce="mean", presence_mask=presence_mask)
                val_loss += loss.item()
                val_n_batches += 1
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / val_n_batches if val_n_batches > 0 else 0.0
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Record history
        history['loss'].append(avg_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_loss:.4f}',
            'val_loss': f'{avg_val_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Progress printing
        print(f"Epoch {epoch+1}/{training_config['epochs']}: "
              f"Loss={avg_loss:.4f}, Val={avg_val_loss:.4f}, "
              f"LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if early_stopping(avg_val_loss, model):
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(output_dir, 'models', f'{model_name}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'model_config': model_config,
                'model_type': 'qae',
                'qae_variant': qae_variant
            }, best_model_path)
    
    # Fit background stats on validation set for anomaly scoring
    print("\nFitting background statistics for Mahalanobis scoring...")
    model.eval()
    all_scores = []
    with torch.no_grad():
        for batch_data in val_loader:
            if use_presence_mask:
                batch, raw_batch = batch_data
                batch = batch.to(device)
                raw_batch = raw_batch.to(device)
                presence_mask = compute_presence_mask(raw_batch)
            else:
                batch = batch_data.to(device)
                presence_mask = None
            
            if qae_variant in ('2circuit', 'qmi'):
                scores = model.block_scores(batch)
            else:
                scores = model.block_scores(batch, presence_mask=presence_mask)
            all_scores.append(scores.cpu())
    all_scores = torch.cat(all_scores, dim=0)
    mu, precision = model.fit_background_stats(all_scores, use_full_cov=True, ridge=1e-3)
    print(f"  Background mean (mu): {mu.numpy()}")
    print(f"  Precision matrix fitted.")
    
    # Update best model checkpoint with fitted background stats
    best_model_path = os.path.join(output_dir, 'models', f'{model_name}_best.pt')
    if os.path.exists(best_model_path):
        print(f"\nUpdating best model checkpoint with background stats...")
        best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        best_checkpoint['bg_mu'] = model.mu.cpu()
        best_checkpoint['bg_precision'] = model.precision.cpu()
        torch.save(best_checkpoint, best_model_path)
        print(f"  Updated {best_model_path} with background statistics")
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'models', f'{model_name}_final.pt')
    torch.save({
        'epoch': training_config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'model_config': model_config,
        'model_type': 'qae',
        'qae_variant': qae_variant,
        'bg_mu': model.mu.cpu(),
        'bg_precision': model.precision.cpu()
    }, final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Plot training history
    plot_path = os.path.join(output_dir, 'training_history.png')
    plot_qae_training_history(history, plot_path)
    
    return model, history


def main():
    """Main training function."""

    # Hyperparameters 
    VALIDATION_SPLIT = 0.1
    STOP_PATIENCE = 15
    LR_PATIENCE = 10
    CYCLE_LENGTH = 10
    MIN_BETA = 0.1
    MAX_BETA = 0.8
    
    parser = argparse.ArgumentParser(description='Train Particle Quantum VAE model')
    
    parser.add_argument('--data', '-d', default='/global/cfs/cdirs/m2616/sagar/QiML/background_for_training.h5',
                       help='Path to training data H5 file')
    parser.add_argument('--max-samples', type=int, default=2000000,
                       help='Maximum number of samples to use for training')
    parser.add_argument('--model-type', choices=['vae', 'qae', 'hybrid', 'extended', 'extended_qmi'], default='vae',
                       help='Model type: vae (quantum 56->19), hybrid (classical 56->19), extended (quantum 56->32), extended_qmi (QMI ordering), or qae (trash qubit)')
    parser.add_argument('--hidden-dim', type=int, default=16, help='Hidden layer dimension (VAE only)')
    parser.add_argument('--latent-dim', type=int, default=3, help='Latent space dimension (VAE only)')
    parser.add_argument('--quantum-depth', type=int, default=2, help='Quantum circuit depth')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='default.qubit', help='PennyLane device name')
    parser.add_argument('--output-dir', default='outputs', help='Output directory for results')
    parser.add_argument('--model-name', default=None, help='Base name for saved models (auto-generated if not set)')
    parser.add_argument('--trash-dim', type=int, choices=[4, 9], default=4,
                       help='Trash vector dimension for QAE: 4 (averaged per block) or 9 (all individual trash wires)')
    parser.add_argument('--qae-variant', choices=['4block', '2circuit', 'qmi'], default='4block',
                       help='QAE variant: 4block (original MET/ele/mu/jet), 2circuit (mixed particle circuits), or qmi (1/4/4/10 grouping)')
    parser.add_argument('--no-presence-mask', action='store_true',
                       help='Disable presence masking for QAE (include all particles even if pt=0)')
    
    args = parser.parse_args()
    
    # Auto-generate model name if not provided
    if args.model_name is None:
        args.model_name = f"particle_quantum_{args.model_type}"
    
    print(f"Particle Quantum {args.model_type.upper()} Training (PyTorch)")
    print("=" * 50)
    
    torch.set_default_dtype(torch.float64)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print(f"Loading data from {args.data}...")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")
    
    # For QAE, also load raw data for presence mask computation (unless disabled or 2circuit/qmi)
    use_presence_mask = (args.model_type == 'qae') and (not args.no_presence_mask) and (args.qae_variant not in ('2circuit', 'qmi'))
    dataset = H5Dataset(args.data, max_samples=args.max_samples, 
                        return_raw_for_mask=use_presence_mask)
    print(f"Dataset size: {len(dataset)}")
    
    # Training configuration (shared)
    training_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'validation_split': VALIDATION_SPLIT,
        'stop_patience': STOP_PATIENCE,
        'lr_patience': LR_PATIENCE
    }
    
    if args.model_type in ['vae', 'hybrid', 'extended', 'extended_qmi']:
        # VAE-style model configuration
        model_config = {
            'input_dim': 56,
            'hidden_dim': args.hidden_dim,
            'latent_dim': args.latent_dim,
            'quantum_depth': args.quantum_depth,
            'cycle_length': CYCLE_LENGTH,
            'min_beta': MIN_BETA,
            'max_beta': MAX_BETA,
            'device': args.device
        }
        
        # Train VAE-style model
        model, history = train_model(
            dataset,
            model_config,
            training_config,
            args.output_dir,
            args.model_name,
            device=device,
            model_type=args.model_type
        )
    else:
        # QAE model configuration
        model_config = {
            'quantum_depth': args.quantum_depth,
            'device': args.device,
            'trash_dim': args.trash_dim,
            'qae_variant': args.qae_variant
        }
        
        # Train QAE model
        model, history = train_qae_model(
            dataset,
            model_config,
            training_config,
            args.output_dir,
            args.model_name,
            device=device,
            trash_dim=args.trash_dim,
            use_presence_mask=use_presence_mask,
            qae_variant=args.qae_variant
        )
    
    print("\nTraining completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
