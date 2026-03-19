#!/usr/bin/env python3
"""
Visualization script for VAE reconstruction quality.
Analyzes reconstruction MSE for background and signal data.
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from quantum_encoding import LazyH5Array
from full_quantum_vae import create_particle_quantum_vae


# Signal labels
signal_labels = ["Ato4l", "leptoquark", "hChToTauNu", "hToTauTau"]

# Feature labels for the 56D input
feature_labels = [
    "MET_pt", "MET_phi",
    "e1_pt", "e2_pt", "e3_pt", "e4_pt",
    "e1_eta", "e2_eta", "e3_eta", "e4_eta",
    "e1_phi", "e2_phi", "e3_phi", "e4_phi",
    "μ1_pt", "μ2_pt", "μ3_pt", "μ4_pt",
    "μ1_eta", "μ2_eta", "μ3_eta", "μ4_eta",
    "μ1_phi", "μ2_phi", "μ3_phi", "μ4_phi",
    "j1_pt", "j2_pt", "j3_pt", "j4_pt", "j5_pt", "j6_pt", "j7_pt", "j8_pt", "j9_pt", "j10_pt",
    "j1_eta", "j2_eta", "j3_eta", "j4_eta", "j5_eta", "j6_eta", "j7_eta", "j8_eta", "j9_eta", "j10_eta",
    "j1_phi", "j2_phi", "j3_phi", "j4_phi", "j5_phi", "j6_phi", "j7_phi", "j8_phi", "j9_phi", "j10_phi",
]

# Group indices for per-particle-type analysis
FEATURE_GROUPS = {
    'MET': slice(0, 2),
    'Electrons': slice(2, 14),
    'Muons': slice(14, 26),
    'Jets': slice(26, 56)
}


def load_data(h5_path, start_idx=0, max_samples=None):
    """Load data from H5 file using LazyH5Array."""
    data_loader = LazyH5Array(h5_path)
    total_samples = len(data_loader)
    
    if max_samples is not None:
        end_idx = min(start_idx + max_samples, total_samples)
    else:
        end_idx = total_samples
    
    print(f"  Loading samples {start_idx:,} to {end_idx:,} ({end_idx - start_idx:,} samples)")
    return np.array(data_loader[start_idx:end_idx], dtype=np.float64)


def get_reconstruction(model, data, device, batch_size=1024):
    """
    Get reconstruction and compute MSE for input data.
    
    Args:
        model: The QuantumVAE model
        data: Input data array (N, 56)
        device: torch device
        batch_size: Batch size for processing
    
    Returns:
        reconstructions: numpy array of shape (N, 56)
        mse_per_sample: numpy array of shape (N,) - MSE for each sample
        mse_per_feature: numpy array of shape (N, 56) - squared error per feature
    """
    model.eval()
    
    n_samples = data.shape[0]
    reconstructions = []
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = data[start_idx:end_idx]
            
            if isinstance(batch, np.ndarray):
                batch_tensor = torch.tensor(batch, dtype=torch.float64, device=device)
            else:
                batch_tensor = batch.to(device)
            
            # Get reconstruction through full VAE
            reconstruction, z_mean, z_log_var = model(batch_tensor)
            reconstructions.append(reconstruction.cpu().numpy())
    
    reconstructions = np.concatenate(reconstructions, axis=0)
    
    # Compute MSE per feature (squared error)
    mse_per_feature = (data - reconstructions) ** 2
    
    # Compute MSE per sample (mean across features)
    mse_per_sample = np.mean(mse_per_feature, axis=1)
    
    return reconstructions, mse_per_sample, mse_per_feature


def get_masked_mse(data, reconstructions):
    """
    Compute MSE only for non-zero (present) particles.
    
    Args:
        data: Original input (N, 56)
        reconstructions: Reconstructed output (N, 56)
    
    Returns:
        masked_mse_per_sample: MSE computed only over non-zero entries
    """
    mask = (data != 0).astype(np.float64)
    squared_error = (data - reconstructions) ** 2
    
    # Sum of squared errors for non-zero entries, divided by count of non-zero entries
    masked_mse = np.sum(squared_error * mask, axis=1) / np.maximum(np.sum(mask, axis=1), 1)
    
    return masked_mse


def plot_mse_distributions(mse_dict, title, save_path=None, use_log=True):
    """
    Plot MSE distributions for background vs signals.
    """
    plt.figure(figsize=(12, 8))
    
    colors = {'Background': 'gray', 'Ato4l': 'red', 'leptoquark': 'blue', 
              'hChToTauNu': 'green', 'hToTauTau': 'orange'}
    
    for label, mse_values in mse_dict.items():
        alpha = 0.5 if label == 'Background' else 0.7
        linewidth = 2 if label != 'Background' else 1.5
        plt.hist(mse_values, bins=100, alpha=alpha, label=label,
                color=colors.get(label, 'black'), density=True,
                histtype='step', linewidth=linewidth)
    
    plt.xlabel('Reconstruction MSE', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'{title} - Reconstruction MSE Distributions', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if use_log:
        plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
        print(f"Saved MSE distribution plot to {save_path}")
    plt.show()


def plot_mse_by_particle_type(mse_per_feature_dict, title, save_path=None):
    """
    Plot MSE broken down by particle type (MET, Electrons, Muons, Jets).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = {'Background': 'gray', 'Ato4l': 'red', 'leptoquark': 'blue', 
              'hChToTauNu': 'green', 'hToTauTau': 'orange'}
    
    for ax_idx, (group_name, group_slice) in enumerate(FEATURE_GROUPS.items()):
        ax = axes[ax_idx]
        
        for label, mse_per_feature in mse_per_feature_dict.items():
            # Mean MSE across features in this group for each sample
            group_mse = np.mean(mse_per_feature[:, group_slice], axis=1)
            
            alpha = 0.5 if label == 'Background' else 0.7
            linewidth = 2 if label != 'Background' else 1.5
            ax.hist(group_mse, bins=100, alpha=alpha, label=label,
                   color=colors.get(label, 'black'), density=True,
                   histtype='step', linewidth=linewidth)
        
        ax.set_xlabel(f'{group_name} MSE', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{group_name} Reconstruction MSE', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.suptitle(f'{title} - Reconstruction MSE by Particle Type', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
        print(f"Saved MSE by particle type plot to {save_path}")
    plt.show()


def plot_mean_mse_per_feature(mse_per_feature_dict, title, save_path=None):
    """
    Plot mean MSE for each of the 56 features, comparing background vs signals.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'Background': 'gray', 'Ato4l': 'red', 'leptoquark': 'blue', 
              'hChToTauNu': 'green', 'hToTauTau': 'orange'}
    
    # Plot each particle group
    group_info = [
        ('MET', slice(0, 2), feature_labels[0:2]),
        ('Electrons', slice(2, 14), feature_labels[2:14]),
        ('Muons', slice(14, 26), feature_labels[14:26]),
        ('Jets', slice(26, 56), feature_labels[26:56]),
    ]
    
    for ax_idx, (group_name, group_slice, labels) in enumerate(group_info):
        ax = axes.flatten()[ax_idx]
        
        x = np.arange(len(labels))
        width = 0.15
        
        for i, (label, mse_per_feature) in enumerate(mse_per_feature_dict.items()):
            mean_mse = np.mean(mse_per_feature[:, group_slice], axis=0)
            offset = (i - len(mse_per_feature_dict)/2 + 0.5) * width
            ax.bar(x + offset, mean_mse, width, label=label, 
                  color=colors.get(label, 'black'), alpha=0.7)
        
        ax.set_xlabel('Feature', fontsize=10)
        ax.set_ylabel('Mean MSE', fontsize=10)
        ax.set_title(f'{group_name} Features', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{title} - Mean Reconstruction MSE per Feature', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
        print(f"Saved mean MSE per feature plot to {save_path}")
    plt.show()


def plot_mse_statistics(mse_dict, title, save_path=None):
    """
    Print and plot summary statistics of MSE for each class.
    """
    print(f"\n{'='*60}")
    print(f"Reconstruction MSE Statistics - {title}")
    print(f"{'='*60}")
    
    stats_data = []
    for label, mse_values in mse_dict.items():
        stats = {
            'Label': label,
            'Mean': np.mean(mse_values),
            'Std': np.std(mse_values),
            'Median': np.median(mse_values),
            'Min': np.min(mse_values),
            'Max': np.max(mse_values),
            '95th': np.percentile(mse_values, 95),
            '99th': np.percentile(mse_values, 99)
        }
        stats_data.append(stats)
        print(f"\n{label}:")
        print(f"  Mean: {stats['Mean']:.6f} ± {stats['Std']:.6f}")
        print(f"  Median: {stats['Median']:.6f}")
        print(f"  Range: [{stats['Min']:.6f}, {stats['Max']:.6f}]")
        print(f"  95th percentile: {stats['95th']:.6f}")
        print(f"  99th percentile: {stats['99th']:.6f}")
    
    # Bar plot of mean MSE
    plt.figure(figsize=(10, 6))
    labels = [s['Label'] for s in stats_data]
    means = [s['Mean'] for s in stats_data]
    stds = [s['Std'] for s in stats_data]
    
    colors = ['gray', 'red', 'blue', 'green', 'orange'][:len(labels)]
    
    bars = plt.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Mean Reconstruction MSE', fontsize=12)
    plt.title(f'{title} - Mean Reconstruction MSE by Dataset', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
        print(f"\nSaved MSE statistics plot to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize VAE reconstruction quality')
    parser.add_argument('--model-path', default='outputs/models/particle_quantum_vae_best.pt',
                       help='Path to VAE model checkpoint')
    parser.add_argument('--max-samples', type=int, default=50000,
                       help='Max samples per dataset for visualization')
    parser.add_argument('--output-dir', default='outputs/plots',
                       help='Output directory for plots')
    parser.add_argument('--model-type', choices=['vae', 'hybrid', 'extended'], default='vae',
                       help='Model type to load')
    args = parser.parse_args()
    
    # Set torch default dtype
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model based on type
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_config = checkpoint.get('model_config', {})
    model_type = checkpoint.get('model_type', args.model_type)
    
    if model_type == 'hybrid':
        from hybrid_quantum_vae import create_hybrid_vae
        model = create_hybrid_vae(
            input_dim=model_config.get('input_dim', 56),
            hidden_dim=model_config.get('hidden_dim', 16),
            latent_dim=model_config.get('latent_dim', 3),
            steps_per_epoch=model_config.get('steps_per_epoch', 3125),
            cycle_length=model_config.get('cycle_length', 10),
            min_beta=model_config.get('min_beta', 0.1),
            max_beta=model_config.get('max_beta', 0.8),
        )
    elif model_type == 'extended':
        from extended_quantum_vae import create_extended_quantum_vae
        model = create_extended_quantum_vae(
            input_dim=model_config.get('input_dim', 56),
            hidden_dim=model_config.get('hidden_dim', 16),
            latent_dim=model_config.get('latent_dim', 3),
            quantum_depth=model_config.get('quantum_depth', 2),
            steps_per_epoch=model_config.get('steps_per_epoch', 3125),
            cycle_length=model_config.get('cycle_length', 10),
            min_beta=model_config.get('min_beta', 0.1),
            max_beta=model_config.get('max_beta', 0.8),
            device=model_config.get('device', 'default.qubit')
        )
    else:
        model = create_particle_quantum_vae(
            input_dim=model_config.get('input_dim', 56),
            hidden_dim=model_config.get('hidden_dim', 16),
            latent_dim=model_config.get('latent_dim', 3),
            quantum_depth=model_config.get('quantum_depth', 2),
            device=model_config.get('device', 'default.qubit')
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully! (type: {model_type})")
    
    # Data paths
    background_path = '/global/cfs/cdirs/m2616/sagar/QiML/background_for_training.h5'
    signal_paths = [
        '/global/cfs/cdirs/m2616/sagar/QiML/Ato4l_lepFilter_13TeV.h5',
        '/global/cfs/cdirs/m2616/sagar/QiML/leptoquark_LOWMASS_lepFilter_13TeV.h5',
        '/global/cfs/cdirs/m2616/sagar/QiML/hChToTauNu_13TeV_PU20.h5',
        '/global/cfs/cdirs/m2616/sagar/QiML/hToTauTau_13TeV_PU20.h5'
    ]
    
    # Store results
    mse_dict = {}
    masked_mse_dict = {}
    mse_per_feature_dict = {}
    
    # Background (use samples from the evaluation set)
    print("\nLoading and reconstructing background data...")
    bg_data = load_data(background_path, start_idx=2_000_000, max_samples=args.max_samples)
    print(f"  Processing {len(bg_data)} background samples...")
    bg_recon, bg_mse, bg_mse_per_feature = get_reconstruction(model, bg_data, device)
    bg_masked_mse = get_masked_mse(bg_data, bg_recon)
    
    mse_dict['Background'] = bg_mse
    masked_mse_dict['Background'] = bg_masked_mse
    mse_per_feature_dict['Background'] = bg_mse_per_feature
    print(f"  Background MSE: mean={np.mean(bg_mse):.6f}, median={np.median(bg_mse):.6f}")
    
    # Signals
    for i, (path, label) in enumerate(zip(signal_paths, signal_labels)):
        if os.path.exists(path):
            print(f"\nLoading and reconstructing {label} data...")
            sig_data = load_data(path, max_samples=args.max_samples)
            print(f"  Processing {len(sig_data)} {label} samples...")
            sig_recon, sig_mse, sig_mse_per_feature = get_reconstruction(model, sig_data, device)
            sig_masked_mse = get_masked_mse(sig_data, sig_recon)
            
            mse_dict[label] = sig_mse
            masked_mse_dict[label] = sig_masked_mse
            mse_per_feature_dict[label] = sig_mse_per_feature
            print(f"  {label} MSE: mean={np.mean(sig_mse):.6f}, median={np.median(sig_mse):.6f}")
        else:
            print(f"\nWarning: {path} not found, skipping {label}")
    
    # Create visualizations
    model_title = f"VAE Model ({model_type})"
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)
    
    # 1. MSE distribution plot
    print("\n1. Creating MSE distribution plots...")
    plot_mse_distributions(
        mse_dict, 
        model_title,
        save_path=os.path.join(args.output_dir, 'reconstruction_mse_distributions.png')
    )
    
    # 2. Masked MSE distribution (only non-zero particles)
    print("\n2. Creating masked MSE distribution plots...")
    plot_mse_distributions(
        masked_mse_dict,
        f"{model_title} (Masked)",
        save_path=os.path.join(args.output_dir, 'reconstruction_mse_masked_distributions.png')
    )
    
    # 3. MSE by particle type
    print("\n3. Creating MSE by particle type plots...")
    plot_mse_by_particle_type(
        mse_per_feature_dict,
        model_title,
        save_path=os.path.join(args.output_dir, 'reconstruction_mse_by_particle.png')
    )
    
    # 4. Mean MSE per feature
    print("\n4. Creating mean MSE per feature plots...")
    plot_mean_mse_per_feature(
        mse_per_feature_dict,
        model_title,
        save_path=os.path.join(args.output_dir, 'reconstruction_mse_per_feature.png')
    )
    
    # 5. Statistics summary
    print("\n5. Creating statistics summary...")
    plot_mse_statistics(
        mse_dict,
        model_title,
        save_path=os.path.join(args.output_dir, 'reconstruction_mse_statistics.png')
    )
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
