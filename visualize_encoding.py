#!/usr/bin/env python3
"""
Visualization script for the quantum encoding representation.
Analyzes the 19D representation after the quantum layer (56 -> 19) for
background and signal data.
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from quantum_encoding import LazyH5Array
from full_quantum_vae import create_particle_quantum_vae
from extended_quantum_vae import create_extended_quantum_vae


# Signal labels
signal_labels = ["Ato4l", "leptoquark", "hChToTauNu", "hToTauTau"]

# Component labels for the 19D encoding
encoding_labels = [
    "MET",
    "e1", "e2", "e3", "e4",
    "μ1", "μ2", "μ3", "μ4", 
    "j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8", "j9", "j10"
]


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


def get_quantum_encoding(model, data, device, batch_size=1024):
    """
    Get the quantum encoding representation for input data.
    
    Args:
        model: The QuantumVAE model
        data: Input data array (N, 56)
        device: torch device
        batch_size: Batch size for processing
    
    Returns:
        numpy array of shape (N, D) - the quantum encoding
    """
    model.eval()
    quantum_encoder = model.encoder.quantum_encoder
    
    n_samples = data.shape[0]
    encodings = []
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = data[start_idx:end_idx]
            
            if isinstance(batch, np.ndarray):
                batch_tensor = torch.tensor(batch, dtype=torch.float64, device=device)
            else:
                batch_tensor = batch.to(device)
            
            # Get quantum encoding (56 -> 19)
            encoding = quantum_encoder(batch_tensor)
            encodings.append(encoding.cpu().numpy())
    
    return np.concatenate(encodings, axis=0)


def plot_tsne(encodings_dict, title, save_path=None, n_samples_per_class=5000):
    """
    Create t-SNE visualization of encodings.
    
    Args:
        encodings_dict: Dict mapping label -> encoding array
        title: Plot title
        save_path: Path to save the plot
        n_samples_per_class: Max samples per class for t-SNE (for speed)
    """
    # Subsample if needed
    all_encodings = []
    all_labels = []
    colors = ['gray', 'red', 'blue', 'green', 'orange']
    
    for i, (label, encoding) in enumerate(encodings_dict.items()):
        n = min(n_samples_per_class, len(encoding))
        indices = np.random.choice(len(encoding), n, replace=False)
        all_encodings.append(encoding[indices])
        all_labels.extend([label] * n)
    
    all_encodings = np.vstack(all_encodings)
    dim = all_encodings.shape[1]
    
    print(f"Running t-SNE on {len(all_encodings)} samples in {dim}D space...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000)
    embeddings = tsne.fit_transform(all_encodings)
    
    plt.figure(figsize=(12, 10))
    
    start_idx = 0
    for i, (label, encoding) in enumerate(encodings_dict.items()):
        n = min(n_samples_per_class, len(encoding))
        end_idx = start_idx + n
        
        alpha = 0.3 if label == 'Background' else 0.6
        plt.scatter(embeddings[start_idx:end_idx, 0], 
                   embeddings[start_idx:end_idx, 1],
                   c=colors[i], label=label, alpha=alpha, s=10)
        start_idx = end_idx
    
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.title(f'{title} - t-SNE of {dim}D Quantum Encoding', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
        print(f"Saved t-SNE plot to {save_path}")
    plt.show()


def plot_pca(encodings_dict, title, save_path=None, n_samples_per_class=10000):
    """
    Create PCA visualization of encodings.
    """
    # Subsample if needed
    all_encodings = []
    all_labels = []
    colors = ['gray', 'red', 'blue', 'green', 'orange']
    
    for i, (label, encoding) in enumerate(encodings_dict.items()):
        n = min(n_samples_per_class, len(encoding))
        indices = np.random.choice(len(encoding), n, replace=False)
        all_encodings.append(encoding[indices])
        all_labels.extend([label] * n)
    
    all_encodings = np.vstack(all_encodings)
    dim = all_encodings.shape[1]
    
    print(f"Running PCA on {len(all_encodings)} samples in {dim}D space...")
    pca = PCA(n_components=2)
    embeddings = pca.fit_transform(all_encodings)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    plt.figure(figsize=(12, 10))
    
    start_idx = 0
    for i, (label, encoding) in enumerate(encodings_dict.items()):
        n = min(n_samples_per_class, len(encoding))
        end_idx = start_idx + n
        
        alpha = 0.3 if label == 'Background' else 0.6
        plt.scatter(embeddings[start_idx:end_idx, 0], 
                   embeddings[start_idx:end_idx, 1],
                   c=colors[i], label=label, alpha=alpha, s=10)
        start_idx = end_idx
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    plt.title(f'{title} - PCA of {dim}D Quantum Encoding', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
        print(f"Saved PCA plot to {save_path}")
    plt.show()


def plot_encoding_distributions(encodings_dict, title, save_path=None):
    """
    Plot distributions of each encoding dimension for background vs signals.
    """
    # Determine encoding dimensionality from the first entry
    first_encoding = next(iter(encodings_dict.values()))
    dim = first_encoding.shape[1]

    # Grid layout: try to keep roughly square
    n_cols = 5
    n_rows = int(np.ceil(dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()
    
    colors = {'Background': 'gray', 'Ato4l': 'red', 'leptoquark': 'blue', 
              'hChToTauNu': 'green', 'hToTauTau': 'orange'}
    
    for i in range(dim):
        ax = axes[i]
        
        for label, encoding in encodings_dict.items():
            alpha = 0.5 if label == 'Background' else 0.7
            linewidth = 2 if label != 'Background' else 1.5
            ax.hist(encoding[:, i], bins=50, alpha=alpha, label=label,
                   color=colors.get(label, 'black'), density=True,
                   histtype='step', linewidth=linewidth)
        
        label_name = encoding_labels[i] if i < len(encoding_labels) else f'Dim {i}'
        ax.set_xlabel(label_name, fontsize=10)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'Dim {i}: {label_name}', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=8)
    
    # Hide any unused subplots
    for j in range(dim, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(f'{title} - Distribution of {dim}D Quantum Encoding Components', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
        print(f"Saved distribution plot to {save_path}")
    plt.show()


def plot_encoding_means(encodings_dict, title, save_path=None):
    """
    Plot mean encoding values for each class.
    """
    # Determine encoding dimensionality from the first entry
    first_encoding = next(iter(encodings_dict.values()))
    dim = first_encoding.shape[1]

    plt.figure(figsize=(max(14, dim * 0.6), 6))
    
    x = np.arange(dim)
    width = 0.15
    colors = ['gray', 'red', 'blue', 'green', 'orange']
    
    for i, (label, encoding) in enumerate(encodings_dict.items()):
        means = encoding.mean(axis=0)
        stds = encoding.std(axis=0)
        offset = (i - len(encodings_dict)/2 + 0.5) * width
        plt.bar(x + offset, means, width, label=label, color=colors[i], alpha=0.7)
        plt.errorbar(x + offset, means, yerr=stds, fmt='none', color='black', 
                    capsize=2, alpha=0.5)
    
    plt.xlabel('Encoding Dimension', fontsize=12)
    plt.ylabel('Mean Value', fontsize=12)
    plt.title(f'{title} - Mean {dim}D Quantum Encoding by Class', fontsize=14)
    tick_labels = [encoding_labels[i] if i < len(encoding_labels) else f'Dim {i}' for i in range(dim)]
    plt.xticks(x, tick_labels, rotation=45, ha='right')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
        print(f"Saved mean encoding plot to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize quantum encoding representation')
    parser.add_argument('--model-path', default='outputs/models/particle_extended_quantum_vae_best.pt',
                       help='Path to VAE model checkpoint')
    parser.add_argument('--max-samples', type=int, default=500000,
                       help='Max samples per dataset for visualization')
    parser.add_argument('--output-dir', default='outputs/plots_extended',
                       help='Output directory for plots')
    parser.add_argument('--skip-tsne', action='store_true',
                       help='Skip t-SNE visualization (can be slow)')
    args = parser.parse_args()
    
    # Set torch default dtype
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_config = checkpoint.get('model_config', {})
    model_type = checkpoint.get('model_type', 'vae')
    
    # Recreate correct architecture based on model_type
    if model_type == 'extended':
        print("Detected extended quantum VAE checkpoint (56 -> 32 encoder).")
        model = create_extended_quantum_vae(**model_config)
        model_label = "Extended VAE Model"
    else:
        print("Detected standard quantum VAE checkpoint (56 -> 19 encoder).")
        model = create_particle_quantum_vae(**model_config)
        model_label = "VAE Model"
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Data paths
    background_path = '/global/cfs/cdirs/m2616/sagar/QiML/background_for_training.h5'
    signal_paths = [
        '/global/cfs/cdirs/m2616/sagar/QiML/Ato4l_lepFilter_13TeV.h5',
        '/global/cfs/cdirs/m2616/sagar/QiML/leptoquark_LOWMASS_lepFilter_13TeV.h5',
        '/global/cfs/cdirs/m2616/sagar/QiML/hChToTauNu_13TeV_PU20.h5',
        '/global/cfs/cdirs/m2616/sagar/QiML/hToTauTau_13TeV_PU20.h5'
    ]
    
    # Load and encode data
    encodings_dict = {}
    
    # Background (use samples from the evaluation set)
    print("\nLoading and encoding background data...")
    bg_data = load_data(background_path, start_idx=2_000_000, max_samples=args.max_samples)
    print(f"  Encoding {len(bg_data)} background samples...")
    encodings_dict['Background'] = get_quantum_encoding(model, bg_data, device)
    print(f"  Background encoding shape: {encodings_dict['Background'].shape}")
    
    # Signals
    for i, (path, label) in enumerate(zip(signal_paths, signal_labels)):
        if os.path.exists(path):
            print(f"\nLoading and encoding {label} data...")
            sig_data = load_data(path, max_samples=args.max_samples)
            print(f"  Encoding {len(sig_data)} {label} samples...")
            encodings_dict[label] = get_quantum_encoding(model, sig_data, device)
            print(f"  {label} encoding shape: {encodings_dict[label].shape}")
        else:
            print(f"\nWarning: {path} not found, skipping {label}")
    
    # Create visualizations
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)
    
    # 1. Distribution plots
    print("\n1. Creating encoding distribution plots...")
    plot_encoding_distributions(
        encodings_dict, 
        model_label,
        save_path=os.path.join(args.output_dir, 'encoding_distributions.png')
    )
    
    # 2. Mean encoding bar plot
    print("\n2. Creating mean encoding plot...")
    plot_encoding_means(
        encodings_dict,
        model_label, 
        save_path=os.path.join(args.output_dir, 'encoding_means.png')
    )
    
    # 3. PCA visualization
    print("\n3. Creating PCA visualization...")
    plot_pca(
        encodings_dict,
        model_label,
        save_path=os.path.join(args.output_dir, 'encoding_pca.png')
    )
    
    # 4. t-SNE visualization (optional, can be slow)
    if not args.skip_tsne:
        print("\n4. Creating t-SNE visualization (this may take a while)...")
        plot_tsne(
            encodings_dict,
            model_label,
            save_path=os.path.join(args.output_dir, 'encoding_tsne.png'),
            n_samples_per_class=5000
        )
    else:
        print("\n4. Skipping t-SNE visualization (use --skip-tsne=False to enable)")
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
