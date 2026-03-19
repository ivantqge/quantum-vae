#!/usr/bin/env python3
"""
Visualization script for the QAE block scores representation.
Analyzes the 4D or 9D block score representation for background and signal data.

For 4D model: scores are averaged per block [MET, Electrons, Muons, Jets]
For 9D model: all individual trash wire scores
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
from block_quantum_ae import ParticleQAEAnomalyModel
from block_quantum_ae_9d import ParticleQAEAnomalyModel9D


# Signal labels
signal_labels = ["Ato4l", "leptoquark", "hChToTauNu", "hToTauTau"]

# Component labels for the 4D encoding (averaged per block)
encoding_labels_4d = ["MET", "Electrons", "Muons", "Jets"]

# Component labels for the 9D encoding (all trash wires)
encoding_labels_9d = [
    "MET",
    "e_trash1", "e_trash2",
    "μ_trash1", "μ_trash2", 
    "j_trash1", "j_trash2", "j_trash3", "j_trash4"
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


def get_block_scores(model, data, device, batch_size=1024):
    """
    Get the block scores representation for input data.
    
    Args:
        model: The ParticleQAEAnomalyModel
        data: Input data array (N, 56)
        device: torch device
        batch_size: Batch size for processing
    
    Returns:
        numpy array of shape (N, 4) or (N, 9) - the block scores
    """
    model.eval()
    
    n_samples = data.shape[0]
    scores = []
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = data[start_idx:end_idx]
            
            if isinstance(batch, np.ndarray):
                batch_tensor = torch.tensor(batch, dtype=torch.float64, device=device)
            else:
                batch_tensor = batch.to(device)
            
            # Get block scores
            block_score = model.block_scores(batch_tensor)
            scores.append(block_score.cpu().numpy())
    
    return np.concatenate(scores, axis=0)


def plot_tsne(encodings_dict, title, save_path=None, n_samples_per_class=5000):
    """
    Create t-SNE visualization of encodings.
    """
    all_encodings = []
    colors = ['gray', 'red', 'blue', 'green', 'orange']
    
    for i, (label, encoding) in enumerate(encodings_dict.items()):
        n = min(n_samples_per_class, len(encoding))
        indices = np.random.choice(len(encoding), n, replace=False)
        all_encodings.append(encoding[indices])
    
    all_encodings = np.vstack(all_encodings)
    
    print(f"Running t-SNE on {len(all_encodings)} samples...")
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
    plt.title(f'{title} - t-SNE of Block Scores', fontsize=14)
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
    all_encodings = []
    colors = ['gray', 'red', 'blue', 'green', 'orange']
    
    for i, (label, encoding) in enumerate(encodings_dict.items()):
        n = min(n_samples_per_class, len(encoding))
        indices = np.random.choice(len(encoding), n, replace=False)
        all_encodings.append(encoding[indices])
    
    all_encodings = np.vstack(all_encodings)
    
    print(f"Running PCA on {len(all_encodings)} samples...")
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
    plt.title(f'{title} - PCA of Block Scores', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
        print(f"Saved PCA plot to {save_path}")
    plt.show()


def plot_encoding_distributions(encodings_dict, encoding_labels, title, save_path=None):
    """
    Plot distributions of each encoding dimension for background vs signals.
    """
    n_dims = len(encoding_labels)
    n_cols = min(n_dims, 5)
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    colors = {'Background': 'gray', 'Ato4l': 'red', 'leptoquark': 'blue', 
              'hChToTauNu': 'green', 'hToTauTau': 'orange'}
    
    for i in range(n_dims):
        ax = axes[i]
        
        for label, encoding in encodings_dict.items():
            alpha = 0.5 if label == 'Background' else 0.7
            linewidth = 2 if label != 'Background' else 1.5
            ax.hist(encoding[:, i], bins=50, alpha=alpha, label=label,
                   color=colors.get(label, 'black'), density=True,
                   histtype='step', linewidth=linewidth)
        
        ax.set_xlabel(encoding_labels[i], fontsize=10)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'Dim {i}: {encoding_labels[i]}', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(n_dims, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{title} - Distribution of Block Scores', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
        print(f"Saved distribution plot to {save_path}")
    plt.show()


def plot_encoding_means(encodings_dict, encoding_labels, title, save_path=None):
    """
    Plot mean encoding values for each class.
    """
    n_dims = len(encoding_labels)
    plt.figure(figsize=(max(10, n_dims * 0.8), 6))
    
    x = np.arange(n_dims)
    width = 0.15
    colors = ['gray', 'red', 'blue', 'green', 'orange']
    
    for i, (label, encoding) in enumerate(encodings_dict.items()):
        means = encoding.mean(axis=0)
        stds = encoding.std(axis=0)
        offset = (i - len(encodings_dict)/2 + 0.5) * width
        plt.bar(x + offset, means, width, label=label, color=colors[i], alpha=0.7)
        plt.errorbar(x + offset, means, yerr=stds, fmt='none', color='black', 
                    capsize=2, alpha=0.5)
    
    plt.xlabel('Block Score Dimension', fontsize=12)
    plt.ylabel('Mean Value', fontsize=12)
    plt.title(f'{title} - Mean Block Scores by Class', fontsize=14)
    plt.xticks(x, encoding_labels, rotation=45, ha='right')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
        print(f"Saved mean encoding plot to {save_path}")
    plt.show()


def plot_2d_scatter(encodings_dict, encoding_labels, title, save_path=None, 
                    dim1=0, dim2=1, n_samples_per_class=10000):
    """
    Create 2D scatter plot of two selected dimensions.
    """
    plt.figure(figsize=(10, 8))
    colors = ['gray', 'red', 'blue', 'green', 'orange']
    
    for i, (label, encoding) in enumerate(encodings_dict.items()):
        n = min(n_samples_per_class, len(encoding))
        indices = np.random.choice(len(encoding), n, replace=False)
        subset = encoding[indices]
        
        alpha = 0.2 if label == 'Background' else 0.5
        plt.scatter(subset[:, dim1], subset[:, dim2], 
                   c=colors[i], label=label, alpha=alpha, s=5)
    
    plt.xlabel(f'{encoding_labels[dim1]}', fontsize=12)
    plt.ylabel(f'{encoding_labels[dim2]}', fontsize=12)
    plt.title(f'{title} - {encoding_labels[dim1]} vs {encoding_labels[dim2]}', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=150)
        print(f"Saved 2D scatter plot to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize QAE block scores representation')
    parser.add_argument('--model-path', default=None,
                       help='Path to QAE model checkpoint (auto-detected if not set)')
    parser.add_argument('--max-samples', type=int, default=50000,
                       help='Max samples per dataset for visualization')
    parser.add_argument('--output-dir', default='outputs/plots',
                       help='Output directory for plots')
    parser.add_argument('--skip-tsne', action='store_true',
                       help='Skip t-SNE visualization (can be slow)')
    parser.add_argument('--prefix', default='qae',
                       help='Prefix for output filenames')
    args = parser.parse_args()
    
    # Set torch default dtype
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Auto-detect model path
    if args.model_path is None:
        # Try to find a QAE model
        candidates = [
            'outputs/models/quantum_qae_duffy_d1_final.pt',
            'outputs/models/quantum_qae_duffy_d1_best.pt',
            'outputs/models/particle_quantum_qae_final.pt',
            'outputs/models/particle_quantum_qae_best.pt',
        ]
        for path in candidates:
            if os.path.exists(path):
                args.model_path = path
                break
        if args.model_path is None:
            raise ValueError("No QAE model found. Please specify --model-path")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_config = checkpoint.get('model_config', {})
    
    # Detect 4D vs 9D model
    trash_dim = model_config.get('trash_dim', 4)
    print(f"Detected {trash_dim}D QAE model")
    
    if trash_dim == 9:
        model = ParticleQAEAnomalyModel9D(depth=model_config.get('quantum_depth', 2))
        encoding_labels = encoding_labels_9d
    else:
        model = ParticleQAEAnomalyModel(depth=model_config.get('quantum_depth', 2))
        encoding_labels = encoding_labels_4d
    
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
    print(f"  Getting block scores for {len(bg_data)} background samples...")
    encodings_dict['Background'] = get_block_scores(model, bg_data, device)
    print(f"  Background block scores shape: {encodings_dict['Background'].shape}")
    
    # Signals
    for i, (path, label) in enumerate(zip(signal_paths, signal_labels)):
        if os.path.exists(path):
            print(f"\nLoading and encoding {label} data...")
            sig_data = load_data(path, max_samples=args.max_samples)
            print(f"  Getting block scores for {len(sig_data)} {label} samples...")
            encodings_dict[label] = get_block_scores(model, sig_data, device)
            print(f"  {label} block scores shape: {encodings_dict[label].shape}")
        else:
            print(f"\nWarning: {path} not found, skipping {label}")
    
    # Create visualizations
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)
    
    model_title = f"QAE Model ({trash_dim}D)"
    
    # 1. Distribution plots
    print("\n1. Creating block score distribution plots...")
    plot_encoding_distributions(
        encodings_dict,
        encoding_labels,
        model_title,
        save_path=os.path.join(args.output_dir, f'{args.prefix}_block_distributions.png')
    )
    
    # 2. Mean encoding bar plot
    print("\n2. Creating mean block scores plot...")
    plot_encoding_means(
        encodings_dict,
        encoding_labels,
        model_title, 
        save_path=os.path.join(args.output_dir, f'{args.prefix}_block_means.png')
    )
    
    # 3. 2D scatter plots for interesting dimension pairs
    if trash_dim == 4:
        print("\n3. Creating 2D scatter plots...")
        # MET vs Jets
        plot_2d_scatter(
            encodings_dict, encoding_labels, model_title,
            save_path=os.path.join(args.output_dir, f'{args.prefix}_scatter_met_jets.png'),
            dim1=0, dim2=3
        )
        # Electrons vs Muons
        plot_2d_scatter(
            encodings_dict, encoding_labels, model_title,
            save_path=os.path.join(args.output_dir, f'{args.prefix}_scatter_ele_mu.png'),
            dim1=1, dim2=2
        )
    
    # 4. PCA visualization
    print("\n4. Creating PCA visualization...")
    plot_pca(
        encodings_dict,
        model_title,
        save_path=os.path.join(args.output_dir, f'{args.prefix}_block_pca.png')
    )
    
    # 5. t-SNE visualization (optional, can be slow)
    if not args.skip_tsne:
        print("\n5. Creating t-SNE visualization (this may take a while)...")
        plot_tsne(
            encodings_dict,
            model_title,
            save_path=os.path.join(args.output_dir, f'{args.prefix}_block_tsne.png'),
            n_samples_per_class=5000
        )
    else:
        print("\n5. Skipping t-SNE visualization (use without --skip-tsne to enable)")
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
