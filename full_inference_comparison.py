#!/usr/bin/env python3
"""
Run inference on full test set (last 2M events) with both Python and C++,
then plot histograms comparing the results.
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import subprocess
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, '/global/homes/i/ivang/quantum-vae')

from quantum_encoding import LazyH5Array
from block_quantum_ae import ParticleQAEAnomalyModel


def load_model(ckpt_path):
    """Load trained model with Mahalanobis stats."""
    model = ParticleQAEAnomalyModel()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    # Load Mahalanobis stats
    if "bg_mu" in ckpt:
        model.mu.copy_(ckpt["bg_mu"])
        model.precision.copy_(ckpt["bg_precision"])
        model._stats_fitted = True
    elif "mu" in ckpt:
        model.mu.copy_(ckpt["mu"])
        model.precision.copy_(ckpt["precision"])
        model._stats_fitted = True
    else:
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        if "mu" in sd:
            model.mu.copy_(sd["mu"])
            model.precision.copy_(sd["precision"])
            model._stats_fitted = True
    
    model.eval()
    return model


def run_python_inference(model, data, batch_size=512):
    """Run Python inference in batches."""
    n_samples = data.shape[0]
    all_scores = []
    all_block_scores = []
    
    print(f"Running Python inference on {n_samples} samples...")
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch = data[i:end]
            
            block_scores = model.block_scores(batch)
            anomaly_scores = model.anomaly_score(batch)
            
            all_block_scores.append(block_scores.numpy())
            all_scores.append(anomaly_scores.numpy())
            
            if (i // batch_size) % 100 == 0:
                print(f"  Processed {end}/{n_samples} samples...")
    
    return np.concatenate(all_block_scores), np.concatenate(all_scores)


def run_cpp_inference(samples_file, output_file, cpp_binary="./qae_inference"):
    """Run C++ inference."""
    print(f"Running C++ inference...")
    
    cmd = [cpp_binary, "-i", samples_file, "-o", output_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"C++ inference failed: {result.stderr}")
        return None, None
    
    print(result.stdout)
    
    # Load results
    data = np.loadtxt(output_file, delimiter=",", skiprows=1)
    block_scores = data[:, :4]
    anomaly_scores = data[:, 4]
    
    return block_scores, anomaly_scores


def plot_comparison(py_scores, cpp_scores, py_block, cpp_block, output_dir):
    """Plot comparison histograms."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Anomaly score distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Python vs C++ overlay
    ax = axes[0]
    bins = np.linspace(min(py_scores.min(), cpp_scores.min()), 
                       max(py_scores.max(), cpp_scores.max()), 100)
    ax.hist(py_scores, bins=bins, alpha=0.7, label='Python', density=True)
    ax.hist(cpp_scores, bins=bins, alpha=0.7, label='C++', density=True)
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title('Anomaly Score Distribution')
    ax.legend()
    ax.set_yscale('log')
    
    # Difference histogram
    ax = axes[1]
    diff = py_scores - cpp_scores
    ax.hist(diff, bins=100, alpha=0.7, color='green')
    ax.set_xlabel('Python - C++ Score')
    ax.set_ylabel('Count')
    ax.set_title(f'Score Difference\nMean: {diff.mean():.2e}, Std: {diff.std():.2e}')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # Scatter plot
    ax = axes[2]
    ax.scatter(py_scores, cpp_scores, alpha=0.1, s=1)
    ax.plot([py_scores.min(), py_scores.max()], 
            [py_scores.min(), py_scores.max()], 'r--', label='y=x')
    ax.set_xlabel('Python Score')
    ax.set_ylabel('C++ Score')
    ax.set_title(f'Python vs C++ Scores\nCorr: {np.corrcoef(py_scores, cpp_scores)[0,1]:.6f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'anomaly_score_comparison.png', dpi=150)
    plt.close()
    print(f"Saved {output_dir / 'anomaly_score_comparison.png'}")
    
    # 2. Block score comparisons
    block_names = ['MET', 'Electron', 'Muon', 'Jet']
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, name in enumerate(block_names):
        # Distribution overlay
        ax = axes[0, i]
        py_b = py_block[:, i]
        cpp_b = cpp_block[:, i]
        bins = np.linspace(min(py_b.min(), cpp_b.min()), 
                           max(py_b.max(), cpp_b.max()), 50)
        ax.hist(py_b, bins=bins, alpha=0.7, label='Python', density=True)
        ax.hist(cpp_b, bins=bins, alpha=0.7, label='C++', density=True)
        ax.set_xlabel(f'{name} Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} Block')
        ax.legend(fontsize=8)
        
        # Difference
        ax = axes[1, i]
        diff = py_b - cpp_b
        ax.hist(diff, bins=50, alpha=0.7, color='green')
        ax.set_xlabel('Python - C++')
        ax.set_ylabel('Count')
        ax.set_title(f'Diff: μ={diff.mean():.2e}, σ={diff.std():.2e}')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'block_score_comparison.png', dpi=150)
    plt.close()
    print(f"Saved {output_dir / 'block_score_comparison.png'}")
    
    # 3. Relative error distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    rel_error = (py_scores - cpp_scores) / (np.abs(py_scores) + 1e-10) * 100
    ax.hist(rel_error, bins=100, alpha=0.7)
    ax.set_xlabel('Relative Error (%)')
    ax.set_ylabel('Count')
    ax.set_title(f'Relative Error Distribution\nMean: {rel_error.mean():.4f}%, Max: {np.abs(rel_error).max():.4f}%')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / 'relative_error.png', dpi=150)
    plt.close()
    print(f"Saved {output_dir / 'relative_error.png'}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total samples: {len(py_scores)}")
    print(f"\nAnomaly Score:")
    print(f"  Python  - Mean: {py_scores.mean():.6f}, Std: {py_scores.std():.6f}")
    print(f"  C++     - Mean: {cpp_scores.mean():.6f}, Std: {cpp_scores.std():.6f}")
    print(f"  Diff    - Mean: {diff.mean():.2e}, Std: {diff.std():.2e}, Max: {np.abs(diff).max():.2e}")
    print(f"  Correlation: {np.corrcoef(py_scores, cpp_scores)[0,1]:.10f}")
    print(f"\nBlock Scores (Mean Abs Diff):")
    for i, name in enumerate(block_names):
        diff = np.abs(py_block[:, i] - cpp_block[:, i])
        print(f"  {name:10s}: {diff.mean():.2e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/models/qae_4block_best.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--data", default="/global/cfs/cdirs/m2616/sagar/QiML/background_for_training.h5",
                        help="Path to HDF5 data file")
    parser.add_argument("--n-samples", type=int, default=2000000,
                        help="Number of samples (from end of dataset)")
    parser.add_argument("--cpp-binary", default="./qae_inference",
                        help="Path to compiled C++ binary")
    parser.add_argument("--output-dir", default="outputs/cpp_comparison",
                        help="Output directory for plots")
    parser.add_argument("--tmp-dir", default="/tmp/qae_comparison",
                        help="Temp directory for intermediate files")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for Python inference")
    args = parser.parse_args()
    
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data (last n_samples)
    print(f"Loading data from {args.data}...")
    data_loader = LazyH5Array(args.data, "Particles", norm=True)
    total_samples = len(data_loader)
    
    start_idx = max(0, total_samples - args.n_samples)
    n_samples = total_samples - start_idx
    
    print(f"Total samples in file: {total_samples}")
    print(f"Using last {n_samples} samples (indices {start_idx} to {total_samples})")
    
    data_array = np.array(data_loader[start_idx:], dtype=np.float32)
    x = torch.tensor(data_array, dtype=torch.float32)
    print(f"Loaded {x.shape[0]} samples with {x.shape[1]} features")
    
    # Load model
    print(f"\nLoading model from {args.ckpt}...")
    model = load_model(args.ckpt)
    
    # Run Python inference
    py_block_scores, py_anomaly_scores = run_python_inference(model, x, args.batch_size)
    
    # Export samples for C++
    samples_file = tmp_dir / "test_samples.csv"
    print(f"\nExporting samples to {samples_file}...")
    np.savetxt(samples_file, data_array, delimiter=",", fmt="%.18e")
    
    # Run C++ inference
    cpp_output_file = tmp_dir / "cpp_scores.csv"
    cpp_block_scores, cpp_anomaly_scores = run_cpp_inference(
        str(samples_file), str(cpp_output_file), args.cpp_binary
    )
    
    if cpp_anomaly_scores is None:
        print("C++ inference failed!")
        return
    
    # Plot comparison
    plot_comparison(py_anomaly_scores, cpp_anomaly_scores, 
                    py_block_scores, cpp_block_scores, args.output_dir)
    
    print(f"\nDone! Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
