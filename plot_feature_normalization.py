#!/usr/bin/env python3
"""
Plot per-feature distributions before and after the physics-aware normalization
defined in `quantum_encoding.LazyH5Array._normalize`.

This script:
- Loads an HDF5 dataset using `LazyH5Array` with and without normalization
- Draws histograms for each feature (raw vs normalized)
- Overlays signal distributions on top of background
- Saves plots as PNG files, one per feature
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from quantum_encoding import LazyH5Array

# Signal paths and labels
SIGNAL_LABELS = ["Ato4l", "leptoquark", "hChToTauNu", "hToTauTau"]
SIGNAL_PATHS = [
    '/global/cfs/cdirs/m2616/sagar/QiML/Ato4l_lepFilter_13TeV.h5',
    '/global/cfs/cdirs/m2616/sagar/QiML/leptoquark_LOWMASS_lepFilter_13TeV.h5',
    '/global/cfs/cdirs/m2616/sagar/QiML/hChToTauNu_13TeV_PU20.h5',
    '/global/cfs/cdirs/m2616/sagar/QiML/hToTauTau_13TeV_PU20.h5'
]

# Colors for plotting
COLORS = {'Background': 'gray', 'Ato4l': 'red', 'leptoquark': 'blue', 
          'hChToTauNu': 'green', 'hToTauTau': 'orange'}


def build_feature_names(n_features: int = 56):
    """
    Construct human-readable names for the 56 feature layout used in LazyH5Array.

    Layout:
    - 0-1: MET pt, phi
    - 2-13: electrons (4): pt(2-5), eta(6-9), phi(10-13)
    - 14-25: muons (4): pt(14-17), eta(18-21), phi(22-25)
    - 26-55: jets (10): pt(26-35), eta(36-45), phi(46-55)
    """
    names = []

    # MET
    names.append("MET_pt")   # 0
    names.append("MET_phi")  # 1

    # Electrons (indices 2-13)
    for i in range(4):
        names.append(f"e{i+1}_pt")   # 2-5
    for i in range(4):
        names.append(f"e{i+1}_eta")  # 6-9
    for i in range(4):
        names.append(f"e{i+1}_phi")  # 10-13

    # Muons (indices 14-25)
    for i in range(4):
        names.append(f"mu{i+1}_pt")   # 14-17
    for i in range(4):
        names.append(f"mu{i+1}_eta")  # 18-21
    for i in range(4):
        names.append(f"mu{i+1}_phi")  # 22-25

    # Jets (indices 26-55)
    for i in range(10):
        names.append(f"jet{i+1}_pt")   # 26-35
    for i in range(10):
        names.append(f"jet{i+1}_eta")  # 36-45
    for i in range(10):
        names.append(f"jet{i+1}_phi")  # 46-55

    if len(names) != n_features:
        raise ValueError(f"Expected {n_features} feature names, got {len(names)}")
    return names


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-feature distributions before and after normalization."
    )
    parser.add_argument(
        "--h5",
        required=True,
        help="Path to the HDF5 file for background (same one used by LazyH5Array).",
    )
    parser.add_argument(
        "--dataset-key",
        default="Particles",
        help="HDF5 dataset key to use (default: %(default)s).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100000,
        help="Number of events to sample from each dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="feature_normalization_plots",
        help="Directory where PNG files will be saved (default: %(default)s).",
    )
    parser.add_argument(
        "--no-signals",
        action="store_true",
        help="Skip plotting signal distributions (only plot background).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load background data
    print("Loading background data...")
    arr_raw_bg = LazyH5Array(
        h5_file_path=args.h5,
        dataset_key=args.dataset_key,
        norm=False,
    )
    arr_norm_bg = LazyH5Array(
        h5_file_path=args.h5,
        dataset_key=args.dataset_key,
        norm=True,
    )

    n_total_bg = len(arr_raw_bg)
    n_samples_bg = min(args.n_samples, n_total_bg)

    print(f"  Loading {n_samples_bg} / {n_total_bg} background events...")
    bg_raw = arr_raw_bg[:n_samples_bg]
    bg_norm = arr_norm_bg[:n_samples_bg]

    n_features = bg_raw.shape[1]
    feature_names = build_feature_names(n_features)

    # Load signal data
    signals_raw = {}
    signals_norm = {}
    
    if not args.no_signals:
        for path, label in zip(SIGNAL_PATHS, SIGNAL_LABELS):
            if os.path.exists(path):
                print(f"Loading {label} data...")
                arr_raw_sig = LazyH5Array(h5_file_path=path, dataset_key=args.dataset_key, norm=False)
                arr_norm_sig = LazyH5Array(h5_file_path=path, dataset_key=args.dataset_key, norm=True)
                
                n_total_sig = len(arr_raw_sig)
                n_samples_sig = min(args.n_samples, n_total_sig)
                
                print(f"  Loading {n_samples_sig} / {n_total_sig} {label} events...")
                signals_raw[label] = arr_raw_sig[:n_samples_sig]
                signals_norm[label] = arr_norm_sig[:n_samples_sig]
            else:
                print(f"Warning: {path} not found, skipping {label}")

    print(f"\nPlotting {n_features} features...")

    for idx in range(n_features):
        fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)

        # Raw distributions
        ax = axes[0]
        # Background
        ax.hist(
            bg_raw[:, idx],
            bins=args.bins,
            histtype="step",
            color=COLORS['Background'],
            label="Background",
            density=True,
            linewidth=1.5,
        )
        # Signals
        for label in signals_raw:
            ax.hist(
                signals_raw[label][:, idx],
                bins=args.bins,
                histtype="step",
                color=COLORS.get(label, 'black'),
                label=label,
                density=True,
                linewidth=2,
            )
        ax.set_title(f"{fname} (raw)")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Normalized distributions
        ax = axes[1]
        # Background
        ax.hist(
            bg_norm[:, idx],
            bins=args.bins,
            histtype="step",
            color=COLORS['Background'],
            label="Background",
            density=True,
            linewidth=1.5,
        )
        # Signals
        for label in signals_norm:
            ax.hist(
                signals_norm[label][:, idx],
                bins=args.bins,
                histtype="step",
                color=COLORS.get(label, 'black'),
                label=label,
                density=True,
                linewidth=2,
            )
        ax.set_title(f"{fname} (normalized)")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        out_path = os.path.join(args.output_dir, f"{idx:02d}_{fname}.png")
        fig.suptitle(f"Feature {idx}: {fname}")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        if (idx + 1) % 10 == 0 or idx == n_features - 1:
            print(f"  Saved {idx + 1}/{n_features} plots...")

    print(f"\nDone! Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

