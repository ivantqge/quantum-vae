#!/usr/bin/env python3
"""
Export test samples to CSV for C++ inference validation.
Also runs Python inference and saves results for comparison.
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import numpy as np
import torch
import sys
sys.path.insert(0, '/global/homes/i/ivang/quantum-vae')

from quantum_encoding import LazyH5Array
from block_quantum_ae import ParticleQAEAnomalyModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data", default="/global/cfs/cdirs/m2616/sagar/QiML/background_for_training.h5",
                        help="Path to HDF5 data file")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples to export")
    parser.add_argument("--out-samples", default="test_samples.csv", help="Output CSV for samples")
    parser.add_argument("--out-python", default="python_scores.csv", help="Output CSV for Python scores")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    data_loader = LazyH5Array(args.data, "Particles", norm=True)
    data_array = np.array(data_loader[:args.n_samples], dtype=np.float64)
    x = torch.tensor(data_array, dtype=torch.float64)
    print(f"Loaded {x.shape[0]} samples with {x.shape[1]} features")

    # Load model
    print(f"Loading model from {args.ckpt}...")
    model = ParticleQAEAnomalyModel()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    # Load Mahalanobis stats from checkpoint
    if "bg_mu" in ckpt:
        model.mu.copy_(ckpt["bg_mu"])
        model.precision.copy_(ckpt["bg_precision"])
        model._stats_fitted = True
        print("  Loaded Mahalanobis stats from bg_mu/bg_precision")
    elif "mu" in ckpt:
        model.mu.copy_(ckpt["mu"])
        model.precision.copy_(ckpt["precision"])
        model._stats_fitted = True
        print("  Loaded Mahalanobis stats from mu/precision")
    else:
        # Check if they're in the state dict (as buffers)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        if "mu" in sd:
            model.mu.copy_(sd["mu"])
            model.precision.copy_(sd["precision"])
            model._stats_fitted = True
            print("  Loaded Mahalanobis stats from state_dict buffers")
        else:
            print("  WARNING: No Mahalanobis stats found in checkpoint!")
            print("  Available keys:", list(ckpt.keys()))
    
    model.eval()

    # Export samples to CSV
    print(f"Exporting {args.n_samples} samples to {args.out_samples}...")
    np.savetxt(args.out_samples, data_array, delimiter=",", fmt="%.18e")

    # Run Python inference
    print("Running Python inference...")
    with torch.no_grad():
        block_scores = model.block_scores(x).numpy()
        anomaly_scores = model.anomaly_score(x).numpy()

    # Save Python results
    print(f"Saving Python scores to {args.out_python}...")
    with open(args.out_python, "w") as f:
        f.write("met_score,ele_score,mu_score,jet_score,anomaly_score\n")
        for i in range(len(anomaly_scores)):
            f.write(f"{block_scores[i,0]:.18e},{block_scores[i,1]:.18e},"
                    f"{block_scores[i,2]:.18e},{block_scores[i,3]:.18e},"
                    f"{anomaly_scores[i]:.18e}\n")

    print("\nPython anomaly score statistics:")
    print(f"  Min:  {anomaly_scores.min():.6f}")
    print(f"  Max:  {anomaly_scores.max():.6f}")
    print(f"  Mean: {anomaly_scores.mean():.6f}")

    print("\nDone! Now run C++ inference:")
    print(f"  ./qae_inference -i {args.out_samples} -o cpp_scores.csv")
    print("\nThen compare results:")
    print(f"  python compare_scores.py --python {args.out_python} --cpp cpp_scores.csv")


if __name__ == "__main__":
    main()
