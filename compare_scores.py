#!/usr/bin/env python3
"""
Compare Python and C++ inference results.
"""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="python_scores.csv", help="Python scores CSV")
    parser.add_argument("--cpp", default="cpp_scores.csv", help="C++ scores CSV")
    args = parser.parse_args()

    # Load scores (skip header)
    py_data = np.loadtxt(args.python, delimiter=",", skiprows=1)
    cpp_data = np.loadtxt(args.cpp, delimiter=",", skiprows=1)

    if py_data.shape != cpp_data.shape:
        print(f"Shape mismatch: Python {py_data.shape} vs C++ {cpp_data.shape}")
        return

    n_samples = py_data.shape[0]
    print(f"Comparing {n_samples} samples...")

    # Column names
    cols = ["met_score", "ele_score", "mu_score", "jet_score", "anomaly_score"]

    print("\n" + "=" * 70)
    print(f"{'Column':<15} {'Max Abs Diff':>15} {'Mean Abs Diff':>15} {'Max Rel Diff':>15}")
    print("=" * 70)

    for i, col in enumerate(cols):
        py_col = py_data[:, i]
        cpp_col = cpp_data[:, i]
        
        abs_diff = np.abs(py_col - cpp_col)
        rel_diff = abs_diff / (np.abs(py_col) + 1e-12)
        
        print(f"{col:<15} {abs_diff.max():>15.2e} {abs_diff.mean():>15.2e} {rel_diff.max():>15.2e}")

    print("=" * 70)

    # Overall anomaly score comparison
    py_anomaly = py_data[:, 4]
    cpp_anomaly = cpp_data[:, 4]
    
    correlation = np.corrcoef(py_anomaly, cpp_anomaly)[0, 1]
    print(f"\nAnomaly score correlation: {correlation:.10f}")

    # Check if they match within tolerance
    tol = 1e-6
    matches = np.allclose(py_anomaly, cpp_anomaly, rtol=tol, atol=tol)
    if matches:
        print(f"SUCCESS: All anomaly scores match within tolerance {tol}")
    else:
        n_mismatch = np.sum(np.abs(py_anomaly - cpp_anomaly) > tol)
        print(f"WARNING: {n_mismatch}/{n_samples} samples differ by more than {tol}")

    # Show first few samples
    print("\nFirst 5 samples comparison:")
    print(f"{'Sample':<8} {'Python':>15} {'C++':>15} {'Diff':>15}")
    print("-" * 55)
    for i in range(min(5, n_samples)):
        diff = py_anomaly[i] - cpp_anomaly[i]
        print(f"{i:<8} {py_anomaly[i]:>15.8f} {cpp_anomaly[i]:>15.8f} {diff:>15.2e}")


if __name__ == "__main__":
    main()
