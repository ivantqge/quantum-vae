#!/usr/bin/env python3
"""Debug script to print normalized feature ranges for QMI model."""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import sys
sys.path.insert(0, '/global/homes/i/ivang/quantum-vae')

import numpy as np
import torch
from block_quantum_ae_qmi import QMIParticleQAEEncoder
from quantum_encoding import LazyH5Array

# Load some real data (same way as training.py)
data_path = "/global/cfs/cdirs/m2616/sagar/QiML/background_for_training.h5"
data_loader = LazyH5Array(data_path, "Particles", norm=False)
data_array = np.array(data_loader[:1000], dtype=np.float64)
x = torch.tensor(data_array, dtype=torch.float64)

print(f"Loaded {x.shape[0]} samples with {x.shape[1]} features")

encoder = QMIParticleQAEEncoder()
encoder.debug_print_ranges(x)
