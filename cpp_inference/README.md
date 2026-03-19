# C++ Inference for Quantum Models

This directory contains C++ implementations of the quantum model inference for FPGA deployment.

## Directory Structure

```
cpp_inference/
├── qae/                          # Block Quantum Autoencoder (block_quantum_ae.py)
│   ├── qae_inference_ref.cpp     # Double-precision reference implementation
│   ├── qae_inference_vitis.cpp   # Fixed-point Xilinx Vitis HLS version
│   └── qae_inference_catapult.cpp # Fixed-point Siemens Catapult HLS version
├── extended_vae/                 # Extended Quantum VAE (extended_quantum_vae.py)
│   ├── extended_vae_inference_ref.cpp      # Double-precision reference
│   ├── extended_vae_inference_vitis.cpp    # Fixed-point Xilinx Vitis HLS version
│   └── extended_vae_inference_catapult.cpp # Fixed-point Siemens Catapult HLS version
├── scripts/                      # Export scripts
│   ├── export_qae_weights.py     # Export QAE weights to C++ headers
│   ├── export_vitis_headers.py   # Export QAE Vitis HLS headers
│   ├── export_catapult_headers.py # Export QAE Catapult HLS headers
│   ├── export_extended_vae_weights.py  # Export Extended VAE weights (reference)
│   ├── export_extended_vae_vitis_headers.py   # Export Extended VAE Vitis HLS headers
│   ├── export_extended_vae_catapult_headers.py # Export Extended VAE Catapult HLS headers
│   └── input_lut.py              # Generate trig lookup tables
└── README.md
```

## Models

### 1. Block Quantum Autoencoder (QAE)

Architecture:
- MET: 1 qubit (trash)
- Electrons: 4 qubits (2 latent, 2 trash)
- Muons: 4 qubits (2 latent, 2 trash)
- Jets: 10 qubits (6 latent, 4 trash)

Output: 4D Mahalanobis anomaly score

### 2. Extended Quantum VAE

Architecture:
- MET: 1 qubit → 3 outputs (PauliZ, PauliX, PauliY)
- Electrons: 4 qubits → 8 outputs (4 PauliZ + 4 PauliX)
- Muons: 4 qubits → 8 outputs (4 PauliZ + 4 PauliX)
- Jets: 10 qubits → 13 outputs (10 PauliZ + 3 PauliX)

Output: 32-dimensional latent encoding (fed to classical decoder for VAE)

## Usage

### QAE Model

```bash
cd cpp_inference/qae

# 1. Export weights from trained model
python ../scripts/export_qae_weights.py --ckpt ../../outputs/models/qae_4block_best.pt --out-dir .

# 2. Compile (no trig LUTs needed - computed directly)
g++ -O3 -std=c++17 -o qae_inference qae_inference_ref.cpp

# 3. Run inference (input should be RAW unnormalized data)
./qae_inference -i test_samples_raw.csv -o cpp_scores.csv
```

### Extended VAE Model

```bash
cd cpp_inference/extended_vae

# 1. Export weights
python ../scripts/export_extended_vae_weights.py --ckpt ../../outputs/models/particle_extended_quantum_vae_best.pt --out-dir .

# 2. Compile
g++ -O3 -std=c++17 -o extended_vae_inference extended_vae_inference_ref.cpp

# 3. Run inference
./extended_vae_inference -i test_samples.csv -o cpp_encoder_outputs.csv
```

## Input Data Format

CSV file with 56 features per row (no header):
- Column 0: MET pt
- Column 1: MET phi
- Columns 2-5: Electron pt (4)
- Columns 6-9: Electron eta (4)
- Columns 10-13: Electron phi (4)
- Columns 14-17: Muon pt (4)
- Columns 18-21: Muon eta (4)
- Columns 22-25: Muon phi (4)
- Columns 26-35: Jet pt (10)
- Columns 36-45: Jet eta (10)
- Columns 46-55: Jet phi (10)

**IMPORTANT: Different models use different normalizations!**

### QAE Normalization (linear to [0,1])
The C++ code normalizes raw inputs internally:
- pt: `pt / 1200`
- eta: `(eta + 5) / 10`
- phi: `(phi + π) / (2π)`

Input CSV should contain **raw (unnormalized)** physics values.

### Extended VAE Normalization (physics-aware, angles)
- pt: `log(pt + 1)` scaled to `[0, π]`
- eta: scaled from `[-3, 3]` to `[-π, π]`
- phi: clamped to `[-π, π]`

Input CSV should contain **normalized** values (use `LazyH5Array` with `norm=True`).

## HLS Synthesis

### Xilinx Vitis HLS

For FPGA synthesis with Xilinx Vitis HLS:

```bash
cd cpp_inference/qae

# Generate Vitis-compatible headers
python ../scripts/export_vitis_headers.py --ckpt ../../outputs/models/qae_4block_best.pt --out-dir vitis_headers

# Copy to your Vitis project:
# - vitis_headers/*.h
# - qae_inference_vitis.cpp

# In Vitis HLS:
# - Set qae_inference_top as the top function
# - Run C simulation, then synthesis
```

The Vitis version uses:
- `ap_fixed<16, 2, AP_RND, AP_SAT>` for angles/trig values
- `ap_fixed<18, 2, AP_RND, AP_SAT>` for statevector amplitudes
- `ap_fixed<24, 4, AP_RND, AP_SAT>` for accumulators
- `ap_fixed<20, 4, AP_RND, AP_SAT>` for scores

Key HLS pragmas used:
- `#pragma HLS INTERFACE mode=s_axilite` for AXI-Lite control
- `#pragma HLS ARRAY_PARTITION` for parallel memory access
- `#pragma HLS PIPELINE II=1` for loop pipelining
- `#pragma HLS DATAFLOW` for block-level parallelism

### Siemens Catapult HLS

For FPGA synthesis with Siemens Catapult:

```bash
cd cpp_inference/qae

# Generate Catapult-compatible headers
python ../scripts/export_catapult_headers.py --ckpt ../../outputs/models/qae_4block_best.pt --out-dir catapult_headers

# Copy to your Catapult project:
# - catapult_headers/*.h
# - qae_inference_catapult.cpp

# In Catapult:
# - Set qae_inference_top as the top function
# - Run C simulation, then synthesis
```

The Catapult version uses:
- `ac_fixed<16, 2, true, AC_RND, AC_SAT>` for angles
- `ac_fixed<18, 2, true, AC_RND, AC_SAT>` for amplitudes
- `ac_fixed<20, 4, true, AC_RND, AC_SAT>` for scores

### Extended VAE HLS

#### Vitis HLS

```bash
cd cpp_inference/extended_vae

# Generate Vitis-compatible headers
python ../scripts/export_extended_vae_vitis_headers.py \
    --ckpt ../../outputs/models/particle_extended_quantum_vae_best.pt \
    --out-dir vitis_headers

# Copy to your Vitis project:
# - vitis_headers/*.h
# - extended_vae_inference_vitis.cpp

# Top function: extended_vae_encoder_top
```

#### Catapult HLS

```bash
cd cpp_inference/extended_vae

# Generate Catapult-compatible headers
python ../scripts/export_extended_vae_catapult_headers.py \
    --ckpt ../../outputs/models/particle_extended_quantum_vae_best.pt \
    --out-dir catapult_headers

# Copy to your Catapult project:
# - catapult_headers/*.h
# - extended_vae_inference_catapult.cpp

# Top function: extended_vae_encoder_top
```

**Note:** Extended VAE expects **pre-normalized** input (physics-aware normalization).
Use `LazyH5Array` with `norm=True` to generate test data.

## Validation

To validate C++ against Python:

```bash
# Export test samples and run Python inference
python export_test_samples.py --ckpt outputs/models/qae_4block_best.pt --n-samples 1000

# Run C++ inference
./qae_inference -i test_samples.csv -o cpp_scores.csv

# Compare results
python compare_scores.py --python python_scores.csv --cpp cpp_scores.csv
```

Expected correlation: >0.9999 for double precision, >0.999 for float32.
