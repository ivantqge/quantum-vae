# SLAC Research: Quantum Machine Learning for Particle Physics

This project implements quantum machine learning techniques for particle physics data analysis using PennyLane and TensorFlow.

## Setup

### Option 1: Using Conda (Recommended)

1. Create the conda environment:
```bash
conda env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate slac-research
```

### Option 2: Using pip

1. Create a virtual environment:
```bash
python -m venv slac-research-env
source slac-research-env/bin/activate  # On Windows: slac-research-env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
1. Place your H5 data file (`background_for_training.h5`) in the project directory
2. Run the script:
```bash
python main.py
```

### Advanced Usage
The script supports command-line arguments for customization:

```bash
# Specify custom input/output files
python main.py --input my_data.h5 --output my_encoded_data.npy

# Use CPU instead of GPU
python main.py --device lightning.qubit

# Disable progress reporting
python main.py --no-progress

# Get help
python main.py --help
```

### Using the Encoding Module Directly
You can also import and use the quantum encoding functions directly:

```python
from quantum_encoding import encode_background_data, LazyH5Array, quantum_encode_particles

# Load data
data = LazyH5Array('my_data.h5', 'Particles')

# Encode with custom parameters
encoded = quantum_encode_particles(data, batch_progress=True)
```

### Training Quantum VAE Models
Train quantum-enhanced Variational Autoencoders on your encoded data:

```bash
# Basic training
python training.py --data data/background_for_training.h5

# Custom training configuration
python training.py --data data/background_for_training.h5 \
                   --epochs 50 \
                   --batch-size 512 \
                   --latent-dim 5 \
                   --h-dim-2 6 \
                   --output-dir my_results

# Training with limited data
python training.py --data data/background_for_training.h5 \
                   --max-samples 100000 \
                   --epochs 30
```

### Training Options
- `--data`: Path to H5 data file
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 1024)
- `--latent-dim`: Latent space dimension (default: 3)
- `--h-dim-2`: Quantum layer dimension/number of qubits (default: 4)
- `--learning-rate`: Learning rate (default: 0.001)
- `--max-samples`: Limit training data size
- `--output-dir`: Output directory for results

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for PennyLane Lightning GPU)
- H5 data file with particle physics data

## File Structure

### Core Modules
- `main.py`: Main execution script for quantum encoding with command-line interface
- `quantum_encoding.py`: Core quantum encoding module with all functions and classes
- `quantum_layers.py`: Quantum circuit definitions using Cirq and TensorFlow Quantum
- `vae_model.py`: Quantum Variational Autoencoder model architecture
- `training.py`: Training script for quantum VAE models

### Configuration Files
- `requirements.txt`: Python package requirements
- `environment.yml`: Conda environment specification

### Data Files
- `data/background_for_training.h5`: Input particle physics data
- `encoded_particles_objectwise.npy`: Output file with quantum-encoded data
- `outputs/`: Directory for training results and model checkpoints

## Notes

- The script uses PennyLane Lightning GPU for quantum circuit execution
- Make sure you have CUDA drivers installed for GPU acceleration
- The quantum encoding preserves the original 56-feature structure
