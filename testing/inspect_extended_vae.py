#!/usr/bin/env python3
"""
Inspect Extended Quantum VAE model weights and intermediate expectation values.

This script:
1. Loads a trained Extended Quantum VAE model and prints all trainable weights
2. For background and signal data, shows the expectation values after each layer
3. Visualizes the distribution of expectation values at different circuit stages
   comparing background vs signals (similar to visualize_encoding.py but at intermediate stages)
4. Creates PCA/t-SNE visualizations at each layer stage
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import numpy as np
import torch
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from quantum_encoding import LazyH5Array
from extended_quantum_vae import create_extended_quantum_vae

# Signal labels
SIGNAL_LABELS = ["Ato4l", "leptoquark", "hChToTauNu", "hToTauTau"]

# Data paths
BACKGROUND_PATH = '/global/cfs/cdirs/m2616/sagar/QiML/background_for_training.h5'
SIGNAL_PATHS = [
    '/global/cfs/cdirs/m2616/sagar/QiML/Ato4l_lepFilter_13TeV.h5',
    '/global/cfs/cdirs/m2616/sagar/QiML/leptoquark_LOWMASS_lepFilter_13TeV.h5',
    '/global/cfs/cdirs/m2616/sagar/QiML/hChToTauNu_13TeV_PU20.h5',
    '/global/cfs/cdirs/m2616/sagar/QiML/hToTauTau_13TeV_PU20.h5'
]

# Colors for plotting
COLORS = {'Background': 'gray', 'Ato4l': 'red', 'leptoquark': 'blue', 
          'hChToTauNu': 'green', 'hToTauTau': 'orange'}


def print_model_weights(model):
    """Print all trainable weights in the Extended Quantum VAE model."""
    print("\n" + "="*60)
    print("MODEL WEIGHTS")
    print("="*60)
    
    encoder = model.encoder.quantum_encoder
    
    print("\n[MET weights] (1 qubit, 3 rotations per layer: RX, RY, RZ)")
    print(f"  Shape: {encoder.met_weights.shape}")
    weights = encoder.met_weights.detach().cpu().numpy()
    n_layers = len(weights) // 3
    for layer in range(n_layers):
        layer_weights = weights[layer*3:(layer+1)*3]
        print(f"  Layer {layer} [RX, RY, RZ]: {layer_weights}")
    
    print("\n[Electron weights] (4 qubits, 8 rotations per layer: 4 RX + 4 RY)")
    print(f"  Shape: {encoder.ele_weights.shape}")
    weights = encoder.ele_weights.detach().cpu().numpy()
    n_layers = len(weights) // 8
    for layer in range(n_layers):
        rx_weights = weights[layer*8:layer*8+4]
        ry_weights = weights[layer*8+4:(layer+1)*8]
        print(f"  Layer {layer} RX: {rx_weights}")
        print(f"  Layer {layer} RY: {ry_weights}")
    
    print("\n[Muon weights] (4 qubits, 8 rotations per layer: 4 RX + 4 RY)")
    print(f"  Shape: {encoder.mu_weights.shape}")
    weights = encoder.mu_weights.detach().cpu().numpy()
    n_layers = len(weights) // 8
    for layer in range(n_layers):
        rx_weights = weights[layer*8:layer*8+4]
        ry_weights = weights[layer*8+4:(layer+1)*8]
        print(f"  Layer {layer} RX: {rx_weights}")
        print(f"  Layer {layer} RY: {ry_weights}")
    
    print("\n[Jet weights] (10 qubits, 13 rotations per layer: 10 RX + 3 RY)")
    print(f"  Shape: {encoder.jet_weights.shape}")
    weights = encoder.jet_weights.detach().cpu().numpy()
    n_layers = len(weights) // 13
    for layer in range(n_layers):
        rx_weights = weights[layer*13:layer*13+10]
        ry_weights = weights[layer*13+10:(layer+1)*13]
        print(f"  Layer {layer} RX (all 10): {rx_weights}")
        print(f"  Layer {layer} RY (first 3): {ry_weights}")
    
    # Also print classical layer weights
    print("\n[Dense hidden layer]")
    print(f"  Weight shape: {model.encoder.dense_hidden.weight.shape}")
    print(f"  Bias shape: {model.encoder.dense_hidden.bias.shape}")
    
    print("\n[z_mean layer]")
    print(f"  Weight shape: {model.encoder.z_mean.weight.shape}")
    
    print("\n[z_log_var layer]")
    print(f"  Weight shape: {model.encoder.z_log_var.weight.shape}")


def create_layerwise_circuits(depth):
    """
    Create circuits that return expectation values after each layer.
    Returns dict of circuit functions for each block.
    
    Uses the original architecture with ladder CNOTs and RX/RY/RZ rotations.
    """
    circuits = {}
    
    # -----------------------
    # MET circuits (1 qubit) - returns Z, X, Y
    # Uses RX, RY, RZ rotations (3 params per layer)
    # -----------------------
    met_dev = qml.device('default.qubit', wires=1)
    
    def make_met_circuit(n_layers):
        @qml.qnode(met_dev, interface='torch', diff_method='backprop')
        def met_after_n_layers(pt, phi, weights):
            qml.RY(pt, wires=0)
            qml.RZ(phi, wires=0)
            for d in range(n_layers):
                qml.RX(weights[d * 3], wires=0)
                qml.RY(weights[d * 3 + 1], wires=0)
                qml.RZ(weights[d * 3 + 2], wires=0)
            return [
                qml.expval(qml.PauliZ(0)),
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliY(0))
            ]
        return met_after_n_layers
    
    @qml.qnode(met_dev, interface='torch', diff_method='backprop')
    def met_after_encoding(pt, phi):
        qml.RY(pt, wires=0)
        qml.RZ(phi, wires=0)
        return [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliX(0)),
            qml.expval(qml.PauliY(0))
        ]
    
    circuits['met'] = {
        'after_encoding': met_after_encoding,
    }
    for d in range(1, depth + 1):
        circuits['met'][f'after_layer_{d}'] = make_met_circuit(d)
    
    # -----------------------
    # Electron circuits (4 qubits) - returns Z(4) + X(4) = 8 values
    # Uses ladder CNOTs and RX + RY rotations (8 params per layer)
    # -----------------------
    ele_dev = qml.device('default.qubit', wires=4)
    
    @qml.qnode(ele_dev, interface='torch', diff_method='backprop')
    def ele_after_encoding(pt_vec, eta_vec, phi_vec):
        for k in range(4):
            qml.RX(eta_vec[:, k], wires=k)
            qml.RY(pt_vec[:, k], wires=k)
            qml.RZ(phi_vec[:, k], wires=k)
        return [qml.expval(qml.PauliZ(k)) for k in range(4)] + \
               [qml.expval(qml.PauliX(k)) for k in range(4)]
    
    @qml.qnode(ele_dev, interface='torch', diff_method='backprop')
    def ele_after_initial_cnots(pt_vec, eta_vec, phi_vec):
        for k in range(4):
            qml.RX(eta_vec[:, k], wires=k)
            qml.RY(pt_vec[:, k], wires=k)
            qml.RZ(phi_vec[:, k], wires=k)
        # Initial ladder CNOTs
        for k in range(3):
            qml.CNOT(wires=[k, k+1])
        return [qml.expval(qml.PauliZ(k)) for k in range(4)] + \
               [qml.expval(qml.PauliX(k)) for k in range(4)]
    
    def make_ele_circuit(n_layers):
        @qml.qnode(ele_dev, interface='torch', diff_method='backprop')
        def ele_after_n_layers(pt_vec, eta_vec, phi_vec, weights):
            for k in range(4):
                qml.RX(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RZ(phi_vec[:, k], wires=k)
            # Initial ladder CNOTs
            for k in range(3):
                qml.CNOT(wires=[k, k+1])
            
            for d in range(n_layers):
                # RX rotations on all 4 qubits
                for k in range(4):
                    qml.RX(weights[d * 8 + k], wires=k)
                # RY rotations on all 4 qubits
                for k in range(4):
                    qml.RY(weights[d * 8 + 4 + k], wires=k)
                # Ladder CNOTs
                for k in range(3):
                    qml.CNOT(wires=[k, k+1])
            
            return [qml.expval(qml.PauliZ(k)) for k in range(4)] + \
                   [qml.expval(qml.PauliX(k)) for k in range(4)]
        return ele_after_n_layers
    
    circuits['ele'] = {
        'after_encoding': ele_after_encoding,
        'after_initial_cnots': ele_after_initial_cnots,
    }
    for d in range(1, depth + 1):
        circuits['ele'][f'after_layer_{d}'] = make_ele_circuit(d)
    
    # -----------------------
    # Muon circuits (4 qubits) - returns Z(4) + X(4) = 8 values
    # Uses ladder CNOTs and RX + RY rotations (8 params per layer)
    # -----------------------
    mu_dev = qml.device('default.qubit', wires=4)
    
    @qml.qnode(mu_dev, interface='torch', diff_method='backprop')
    def mu_after_encoding(pt_vec, eta_vec, phi_vec):
        for k in range(4):
            qml.RX(eta_vec[:, k], wires=k)
            qml.RY(pt_vec[:, k], wires=k)
            qml.RZ(phi_vec[:, k], wires=k)
        return [qml.expval(qml.PauliZ(k)) for k in range(4)] + \
               [qml.expval(qml.PauliX(k)) for k in range(4)]
    
    @qml.qnode(mu_dev, interface='torch', diff_method='backprop')
    def mu_after_initial_cnots(pt_vec, eta_vec, phi_vec):
        for k in range(4):
            qml.RX(eta_vec[:, k], wires=k)
            qml.RY(pt_vec[:, k], wires=k)
            qml.RZ(phi_vec[:, k], wires=k)
        # Initial ladder CNOTs
        for k in range(3):
            qml.CNOT(wires=[k, k+1])
        return [qml.expval(qml.PauliZ(k)) for k in range(4)] + \
               [qml.expval(qml.PauliX(k)) for k in range(4)]
    
    def make_mu_circuit(n_layers):
        @qml.qnode(mu_dev, interface='torch', diff_method='backprop')
        def mu_after_n_layers(pt_vec, eta_vec, phi_vec, weights):
            for k in range(4):
                qml.RX(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RZ(phi_vec[:, k], wires=k)
            # Initial ladder CNOTs
            for k in range(3):
                qml.CNOT(wires=[k, k+1])
            
            for d in range(n_layers):
                # RX rotations on all 4 qubits
                for k in range(4):
                    qml.RX(weights[d * 8 + k], wires=k)
                # RY rotations on all 4 qubits
                for k in range(4):
                    qml.RY(weights[d * 8 + 4 + k], wires=k)
                # Ladder CNOTs
                for k in range(3):
                    qml.CNOT(wires=[k, k+1])
            
            return [qml.expval(qml.PauliZ(k)) for k in range(4)] + \
                   [qml.expval(qml.PauliX(k)) for k in range(4)]
        return mu_after_n_layers
    
    circuits['mu'] = {
        'after_encoding': mu_after_encoding,
        'after_initial_cnots': mu_after_initial_cnots,
    }
    for d in range(1, depth + 1):
        circuits['mu'][f'after_layer_{d}'] = make_mu_circuit(d)
    
    # -----------------------
    # Jet circuits (10 qubits) - returns Z(10) + X(3) = 13 values
    # Uses ladder CNOTs and RX + RY rotations (13 params per layer)
    # -----------------------
    jet_dev = qml.device('default.qubit', wires=10)
    
    @qml.qnode(jet_dev, interface='torch', diff_method='backprop')
    def jet_after_encoding(pt_vec, eta_vec, phi_vec):
        for k in range(10):
            qml.RX(eta_vec[:, k], wires=k)
            qml.RY(pt_vec[:, k], wires=k)
            qml.RZ(phi_vec[:, k], wires=k)
        return [qml.expval(qml.PauliZ(k)) for k in range(10)] + \
               [qml.expval(qml.PauliX(k)) for k in range(3)]
    
    @qml.qnode(jet_dev, interface='torch', diff_method='backprop')
    def jet_after_initial_cnots(pt_vec, eta_vec, phi_vec):
        for k in range(10):
            qml.RX(eta_vec[:, k], wires=k)
            qml.RY(pt_vec[:, k], wires=k)
            qml.RZ(phi_vec[:, k], wires=k)
        # Initial ladder CNOTs
        for k in range(9):
            qml.CNOT(wires=[k, k+1])
        return [qml.expval(qml.PauliZ(k)) for k in range(10)] + \
               [qml.expval(qml.PauliX(k)) for k in range(3)]
    
    def make_jet_circuit(n_layers):
        @qml.qnode(jet_dev, interface='torch', diff_method='backprop')
        def jet_after_n_layers(pt_vec, eta_vec, phi_vec, weights):
            for k in range(10):
                qml.RX(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RZ(phi_vec[:, k], wires=k)
            # Initial ladder CNOTs
            for k in range(9):
                qml.CNOT(wires=[k, k+1])
            
            for d in range(n_layers):
                # RX rotations on all 10 qubits
                for k in range(10):
                    qml.RX(weights[d * 13 + k], wires=k)
                # RY rotations on first 3 qubits
                for k in range(3):
                    qml.RY(weights[d * 13 + 10 + k], wires=k)
                # Ladder CNOTs
                for k in range(9):
                    qml.CNOT(wires=[k, k+1])
            
            return [qml.expval(qml.PauliZ(k)) for k in range(10)] + \
                   [qml.expval(qml.PauliX(k)) for k in range(3)]
        return jet_after_n_layers
    
    circuits['jet'] = {
        'after_encoding': jet_after_encoding,
        'after_initial_cnots': jet_after_initial_cnots,
    }
    for d in range(1, depth + 1):
        circuits['jet'][f'after_layer_{d}'] = make_jet_circuit(d)
    
    return circuits


def get_layerwise_expectations(model, data, circuits, depth, batch_size=2048):
    """
    Get expectation values at each layer for a batch of data.
    
    Returns dict: {block: {stage: array of shape (n_samples, n_outputs)}}
    """
    encoder = model.encoder.quantum_encoder
    n_samples = data.shape[0]
    
    results = {
        'met': {},
        'ele': {},
        'mu': {},
        'jet': {},
    }
    
    # Initialize storage for each stage
    # MET doesn't have initial CNOTs (single qubit)
    met_stages = ['after_encoding'] + [f'after_layer_{d}' for d in range(1, depth + 1)]
    # Other blocks have initial CNOTs
    other_stages = ['after_encoding', 'after_initial_cnots'] + [f'after_layer_{d}' for d in range(1, depth + 1)]
    
    for stage in met_stages:
        results['met'][stage] = []
    for block in ['ele', 'mu', 'jet']:
        for stage in other_stages:
            results[block][stage] = []
    
    print(f"Processing {n_samples} samples in batches of {batch_size}...")
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = data[start_idx:end_idx]
            
            if isinstance(batch, np.ndarray):
                batch = torch.tensor(batch, dtype=torch.float64)
            
            # MET (batch processing with parameter broadcasting)
            pt = batch[:, 0]
            phi = batch[:, 1]
            
            # After encoding
            result = circuits['met']['after_encoding'](pt, phi)
            result_arr = torch.stack(result, dim=1).cpu().numpy()
            results['met']['after_encoding'].append(result_arr)
            
            # After each layer
            for d in range(1, depth + 1):
                result = circuits['met'][f'after_layer_{d}'](pt, phi, encoder.met_weights)
                result_arr = torch.stack(result, dim=1).cpu().numpy()
                results['met'][f'after_layer_{d}'].append(result_arr)
            
            # Electrons (batch processing)
            pt_vec = batch[:, 2:6]
            eta_vec = batch[:, 6:10]
            phi_vec = batch[:, 10:14]
            
            result = circuits['ele']['after_encoding'](pt_vec, eta_vec, phi_vec)
            result_arr = torch.stack(result, dim=1).cpu().numpy()
            results['ele']['after_encoding'].append(result_arr)
            
            result = circuits['ele']['after_initial_cnots'](pt_vec, eta_vec, phi_vec)
            result_arr = torch.stack(result, dim=1).cpu().numpy()
            results['ele']['after_initial_cnots'].append(result_arr)
            
            for d in range(1, depth + 1):
                result = circuits['ele'][f'after_layer_{d}'](pt_vec, eta_vec, phi_vec, encoder.ele_weights)
                result_arr = torch.stack(result, dim=1).cpu().numpy()
                results['ele'][f'after_layer_{d}'].append(result_arr)
            
            # Muons (batch processing)
            pt_vec = batch[:, 14:18]
            eta_vec = batch[:, 18:22]
            phi_vec = batch[:, 22:26]
            
            result = circuits['mu']['after_encoding'](pt_vec, eta_vec, phi_vec)
            result_arr = torch.stack(result, dim=1).cpu().numpy()
            results['mu']['after_encoding'].append(result_arr)
            
            result = circuits['mu']['after_initial_cnots'](pt_vec, eta_vec, phi_vec)
            result_arr = torch.stack(result, dim=1).cpu().numpy()
            results['mu']['after_initial_cnots'].append(result_arr)
            
            for d in range(1, depth + 1):
                result = circuits['mu'][f'after_layer_{d}'](pt_vec, eta_vec, phi_vec, encoder.mu_weights)
                result_arr = torch.stack(result, dim=1).cpu().numpy()
                results['mu'][f'after_layer_{d}'].append(result_arr)
            
            # Jets (batch processing)
            pt_vec = batch[:, 26:36]
            eta_vec = batch[:, 36:46]
            phi_vec = batch[:, 46:56]
            
            result = circuits['jet']['after_encoding'](pt_vec, eta_vec, phi_vec)
            result_arr = torch.stack(result, dim=1).cpu().numpy()
            results['jet']['after_encoding'].append(result_arr)
            
            result = circuits['jet']['after_initial_cnots'](pt_vec, eta_vec, phi_vec)
            result_arr = torch.stack(result, dim=1).cpu().numpy()
            results['jet']['after_initial_cnots'].append(result_arr)
            
            for d in range(1, depth + 1):
                result = circuits['jet'][f'after_layer_{d}'](pt_vec, eta_vec, phi_vec, encoder.jet_weights)
                result_arr = torch.stack(result, dim=1).cpu().numpy()
                results['jet'][f'after_layer_{d}'].append(result_arr)
            
            if (end_idx % 1000) == 0 or end_idx == n_samples:
                print(f"  Processed {end_idx}/{n_samples} samples")
    
    # Concatenate results
    for block in results:
        for stage in results[block]:
            results[block][stage] = np.concatenate(results[block][stage], axis=0)
    
    return results


def plot_expectation_distributions(all_results, block, depth, output_labels, title_prefix, save_path=None):
    """
    Plot distributions of expectation values at each layer stage.
    Compares background vs all signals (similar to visualize_encoding.py style).
    
    all_results: dict {dataset_name: {block: {stage: array}}}
    """
    # MET doesn't have initial CNOTs (single qubit)
    if block == 'met':
        stages = ['after_encoding'] + [f'after_layer_{d}' for d in range(1, depth + 1)]
    else:
        stages = ['after_encoding', 'after_initial_cnots'] + [f'after_layer_{d}' for d in range(1, depth + 1)]
    n_stages = len(stages)
    
    # Get number of outputs from first dataset
    first_dataset = next(iter(all_results.values()))
    n_outputs = first_dataset[block]['after_encoding'].shape[1]
    
    # Create figure with subplots: rows = outputs, cols = stages
    fig, axes = plt.subplots(n_outputs, n_stages, figsize=(4 * n_stages, 3 * n_outputs))
    if n_outputs == 1:
        axes = axes.reshape(1, -1)
    
    for col, stage in enumerate(stages):
        for row in range(n_outputs):
            ax = axes[row, col]
            
            # Plot each dataset
            for dataset_name, results in all_results.items():
                data = results[block][stage]
                values = data[:, row]
                
                alpha = 0.5 if dataset_name == 'Background' else 0.7
                linewidth = 1.5 if dataset_name == 'Background' else 2
                color = COLORS.get(dataset_name, 'black')
                
                ax.hist(values, bins=50, alpha=alpha, label=dataset_name,
                       color=color, density=True, histtype='step', linewidth=linewidth)
            
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlim(-1.1, 1.1)
            ax.set_xlabel('Expectation value')
            ax.set_ylabel('Density')
            
            if row == 0:
                stage_name = stage.replace('_', ' ').title()
                ax.set_title(stage_name)
            
            if col == 0:
                ax.set_ylabel(f'{output_labels[row]}\nDensity')
            
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
            
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{title_prefix}: Expectation Value Distributions by Layer', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {save_path}")
    
    return fig


def plot_mean_evolution(all_results, depth, save_path=None):
    """
    Plot how mean expectation values evolve through layers for each block.
    Compares background vs signals.
    """
    met_stages = ['after_encoding'] + [f'after_layer_{d}' for d in range(1, depth + 1)]
    other_stages = ['after_encoding', 'after_initial_cnots'] + [f'after_layer_{d}' for d in range(1, depth + 1)]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MET - compare datasets (no initial CNOTs)
    ax = axes[0, 0]
    x_met = np.arange(len(met_stages))
    for dataset_name, results in all_results.items():
        color = COLORS.get(dataset_name, 'black')
        # Average over Z, X, Y
        means = [results['met'][stage].mean() for stage in met_stages]
        ax.plot(x_met, means, 'o-', label=dataset_name, color=color, alpha=0.7)
    ax.set_xticks(x_met)
    ax.set_xticklabels([s.replace('after_', '').replace('_', ' ') for s in met_stages], rotation=45)
    ax.set_ylabel('Mean expectation')
    ax.set_title('MET (1 qubit)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    
    # Electrons (with initial CNOTs)
    ax = axes[0, 1]
    x_other = np.arange(len(other_stages))
    for dataset_name, results in all_results.items():
        color = COLORS.get(dataset_name, 'black')
        z_means = [results['ele'][stage][:, :4].mean() for stage in other_stages]
        ax.plot(x_other, z_means, 'o-', label=f'{dataset_name}', color=color, alpha=0.7)
    ax.set_xticks(x_other)
    ax.set_xticklabels([s.replace('after_', '').replace('_', ' ') for s in other_stages], rotation=45)
    ax.set_ylabel('Mean PauliZ expectation')
    ax.set_title('Electrons (4 qubits)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    
    # Muons (with initial CNOTs)
    ax = axes[1, 0]
    for dataset_name, results in all_results.items():
        color = COLORS.get(dataset_name, 'black')
        z_means = [results['mu'][stage][:, :4].mean() for stage in other_stages]
        ax.plot(x_other, z_means, 'o-', label=f'{dataset_name}', color=color, alpha=0.7)
    ax.set_xticks(x_other)
    ax.set_xticklabels([s.replace('after_', '').replace('_', ' ') for s in other_stages], rotation=45)
    ax.set_ylabel('Mean PauliZ expectation')
    ax.set_title('Muons (4 qubits)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    
    # Jets (with initial CNOTs)
    ax = axes[1, 1]
    for dataset_name, results in all_results.items():
        color = COLORS.get(dataset_name, 'black')
        z_means = [results['jet'][stage][:, :10].mean() for stage in other_stages]
        ax.plot(x_other, z_means, 'o-', label=f'{dataset_name}', color=color, alpha=0.7)
    ax.set_xticks(x_other)
    ax.set_xticklabels([s.replace('after_', '').replace('_', ' ') for s in other_stages], rotation=45)
    ax.set_ylabel('Mean PauliZ expectation')
    ax.set_title('Jets (10 qubits)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    
    plt.suptitle('Mean Expectation Value Evolution Through Layers (Background vs Signals)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {save_path}")
    
    return fig


def plot_full_encoding_comparison(all_results, depth, save_path=None):
    """
    Plot the full 32D encoding distribution at each layer stage.
    Compares background vs signals (similar to visualize_encoding.py).
    
    Note: Uses the 'after_layer_N' stages for full encoding comparison since
    MET doesn't have initial CNOTs but other blocks do.
    """
    # Use layer stages for comparison (after all blocks have been processed)
    stages = ['after_encoding'] + [f'after_layer_{d}' for d in range(1, depth + 1)]
    
    # Labels for 32 outputs
    labels = (
        ['MET_Z', 'MET_X', 'MET_Y'] +
        [f'e{i}_Z' for i in range(4)] + [f'e{i}_X' for i in range(4)] +
        [f'μ{i}_Z' for i in range(4)] + [f'μ{i}_X' for i in range(4)] +
        [f'j{i}_Z' for i in range(10)] + [f'j{i}_X' for i in range(3)]
    )
    
    n_stages = len(stages)
    fig, axes = plt.subplots(n_stages, 1, figsize=(18, 4 * n_stages))
    if n_stages == 1:
        axes = [axes]
    
    for idx, stage in enumerate(stages):
        ax = axes[idx]
        
        x = np.arange(32)
        width = 0.15
        n_datasets = len(all_results)
        
        for i, (dataset_name, results) in enumerate(all_results.items()):
            # Concatenate all outputs
            met = results['met'][stage]
            ele = results['ele'][stage]
            mu = results['mu'][stage]
            jet = results['jet'][stage]
            full_encoding = np.concatenate([met, ele, mu, jet], axis=1)
            
            means = full_encoding.mean(axis=0)
            offset = (i - n_datasets/2 + 0.5) * width
            color = COLORS.get(dataset_name, 'black')
            ax.bar(x + offset, means, width, label=dataset_name, color=color, alpha=0.7)
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_ylabel('Mean')
        ax.set_ylim(-1.2, 1.2)
        
        stage_name = stage.replace('_', ' ').title()
        ax.set_title(f'{stage_name}')
        ax.grid(True, alpha=0.3, axis='y')
        
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    plt.suptitle('Full 32D Encoding Evolution: Background vs Signals', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {save_path}")
    
    return fig


def get_full_encoding_at_stage(results, stage):
    """Helper to concatenate all block outputs into full 32D encoding."""
    met = results['met'][stage]
    ele = results['ele'][stage]
    mu = results['mu'][stage]
    jet = results['jet'][stage]
    return np.concatenate([met, ele, mu, jet], axis=1)


def plot_pca_by_stage(all_results, depth, output_dir, n_samples_per_class=5000):
    """
    Create PCA visualization at each layer stage, comparing background vs signals.
    """
    # Use layer stages for PCA (after all blocks have been processed)
    stages = ['after_encoding'] + [f'after_layer_{d}' for d in range(1, depth + 1)]
    
    for stage in stages:
        print(f"  Creating PCA for {stage}...")
        
        # Collect encodings from all datasets
        all_encodings = []
        all_labels = []
        
        for dataset_name, results in all_results.items():
            encoding = get_full_encoding_at_stage(results, stage)
            n = min(n_samples_per_class, len(encoding))
            indices = np.random.choice(len(encoding), n, replace=False)
            all_encodings.append(encoding[indices])
            all_labels.extend([dataset_name] * n)
        
        all_encodings = np.vstack(all_encodings)
        
        # PCA
        pca = PCA(n_components=2)
        embeddings = pca.fit_transform(all_encodings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        start_idx = 0
        for dataset_name, results in all_results.items():
            encoding = get_full_encoding_at_stage(results, stage)
            n = min(n_samples_per_class, len(encoding))
            end_idx = start_idx + n
            
            color = COLORS.get(dataset_name, 'black')
            alpha = 0.3 if dataset_name == 'Background' else 0.6
            plt.scatter(embeddings[start_idx:end_idx, 0], 
                       embeddings[start_idx:end_idx, 1],
                       c=color, label=dataset_name, alpha=alpha, s=10)
            start_idx = end_idx
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        stage_name = stage.replace('_', ' ').title()
        plt.title(f'PCA of 32D Encoding - {stage_name}', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(output_dir, f'pca_{stage}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved {save_path}")
        plt.close()


def plot_tsne_by_stage(all_results, depth, output_dir, n_samples_per_class=3000):
    """
    Create t-SNE visualization at each layer stage, comparing background vs signals.
    """
    # Use layer stages for t-SNE (after all blocks have been processed)
    stages = ['after_encoding'] + [f'after_layer_{d}' for d in range(1, depth + 1)]
    
    for stage in stages:
        print(f"  Creating t-SNE for {stage}...")
        
        # Collect encodings from all datasets
        all_encodings = []
        all_labels = []
        
        for dataset_name, results in all_results.items():
            encoding = get_full_encoding_at_stage(results, stage)
            n = min(n_samples_per_class, len(encoding))
            indices = np.random.choice(len(encoding), n, replace=False)
            all_encodings.append(encoding[indices])
            all_labels.extend([dataset_name] * n)
        
        all_encodings = np.vstack(all_encodings)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000)
        embeddings = tsne.fit_transform(all_encodings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        start_idx = 0
        for dataset_name, results in all_results.items():
            encoding = get_full_encoding_at_stage(results, stage)
            n = min(n_samples_per_class, len(encoding))
            end_idx = start_idx + n
            
            color = COLORS.get(dataset_name, 'black')
            alpha = 0.3 if dataset_name == 'Background' else 0.6
            plt.scatter(embeddings[start_idx:end_idx, 0], 
                       embeddings[start_idx:end_idx, 1],
                       c=color, label=dataset_name, alpha=alpha, s=10)
            start_idx = end_idx
        
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        stage_name = stage.replace('_', ' ').title()
        plt.title(f't-SNE of 32D Encoding - {stage_name}', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(output_dir, f'tsne_{stage}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved {save_path}")
        plt.close()


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


def main():
    parser = argparse.ArgumentParser(description='Inspect Extended Quantum VAE model')
    parser.add_argument('--model-path', default='outputs/models/particle_extended_quantum_vae_best.pt',
                       help='Path to Extended Quantum VAE model checkpoint')
    parser.add_argument('--max-samples', type=int, default=10000,
                       help='Number of samples to process per dataset')
    parser.add_argument('--output-dir', default='outputs/inspection_extended',
                       help='Directory to save plots')
    parser.add_argument('--depth', type=int, default=None,
                       help='Circuit depth (auto-detected from model if not set)')
    parser.add_argument('--skip-tsne', action='store_true',
                       help='Skip t-SNE visualization (can be slow)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    device = torch.device('cpu')
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_config = checkpoint.get('model_config', {})
    
    # Get depth from config or argument
    depth = args.depth if args.depth else model_config.get('quantum_depth', 2)
    print(f"Using circuit depth: {depth}")
    
    model = create_extended_quantum_vae(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")
    
    # Print weights
    print_model_weights(model)
    
    # Create layerwise circuits
    print("\nCreating layerwise inspection circuits...")
    circuits = create_layerwise_circuits(depth)
    
    # Load and process all datasets
    all_results = {}
    
    # Background (use samples from evaluation set, starting at 2M)
    print("\n" + "="*60)
    print("Loading and processing Background data...")
    print("="*60)
    bg_data = load_data(BACKGROUND_PATH, start_idx=2_000_000, max_samples=args.max_samples)
    print(f"Computing expectations at each layer for {len(bg_data)} samples...")
    all_results['Background'] = get_layerwise_expectations(model, bg_data, circuits, depth)
    
    # Signals
    for path, label in zip(SIGNAL_PATHS, SIGNAL_LABELS):
        if os.path.exists(path):
            print(f"\n" + "="*60)
            print(f"Loading and processing {label} data...")
            print("="*60)
            sig_data = load_data(path, max_samples=args.max_samples)
            print(f"Computing expectations at each layer for {len(sig_data)} samples...")
            all_results[label] = get_layerwise_expectations(model, sig_data, circuits, depth)
        else:
            print(f"\nWarning: {path} not found, skipping {label}")
    
    # Create visualizations
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)
    
    # 1. Distribution plots for each block (background vs signals)
    print("\n1. Creating distribution plots...")
    
    met_labels = ['PauliZ', 'PauliX', 'PauliY']
    fig_met = plot_expectation_distributions(
        all_results, 'met', depth, met_labels, 'MET',
        save_path=os.path.join(args.output_dir, 'met_distributions.png')
    )
    plt.close(fig_met)
    
    ele_labels = [f'Z{i}' for i in range(4)] + [f'X{i}' for i in range(4)]
    fig_ele = plot_expectation_distributions(
        all_results, 'ele', depth, ele_labels, 'Electrons',
        save_path=os.path.join(args.output_dir, 'electron_distributions.png')
    )
    plt.close(fig_ele)
    
    mu_labels = [f'Z{i}' for i in range(4)] + [f'X{i}' for i in range(4)]
    fig_mu = plot_expectation_distributions(
        all_results, 'mu', depth, mu_labels, 'Muons',
        save_path=os.path.join(args.output_dir, 'muon_distributions.png')
    )
    plt.close(fig_mu)
    
    jet_labels = [f'Z{i}' for i in range(10)] + [f'X{i}' for i in range(3)]
    fig_jet = plot_expectation_distributions(
        all_results, 'jet', depth, jet_labels, 'Jets',
        save_path=os.path.join(args.output_dir, 'jet_distributions.png')
    )
    plt.close(fig_jet)
    
    # 2. Mean evolution through layers
    print("\n2. Creating mean evolution plot...")
    fig_evolution = plot_mean_evolution(
        all_results, depth,
        save_path=os.path.join(args.output_dir, 'mean_evolution.png')
    )
    plt.close(fig_evolution)
    
    # 3. Full 32D encoding comparison
    print("\n3. Creating full 32D encoding comparison...")
    fig_full = plot_full_encoding_comparison(
        all_results, depth,
        save_path=os.path.join(args.output_dir, 'full_encoding_evolution.png')
    )
    plt.close(fig_full)
    
    # 4. PCA at each stage
    print("\n4. Creating PCA visualizations at each stage...")
    plot_pca_by_stage(all_results, depth, args.output_dir)
    
    # 5. t-SNE at each stage (optional)
    if not args.skip_tsne:
        print("\n5. Creating t-SNE visualizations at each stage (this may take a while)...")
        plot_tsne_by_stage(all_results, depth, args.output_dir)
    else:
        print("\n5. Skipping t-SNE visualizations (use without --skip-tsne to enable)")
    
    print(f"\n" + "="*60)
    print(f"Inspection complete! Plots saved to {args.output_dir}/")
    print("="*60)
    print("\nGenerated plots:")
    print("  - met_distributions.png")
    print("  - electron_distributions.png")
    print("  - muon_distributions.png")
    print("  - jet_distributions.png")
    print("  - mean_evolution.png")
    print("  - full_encoding_evolution.png")
    print("  - pca_after_encoding.png, pca_after_layer_1.png, ...")
    if not args.skip_tsne:
        print("  - tsne_after_encoding.png, tsne_after_layer_1.png, ...")


if __name__ == "__main__":
    main()
