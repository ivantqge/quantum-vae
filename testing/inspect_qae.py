#!/usr/bin/env python3
"""
Inspect QAE model weights and intermediate expectation values.

This script:
1. Loads a trained QAE model and prints all trainable weights
2. For background and signal data, shows the expectation values after each layer
3. Visualizes the distribution of expectation values at different circuit stages
   comparing background vs signals (similar to inspect_extended_vae.py)
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
from block_quantum_ae import ParticleQAEAnomalyModel

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
    """Print all trainable weights in the QAE model."""
    print("\n" + "="*60)
    print("MODEL WEIGHTS")
    print("="*60)
    
    encoder = model.encoder
    
    print("\n[MET weights] (1 qubit, depth=1)")
    print(f"  Shape: {encoder.met_weights.shape}")
    print(f"  Values: {encoder.met_weights.detach().cpu().numpy()}")
    
    print("\n[Electron weights] (4 qubits, depth=1)")
    print(f"  Shape: {encoder.ele_weights.shape}")
    weights = encoder.ele_weights.detach().cpu().numpy()
    print(f"  Values: {weights}")
    
    print("\n[Muon weights] (4 qubits, depth=1)")
    print(f"  Shape: {encoder.mu_weights.shape}")
    weights = encoder.mu_weights.detach().cpu().numpy()
    print(f"  Values: {weights}")
    
    print("\n[Jet weights] (10 qubits, depth=4)")
    print(f"  Shape: {encoder.jet_weights.shape}")
    weights = encoder.jet_weights.detach().cpu().numpy()
    n_layers = len(weights) // 10
    for layer in range(n_layers):
        layer_weights = weights[layer*10:(layer+1)*10]
        print(f"  Layer {layer}: {layer_weights}")


def create_layerwise_circuits():
    """
    Create circuits that return expectation values after each layer.
    Returns dict of circuit functions for each block.
    
    Uses all-to-all CNOT connections between latent and trash qubits.
    """
    circuits = {}
    
    # -----------------------
    # MET circuits (1 qubit) - returns Z expectation
    # -----------------------
    met_dev = qml.device('default.qubit', wires=1)
    
    @qml.qnode(met_dev, interface='torch', diff_method='backprop')
    def met_after_encoding(pt, phi):
        qml.RY(pt, wires=0)
        qml.RZ(phi, wires=0)
        return qml.expval(qml.PauliZ(0))
    
    @qml.qnode(met_dev, interface='torch', diff_method='backprop')
    def met_final(pt, phi, weights):
        qml.RY(pt, wires=0)
        qml.RZ(phi, wires=0)
        qml.RY(weights[0], wires=0)
        return qml.expval(qml.PauliZ(0))
    
    circuits['met'] = {
        'after_encoding': met_after_encoding,
        'final': met_final,
    }
    
    # -----------------------
    # Electron circuits (4 qubits) - returns Z on trash qubits (2, 3)
    # All-to-all CNOTs between latent (0,1) and trash (2,3)
    # -----------------------
    ele_dev = qml.device('default.qubit', wires=4)
    ele_trash = [2, 3]
    ele_latent = [0, 1]
    
    @qml.qnode(ele_dev, interface='torch', diff_method='backprop')
    def ele_after_encoding(pt_vec, eta_vec, phi_vec):
        for k in range(4):
            qml.RX(eta_vec[:, k], wires=k)
            qml.RY(pt_vec[:, k], wires=k)
            qml.RZ(phi_vec[:, k], wires=k)
        return [qml.expval(qml.PauliZ(w)) for w in ele_trash]
    
    @qml.qnode(ele_dev, interface='torch', diff_method='backprop')
    def ele_after_cnot1(pt_vec, eta_vec, phi_vec):
        for k in range(4):
            qml.RX(eta_vec[:, k], wires=k)
            qml.RY(pt_vec[:, k], wires=k)
            qml.RZ(phi_vec[:, k], wires=k)
        # trash -> latent (all-to-all)
        for t in ele_trash:
            for l in ele_latent:
                qml.CNOT(wires=[t, l])
        return [qml.expval(qml.PauliZ(w)) for w in ele_trash]
    
    @qml.qnode(ele_dev, interface='torch', diff_method='backprop')
    def ele_after_ry(pt_vec, eta_vec, phi_vec, weights):
        for k in range(4):
            qml.RX(eta_vec[:, k], wires=k)
            qml.RY(pt_vec[:, k], wires=k)
            qml.RZ(phi_vec[:, k], wires=k)
        for t in ele_trash:
            for l in ele_latent:
                qml.CNOT(wires=[t, l])
        for k in range(4):
            qml.RY(weights[k], wires=k)
        return [qml.expval(qml.PauliZ(w)) for w in ele_trash]
    
    @qml.qnode(ele_dev, interface='torch', diff_method='backprop')
    def ele_final(pt_vec, eta_vec, phi_vec, weights):
        for k in range(4):
            qml.RX(eta_vec[:, k], wires=k)
            qml.RY(pt_vec[:, k], wires=k)
            qml.RZ(phi_vec[:, k], wires=k)
        for t in ele_trash:
            for l in ele_latent:
                qml.CNOT(wires=[t, l])
        for k in range(4):
            qml.RY(weights[k], wires=k)
        # latent -> trash (all-to-all)
        for l in ele_latent:
            for t in ele_trash:
                qml.CNOT(wires=[l, t])
        return [qml.expval(qml.PauliZ(w)) for w in ele_trash]
    
    circuits['ele'] = {
        'after_encoding': ele_after_encoding,
        'after_cnot1': ele_after_cnot1,
        'after_ry': ele_after_ry,
        'final': ele_final,
    }
    
    # -----------------------
    # Muon circuits (4 qubits) - same structure as electrons
    # -----------------------
    mu_dev = qml.device('default.qubit', wires=4)
    mu_trash = [2, 3]
    mu_latent = [0, 1]
    
    @qml.qnode(mu_dev, interface='torch', diff_method='backprop')
    def mu_after_encoding(pt_vec, eta_vec, phi_vec):
        for k in range(4):
            qml.RX(eta_vec[:, k], wires=k)
            qml.RY(pt_vec[:, k], wires=k)
            qml.RZ(phi_vec[:, k], wires=k)
        return [qml.expval(qml.PauliZ(w)) for w in mu_trash]
    
    @qml.qnode(mu_dev, interface='torch', diff_method='backprop')
    def mu_after_cnot1(pt_vec, eta_vec, phi_vec):
        for k in range(4):
            qml.RX(eta_vec[:, k], wires=k)
            qml.RY(pt_vec[:, k], wires=k)
            qml.RZ(phi_vec[:, k], wires=k)
        for t in mu_trash:
            for l in mu_latent:
                qml.CNOT(wires=[t, l])
        return [qml.expval(qml.PauliZ(w)) for w in mu_trash]
    
    @qml.qnode(mu_dev, interface='torch', diff_method='backprop')
    def mu_after_ry(pt_vec, eta_vec, phi_vec, weights):
        for k in range(4):
            qml.RX(eta_vec[:, k], wires=k)
            qml.RY(pt_vec[:, k], wires=k)
            qml.RZ(phi_vec[:, k], wires=k)
        for t in mu_trash:
            for l in mu_latent:
                qml.CNOT(wires=[t, l])
        for k in range(4):
            qml.RY(weights[k], wires=k)
        return [qml.expval(qml.PauliZ(w)) for w in mu_trash]
    
    @qml.qnode(mu_dev, interface='torch', diff_method='backprop')
    def mu_final(pt_vec, eta_vec, phi_vec, weights):
        for k in range(4):
            qml.RX(eta_vec[:, k], wires=k)
            qml.RY(pt_vec[:, k], wires=k)
            qml.RZ(phi_vec[:, k], wires=k)
        for t in mu_trash:
            for l in mu_latent:
                qml.CNOT(wires=[t, l])
        for k in range(4):
            qml.RY(weights[k], wires=k)
        for l in mu_latent:
            for t in mu_trash:
                qml.CNOT(wires=[l, t])
        return [qml.expval(qml.PauliZ(w)) for w in mu_trash]
    
    circuits['mu'] = {
        'after_encoding': mu_after_encoding,
        'after_cnot1': mu_after_cnot1,
        'after_ry': mu_after_ry,
        'final': mu_final,
    }
    
    # -----------------------
    # Jet circuits (10 qubits) - returns Z on trash qubits (6,7,8,9)
    # All-to-all CNOTs between latent (0-5) and trash (6-9)
    # Depth = 4
    # -----------------------
    jet_dev = qml.device('default.qubit', wires=10)
    jet_latent = [0, 1, 2, 3, 4, 5]
    jet_trash = [6, 7, 8, 9]
    
    @qml.qnode(jet_dev, interface='torch', diff_method='backprop')
    def jet_after_encoding(pt_vec, eta_vec, phi_vec):
        for k in range(10):
            qml.RX(eta_vec[:, k], wires=k)
            qml.RY(pt_vec[:, k], wires=k)
            qml.RZ(phi_vec[:, k], wires=k)
        return [qml.expval(qml.PauliZ(w)) for w in jet_trash]
    
    def make_jet_circuit(n_layers):
        @qml.qnode(jet_dev, interface='torch', diff_method='backprop')
        def jet_after_n_layers(pt_vec, eta_vec, phi_vec, weights):
            for k in range(10):
                qml.RX(eta_vec[:, k], wires=k)
                qml.RY(pt_vec[:, k], wires=k)
                qml.RZ(phi_vec[:, k], wires=k)
            
            for d in range(n_layers):
                # trash -> latent (all-to-all)
                for t in jet_trash:
                    for l in jet_latent:
                        qml.CNOT(wires=[t, l])
                # RY
                base = d * 10
                for k in range(10):
                    qml.RY(weights[base + k], wires=k)
                # latent -> trash (all-to-all)
                for l in jet_latent:
                    for t in jet_trash:
                        qml.CNOT(wires=[l, t])
            
            return [qml.expval(qml.PauliZ(w)) for w in jet_trash]
        return jet_after_n_layers
    
    circuits['jet'] = {
        'after_encoding': jet_after_encoding,
        'after_layer_1': make_jet_circuit(1),
        'after_layer_2': make_jet_circuit(2),
        'after_layer_3': make_jet_circuit(3),
        'after_layer_4': make_jet_circuit(4),
    }
    
    return circuits


def get_layerwise_expectations(model, data, circuits, batch_size=2048):
    """
    Get expectation values at each layer for a batch of data.
    
    Returns dict: {block: {stage: array of shape (n_samples, n_outputs)}}
    """
    encoder = model.encoder
    n_samples = data.shape[0]
    
    results = {
        'met': {},
        'ele': {},
        'mu': {},
        'jet': {},
    }
    
    # Initialize storage for each stage
    met_stages = ['after_encoding', 'final']
    ele_mu_stages = ['after_encoding', 'after_cnot1', 'after_ry', 'final']
    jet_stages = ['after_encoding', 'after_layer_1', 'after_layer_2', 'after_layer_3', 'after_layer_4']
    
    for stage in met_stages:
        results['met'][stage] = []
    for stage in ele_mu_stages:
        results['ele'][stage] = []
        results['mu'][stage] = []
    for stage in jet_stages:
        results['jet'][stage] = []
    
    print(f"Processing {n_samples} samples in batches of {batch_size}...")
    
    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = data[start_idx:end_idx]
            
            if isinstance(batch, np.ndarray):
                batch = torch.tensor(batch, dtype=torch.float64)
            
            # MET (batch processing)
            pt = batch[:, 0]
            phi = batch[:, 1]
            
            result = circuits['met']['after_encoding'](pt, phi)
            results['met']['after_encoding'].append(result.unsqueeze(1).cpu().numpy())
            
            result = circuits['met']['final'](pt, phi, encoder.met_weights)
            results['met']['final'].append(result.unsqueeze(1).cpu().numpy())
            
            # Electrons (batch processing)
            pt_vec = batch[:, 2:6]
            eta_vec = batch[:, 6:10]
            phi_vec = batch[:, 10:14]
            
            result = circuits['ele']['after_encoding'](pt_vec, eta_vec, phi_vec)
            result_arr = torch.stack(result, dim=1).cpu().numpy()
            results['ele']['after_encoding'].append(result_arr)
            
            result = circuits['ele']['after_cnot1'](pt_vec, eta_vec, phi_vec)
            result_arr = torch.stack(result, dim=1).cpu().numpy()
            results['ele']['after_cnot1'].append(result_arr)
            
            result = circuits['ele']['after_ry'](pt_vec, eta_vec, phi_vec, encoder.ele_weights)
            result_arr = torch.stack(result, dim=1).cpu().numpy()
            results['ele']['after_ry'].append(result_arr)
            
            result = circuits['ele']['final'](pt_vec, eta_vec, phi_vec, encoder.ele_weights)
            result_arr = torch.stack(result, dim=1).cpu().numpy()
            results['ele']['final'].append(result_arr)
            
            # Muons (batch processing)
            pt_vec = batch[:, 14:18]
            eta_vec = batch[:, 18:22]
            phi_vec = batch[:, 22:26]
            
            result = circuits['mu']['after_encoding'](pt_vec, eta_vec, phi_vec)
            result_arr = torch.stack(result, dim=1).cpu().numpy()
            results['mu']['after_encoding'].append(result_arr)
            
            result = circuits['mu']['after_cnot1'](pt_vec, eta_vec, phi_vec)
            result_arr = torch.stack(result, dim=1).cpu().numpy()
            results['mu']['after_cnot1'].append(result_arr)
            
            result = circuits['mu']['after_ry'](pt_vec, eta_vec, phi_vec, encoder.mu_weights)
            result_arr = torch.stack(result, dim=1).cpu().numpy()
            results['mu']['after_ry'].append(result_arr)
            
            result = circuits['mu']['final'](pt_vec, eta_vec, phi_vec, encoder.mu_weights)
            result_arr = torch.stack(result, dim=1).cpu().numpy()
            results['mu']['final'].append(result_arr)
            
            # Jets (batch processing)
            pt_vec = batch[:, 26:36]
            eta_vec = batch[:, 36:46]
            phi_vec = batch[:, 46:56]
            
            result = circuits['jet']['after_encoding'](pt_vec, eta_vec, phi_vec)
            result_arr = torch.stack(result, dim=1).cpu().numpy()
            results['jet']['after_encoding'].append(result_arr)
            
            for d in range(1, 5):
                result = circuits['jet'][f'after_layer_{d}'](pt_vec, eta_vec, phi_vec, encoder.jet_weights)
                result_arr = torch.stack(result, dim=1).cpu().numpy()
                results['jet'][f'after_layer_{d}'].append(result_arr)
            
            if (end_idx % 2000) == 0 or end_idx == n_samples:
                print(f"  Processed {end_idx}/{n_samples} samples")
    
    # Concatenate results
    for block in results:
        for stage in results[block]:
            results[block][stage] = np.concatenate(results[block][stage], axis=0)
    
    return results


def plot_expectation_distributions(all_results, block, output_labels, title_prefix, save_path=None):
    """
    Plot distributions of expectation values at each layer stage.
    Compares background vs all signals.
    """
    # Get stages for this block
    first_dataset = next(iter(all_results.values()))
    stages = list(first_dataset[block].keys())
    n_stages = len(stages)
    n_outputs = first_dataset[block][stages[0]].shape[1]
    
    fig, axes = plt.subplots(n_outputs, n_stages, figsize=(4 * n_stages, 3 * n_outputs))
    if n_outputs == 1:
        axes = axes.reshape(1, -1)
    
    for col, stage in enumerate(stages):
        for row in range(n_outputs):
            ax = axes[row, col]
            
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
    
    plt.suptitle(f'{title_prefix}: Trash Qubit Z Expectation Distributions by Stage', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {save_path}")
    
    return fig


def plot_trash_excitation_evolution(all_results, save_path=None):
    """
    Plot how trash qubit excitation P(|1>) = (1-<Z>)/2 evolves through stages.
    Compares background vs signals.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MET
    ax = axes[0, 0]
    stages = ['after_encoding', 'final']
    x = np.arange(len(stages))
    for dataset_name, results in all_results.items():
        color = COLORS.get(dataset_name, 'black')
        # Convert Z to P(|1>)
        p1_means = [0.5 * (1 - results['met'][stage].mean()) for stage in stages]
        ax.plot(x, p1_means, 'o-', label=dataset_name, color=color, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in stages])
    ax.set_ylabel('Mean P(|1>)')
    ax.set_title('MET (1 qubit)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Electrons
    ax = axes[0, 1]
    stages = ['after_encoding', 'after_cnot1', 'after_ry', 'final']
    x = np.arange(len(stages))
    for dataset_name, results in all_results.items():
        color = COLORS.get(dataset_name, 'black')
        p1_means = [0.5 * (1 - results['ele'][stage].mean()) for stage in stages]
        ax.plot(x, p1_means, 'o-', label=dataset_name, color=color, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').replace('after ', '').title() for s in stages], rotation=30)
    ax.set_ylabel('Mean P(|1>) on trash')
    ax.set_title('Electrons (trash qubits 2,3)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Muons
    ax = axes[1, 0]
    for dataset_name, results in all_results.items():
        color = COLORS.get(dataset_name, 'black')
        p1_means = [0.5 * (1 - results['mu'][stage].mean()) for stage in stages]
        ax.plot(x, p1_means, 'o-', label=dataset_name, color=color, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').replace('after ', '').title() for s in stages], rotation=30)
    ax.set_ylabel('Mean P(|1>) on trash')
    ax.set_title('Muons (trash qubits 2,3)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Jets
    ax = axes[1, 1]
    stages = ['after_encoding', 'after_layer_1', 'after_layer_2', 'after_layer_3', 'after_layer_4']
    x = np.arange(len(stages))
    for dataset_name, results in all_results.items():
        color = COLORS.get(dataset_name, 'black')
        p1_means = [0.5 * (1 - results['jet'][stage].mean()) for stage in stages]
        ax.plot(x, p1_means, 'o-', label=dataset_name, color=color, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['Encoding', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'], rotation=30)
    ax.set_ylabel('Mean P(|1>) on trash')
    ax.set_title('Jets (trash qubits 6,7,8,9)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.suptitle('Trash Qubit Excitation Evolution: Background vs Signals', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {save_path}")
    
    return fig


def plot_block_scores_comparison(all_results, save_path=None):
    """
    Plot the final block scores (mean trash excitation per block) for background vs signals.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    blocks = ['met', 'ele', 'mu', 'jet']
    block_labels = ['MET', 'Electrons', 'Muons', 'Jets']
    x = np.arange(len(blocks))
    width = 0.15
    n_datasets = len(all_results)
    
    for i, (dataset_name, results) in enumerate(all_results.items()):
        # Get final stage for each block
        scores = []
        stds = []
        for block in blocks:
            if block == 'met':
                final_z = results['met']['final']
            elif block == 'jet':
                final_z = results['jet']['after_layer_4']
            else:
                final_z = results[block]['final']
            
            # Convert to P(|1>) = (1 - <Z>) / 2
            p1 = 0.5 * (1 - final_z)
            scores.append(p1.mean())
            stds.append(p1.std())
        
        offset = (i - n_datasets/2 + 0.5) * width
        color = COLORS.get(dataset_name, 'black')
        ax.bar(x + offset, scores, width, label=dataset_name, color=color, alpha=0.7)
        ax.errorbar(x + offset, scores, yerr=stds, fmt='none', color='black', capsize=2, alpha=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(block_labels)
    ax.set_ylabel('Mean Block Score (P(|1>) on trash)')
    ax.set_title('Final Block Scores: Background vs Signals')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {save_path}")
    
    return fig


def get_full_encoding_at_stage(results, stage_map):
    """
    Helper to concatenate all block outputs into full encoding.
    stage_map: dict mapping block -> stage name
    """
    met = results['met'][stage_map['met']]
    ele = results['ele'][stage_map['ele']]
    mu = results['mu'][stage_map['mu']]
    jet = results['jet'][stage_map['jet']]
    return np.concatenate([met, ele, mu, jet], axis=1)


def plot_pca_final(all_results, output_dir, n_samples_per_class=5000):
    """
    Create PCA visualization of final block scores, comparing background vs signals.
    """
    print("  Creating PCA for final scores...")
    
    stage_map = {'met': 'final', 'ele': 'final', 'mu': 'final', 'jet': 'after_layer_4'}
    
    # Collect encodings from all datasets
    all_encodings = []
    all_labels = []
    
    for dataset_name, results in all_results.items():
        encoding = get_full_encoding_at_stage(results, stage_map)
        # Convert to P(|1>)
        encoding = 0.5 * (1 - encoding)
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
        encoding = get_full_encoding_at_stage(results, stage_map)
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
    plt.title('PCA of Final Block Scores (8D: MET + Ele + Mu + Jet trash)', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, 'pca_final_scores.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"    Saved {save_path}")
    plt.close()


def plot_tsne_final(all_results, output_dir, n_samples_per_class=3000):
    """
    Create t-SNE visualization of final block scores, comparing background vs signals.
    """
    print("  Creating t-SNE for final scores...")
    
    stage_map = {'met': 'final', 'ele': 'final', 'mu': 'final', 'jet': 'after_layer_4'}
    
    # Collect encodings from all datasets
    all_encodings = []
    all_labels = []
    
    for dataset_name, results in all_results.items():
        encoding = get_full_encoding_at_stage(results, stage_map)
        # Convert to P(|1>)
        encoding = 0.5 * (1 - encoding)
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
        encoding = get_full_encoding_at_stage(results, stage_map)
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
    plt.title('t-SNE of Final Block Scores (8D: MET + Ele + Mu + Jet trash)', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, 'tsne_final_scores.png')
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
    parser = argparse.ArgumentParser(description='Inspect QAE model')
    parser.add_argument('--model-path', default='outputs/models/quantum_qae_best.pt',
                       help='Path to QAE model checkpoint')
    parser.add_argument('--max-samples', type=int, default=10000,
                       help='Number of samples to process per dataset')
    parser.add_argument('--output-dir', default='outputs/inspection_qae',
                       help='Directory to save plots')
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
    
    model = ParticleQAEAnomalyModel(depth=model_config.get('quantum_depth', 2))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")
    
    # Print weights
    print_model_weights(model)
    
    # Create layerwise circuits
    print("\nCreating layerwise inspection circuits...")
    circuits = create_layerwise_circuits()
    
    # Load and process all datasets
    all_results = {}
    
    # Background (use samples from evaluation set, starting at 2M)
    print("\n" + "="*60)
    print("Loading and processing Background data...")
    print("="*60)
    bg_data = load_data(BACKGROUND_PATH, start_idx=2_000_000, max_samples=args.max_samples)
    print(f"Computing expectations at each stage for {len(bg_data)} samples...")
    all_results['Background'] = get_layerwise_expectations(model, bg_data, circuits)
    
    # Signals
    for path, label in zip(SIGNAL_PATHS, SIGNAL_LABELS):
        if os.path.exists(path):
            print(f"\n" + "="*60)
            print(f"Loading and processing {label} data...")
            print("="*60)
            sig_data = load_data(path, max_samples=args.max_samples)
            print(f"Computing expectations at each stage for {len(sig_data)} samples...")
            all_results[label] = get_layerwise_expectations(model, sig_data, circuits)
        else:
            print(f"\nWarning: {path} not found, skipping {label}")
    
    # Create visualizations
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)
    
    # 1. Distribution plots for each block
    print("\n1. Creating distribution plots...")
    
    met_labels = ['MET Z']
    fig_met = plot_expectation_distributions(
        all_results, 'met', met_labels, 'MET',
        save_path=os.path.join(args.output_dir, 'met_distributions.png')
    )
    plt.close(fig_met)
    
    ele_labels = ['Trash q2 Z', 'Trash q3 Z']
    fig_ele = plot_expectation_distributions(
        all_results, 'ele', ele_labels, 'Electrons',
        save_path=os.path.join(args.output_dir, 'electron_distributions.png')
    )
    plt.close(fig_ele)
    
    mu_labels = ['Trash q2 Z', 'Trash q3 Z']
    fig_mu = plot_expectation_distributions(
        all_results, 'mu', mu_labels, 'Muons',
        save_path=os.path.join(args.output_dir, 'muon_distributions.png')
    )
    plt.close(fig_mu)
    
    jet_labels = ['Trash q6 Z', 'Trash q7 Z', 'Trash q8 Z', 'Trash q9 Z']
    fig_jet = plot_expectation_distributions(
        all_results, 'jet', jet_labels, 'Jets',
        save_path=os.path.join(args.output_dir, 'jet_distributions.png')
    )
    plt.close(fig_jet)
    
    # 2. Trash excitation evolution
    print("\n2. Creating trash excitation evolution plot...")
    fig_evolution = plot_trash_excitation_evolution(
        all_results,
        save_path=os.path.join(args.output_dir, 'trash_excitation_evolution.png')
    )
    plt.close(fig_evolution)
    
    # 3. Block scores comparison
    print("\n3. Creating block scores comparison...")
    fig_scores = plot_block_scores_comparison(
        all_results,
        save_path=os.path.join(args.output_dir, 'block_scores_comparison.png')
    )
    plt.close(fig_scores)
    
    # 4. PCA
    print("\n4. Creating PCA visualization...")
    plot_pca_final(all_results, args.output_dir)
    
    # 5. t-SNE (optional)
    if not args.skip_tsne:
        print("\n5. Creating t-SNE visualization (this may take a while)...")
        plot_tsne_final(all_results, args.output_dir)
    else:
        print("\n5. Skipping t-SNE visualization (use without --skip-tsne to enable)")
    
    print(f"\n" + "="*60)
    print(f"Inspection complete! Plots saved to {args.output_dir}/")
    print("="*60)
    print("\nGenerated plots:")
    print("  - met_distributions.png")
    print("  - electron_distributions.png")
    print("  - muon_distributions.png")
    print("  - jet_distributions.png")
    print("  - trash_excitation_evolution.png")
    print("  - block_scores_comparison.png")
    print("  - pca_final_scores.png")
    if not args.skip_tsne:
        print("  - tsne_final_scores.png")


if __name__ == "__main__":
    main()
