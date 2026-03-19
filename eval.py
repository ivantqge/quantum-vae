#!/usr/bin/env python3
"""
Evaluation script for Quantum VAE model - PyTorch implementation.
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import torch
import sklearn.metrics as sk
import matplotlib.pyplot as plt

import h5py
from quantum_encoding import LazyH5Array
from full_quantum_vae import create_particle_quantum_vae
from hybrid_quantum_vae import create_hybrid_vae
from extended_quantum_vae import create_extended_quantum_vae
from extended_quantum_vae_qmi import create_qmi_extended_quantum_vae
from block_quantum_ae import ParticleQAEAnomalyModel #, compute_presence_mask
from block_quantum_ae_qmi import QMIParticleQAEAnomalyModel

# Signal labels
signal_labels = ["Ato4l"]
                # "hToTauTau",
                # "hChToTauNu",
                # "leptoquark"]


def AD_score_KL(z_mean, z_log_var):
    kl_loss = np.mean(-0.5 * (1 + z_log_var - (z_mean) ** 2 - np.exp(z_log_var)))
    return kl_loss

def AD_score_CKL(z_mean, z_log_var):
    CKL = np.mean(z_mean**2)
    return CKL


class Model_Evaluator:
    def __init__(
        self,
        model_path,
        background,
        br_weights,
        signal,
        signal_weights,
        input_dim,
        title="placeholder",
        save=False,
        labels=None,
        device=None,
        model_type='vae',
    ):
        self.input_dim = input_dim
        
        # Set up device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model_config = checkpoint.get('model_config', {})
        
        # Auto-detect model type from checkpoint if available
        model_type = checkpoint.get('model_type', model_type)
        print(f"Loading model type: {model_type}")
        
        # Create model based on type
        if model_type == 'hybrid':
            self.model = create_hybrid_vae(
                input_dim=model_config.get('input_dim', input_dim),
                hidden_dim=model_config.get('hidden_dim', 16),
                latent_dim=model_config.get('latent_dim', 3),
                steps_per_epoch=model_config.get('steps_per_epoch', 3125),
                cycle_length=model_config.get('cycle_length', 10),
                min_beta=model_config.get('min_beta', 0.1),
                max_beta=model_config.get('max_beta', 0.8),
            )
        elif model_type == 'extended':
            self.model = create_extended_quantum_vae(
                input_dim=model_config.get('input_dim', input_dim),
                hidden_dim=model_config.get('hidden_dim', 16),
                latent_dim=model_config.get('latent_dim', 3),
                quantum_depth=model_config.get('quantum_depth', 2),
                steps_per_epoch=model_config.get('steps_per_epoch', 3125),
                cycle_length=model_config.get('cycle_length', 10),
                min_beta=model_config.get('min_beta', 0.1),
                max_beta=model_config.get('max_beta', 0.8),
                device=model_config.get('device', 'default.qubit')
            )
        elif model_type == 'extended_qmi':
            self.model = create_qmi_extended_quantum_vae(
                input_dim=model_config.get('input_dim', input_dim),
                hidden_dim=model_config.get('hidden_dim', 16),
                latent_dim=model_config.get('latent_dim', 3),
                quantum_depth=model_config.get('quantum_depth', 2),
                steps_per_epoch=model_config.get('steps_per_epoch', 3125),
                cycle_length=model_config.get('cycle_length', 10),
                min_beta=model_config.get('min_beta', 0.1),
                max_beta=model_config.get('max_beta', 0.8),
                device=model_config.get('device', 'default.qubit')
            )
        else:
            self.model = create_particle_quantum_vae(
                input_dim=model_config.get('input_dim', input_dim),
                hidden_dim=model_config.get('hidden_dim', 16),
                latent_dim=model_config.get('latent_dim', 3),
                quantum_depth=model_config.get('quantum_depth', 2),
                steps_per_epoch=model_config.get('steps_per_epoch', 3125),
                cycle_length=model_config.get('cycle_length', 10),
                min_beta=model_config.get('min_beta', 0.1),
                max_beta=model_config.get('max_beta', 0.8),
                device=model_config.get('device', 'default.qubit')
            )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.encoder = self.model.encoder
        self.signal = signal
        self.background = background
        self.br_loss = []
        self.signal_loss = []
        self.background_outputs = []
        self.signal_outputs = []
        self.title = title
        self.saveplots = save
        self.labels = labels
        self.latent_info = []
        self.br_weights = br_weights
        self.signal_weights = signal_weights

    def _encode(self, data, batch_size=1024):
        """Encode data using the encoder and return z_mean, z_log_var as numpy arrays.
        
        Processes data in mini-batches to avoid GPU memory issues.
        """
        n_samples = data.shape[0]
        z_means = []
        z_log_vars = []
        
        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch = data[start_idx:end_idx]
                
                if isinstance(batch, np.ndarray):
                    batch_tensor = torch.tensor(batch, dtype=torch.float64, device=self.device)
                else:
                    batch_tensor = batch.to(self.device)
                
                z_mean, z_log_var, _ = self.encoder(batch_tensor)
                z_means.append(z_mean.cpu().numpy())
                z_log_vars.append(z_log_var.cpu().numpy())
        
        return np.concatenate(z_means, axis=0), np.concatenate(z_log_vars, axis=0)

    def calculate_loss(self, l_type):
        self.signal_loss = []
        self.br_loss = []
        br = self.background

        if l_type == "CKL":
            br_latent = self._encode(br)
            self.latent_info += [br_latent[0]]
            l = []
            for i in range(0, br.shape[0]):
                loss = AD_score_CKL(br_latent[0][i], br_latent[1][i])
                l += [loss]
            self.br_loss = l

            for i, batch in enumerate(self.signal):
                sg_latent = self._encode(batch)
                self.latent_info += [sg_latent[0]]
                l = []

                for i in range(0, batch.shape[0]):
                    loss = AD_score_CKL(sg_latent[0][i], sg_latent[1][i])
                    l += [loss]

                sg_loss = l

                self.signal_loss += [sg_loss]

        if l_type == "KL":
            br_latent = self._encode(br)
            l = []
            for i in range(0, br.shape[0]):
                loss = AD_score_KL(br_latent[0][i], br_latent[1][i])
                l += [loss]
            self.br_loss = l

            for i, batch in enumerate(self.signal):
                sg_latent = self._encode(batch)

                l = []

                for i in range(0, batch.shape[0]):
                    loss = AD_score_KL(sg_latent[0][i], sg_latent[1][i])
                    l += [loss]

                sg_loss = l

                self.signal_loss += [sg_loss]        

    def ROC(self):
        target_fpr = 1e-5
        tpr_at_target = []
        thresholds_at_target = []

        plt.plot(
            np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), "--", label="diagonal"
        )
        for j, batch in enumerate(self.signal_loss):
            sig_w = self.signal_weights[j]
            br_w = self.br_weights
            weights = np.concatenate((br_w, sig_w))
            truth = []
            for i in range(len(self.br_loss)):
                truth += [0]
            for i in range(len(batch)):
                truth += [1]
            ROC_data = np.concatenate((self.br_loss, batch))
            fpr, tpr, thresholds = sk.roc_curve(truth, ROC_data, sample_weight=weights)

            auc = sk.roc_auc_score(truth, ROC_data)
            
            idx = np.argmin(np.abs(fpr - target_fpr))
            tpr_at_fpr = tpr[idx]
            tpr_at_target.append(tpr_at_fpr)
            thresholds_at_target.append(thresholds[idx])
            
            plt.plot(fpr, tpr, label=f"{self.labels[j]}: AUC={auc:.3f}, TPR={tpr_at_fpr*100:.2f}%")


        plt.xlabel("fpr")
        plt.xlim(1e-7, 1)
        plt.ylim(1e-7, 1)
        plt.semilogx()
        plt.ylabel("tpr")
        plt.semilogy()
        plt.title("{}_ROC".format(self.title))
        plt.vlines(10**-5, 0, 1, colors="r", linestyles="dashed")
        plt.legend(loc="lower right")
        if self.saveplots == True:
            os.makedirs("outputs/plots", exist_ok=True)
            plt.savefig(
                "outputs/plots/{}_ROC.png".format(
                    self.title
                ),
                format="png",
                bbox_inches="tight",
            )
        plt.show()
        
        print(f"\nTPR at FPR = {target_fpr} for each channel:")
        for label, tpr, threshold in zip(self.labels, tpr_at_target, thresholds_at_target):
            print(f"{label}: {tpr*100:.6f}%, Theshold = {threshold:.6f}")

    def GetPerformance(self):
        target_fpr = 1e-5
        tpr_at_target = []

        print(f"Number of signal losses: {len(self.signal_loss)}")
        print(f"Number of labels: {len(self.labels)}")
        print(f"Number of signal weights: {len(self.signal_weights)}")
        print(f"Length of br_loss: {len(self.br_loss)}")

        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), "--", label="diagonal")

        for j, batch in enumerate(self.signal_loss):
            print(f"Processing batch {j}")
            print(f"Batch length: {len(batch)}")
            print(f"Signal weight length: {len(self.signal_weights[j])}")

            sig_w = self.signal_weights[j]
            br_w = self.br_weights
            weights = np.concatenate((br_w, sig_w))
            truth = np.concatenate([np.zeros(len(self.br_loss)), np.ones(len(batch))])
            ROC_data = np.concatenate((self.br_loss, batch))

            print(f"ROC_data shape: {ROC_data.shape}")
            print(f"truth shape: {truth.shape}")
            print(f"weights shape: {weights.shape}")


            fpr, tpr, _ = sk.roc_curve(truth, ROC_data, sample_weight=weights)
            auc = sk.roc_auc_score(truth, ROC_data)

            idx = np.argmin(np.abs(fpr - target_fpr))
            tpr_at_fpr = tpr[idx]
            tpr_at_target.append(tpr_at_fpr)

            plt.plot(fpr, tpr, label=f"{self.labels[j]}: AUC={auc:.3f}, TPR={tpr_at_fpr*100:.2f}%")

            print(f"Successfully processed batch {j}")


        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f"{self.title} ROC")
        plt.vlines(target_fpr, 0, 1, colors="r", linestyles="dashed")
        plt.legend(loc="lower right")
        if self.saveplots:
            os.makedirs("outputs/plots", exist_ok=True)
            plt.savefig(
                f"outputs/plots/{self.title}_ROC.png",
                format="png",
                bbox_inches="tight",
            )
        plt.show()

        print(f"\nTPR at FPR = {target_fpr} for each channel:")
        results = list(zip(self.labels, tpr_at_target))
        for label, tpr in results:
            print(f"{label}: {tpr*100:.6f}%")

        print(f"Number of results: {len(results)}")
        return results

    def plot_anomaly_scores(self):
        """Plot anomaly score distributions for background and each signal event."""
        if not self.br_loss or not self.signal_loss:
            print("Error: Must call calculate_loss() before plotting anomaly scores")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot background as histogram
        plt.hist(self.br_loss, bins=100, alpha=0.6, label='Background', 
                color='gray', density=True, histtype='step', linewidth=2)
        
        # Plot each signal overlaid
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for j, (batch, label) in enumerate(zip(self.signal_loss, self.labels)):
            color = colors[j % len(colors)]
            plt.hist(batch, bins=100, alpha=0.5, label=label, 
                    color=color, density=True, histtype='step', linewidth=2)
        
        plt.xlabel('Anomaly Score', fontsize=12)
        plt.ylabel('Normalized Density', fontsize=12)
        plt.title(f'{self.title} - Anomaly Score Distributions', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if self.saveplots:
            os.makedirs("outputs/plots", exist_ok=True)
            plt.savefig(
                f"outputs/plots/{self.title}_anomaly_scores.png",
                format="png",
                bbox_inches="tight",
            )
            print(f"Saved anomaly score plot to outputs/plots/{self.title}_anomaly_scores.png")
        plt.show()


class QAE_Evaluator:
    """Evaluator for QAE model using Mahalanobis anomaly scores."""
    
    def __init__(
        self,
        model_path,
        background,
        br_weights,
        signal,
        signal_weights,
        title="QAE Model",
        save=False,
        labels=None,
        device=None,
        background_raw=None,
        signal_raw=None,
        use_presence_mask=True,
    ):
        # Set up device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model_config = checkpoint.get('model_config', {})
        
        # Detect QAE variant from checkpoint
        qae_variant = checkpoint.get('qae_variant', model_config.get('qae_variant', '4block'))
        self.qae_variant = qae_variant
        
        # Create QAE model based on variant
        if qae_variant == '2circuit':
            print(f"Loading 2-circuit QAE model...")
            self.model = TwoCircuitQAEAnomalyModel(depth=model_config.get('quantum_depth', 2))
        elif qae_variant == 'qmi':
            print(f"Loading QMI QAE model (4/4/4/7 grouping)...")
            self.model = QMIParticleQAEAnomalyModel(depth=model_config.get('quantum_depth', 2))
        else:
            trash_dim = model_config.get('trash_dim', 4)
            print(f"Loading QAE model with {trash_dim}D trash vectors...")
            if trash_dim == 9:
                self.model = ParticleQAEAnomalyModel9D(depth=model_config.get('quantum_depth', 2))
            else:
                self.model = ParticleQAEAnomalyModel(depth=model_config.get('quantum_depth', 2))
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load fitted background stats if available
        # Load fitted background stats if available
        if 'bg_mu' in checkpoint and 'bg_precision' in checkpoint:
            self.model.mu.copy_(checkpoint['bg_mu'].to(self.device))
            self.model.precision.copy_(checkpoint['bg_precision'].to(self.device))
            self.model._stats_fitted = True
            print("Loaded fitted background statistics from checkpoint")
        else:
            print("No background stats in checkpoint - will fit on eval background data")
            self._need_fit = True
        
        self.signal = signal
        self.background = background
        self.br_loss = []
        self.signal_loss = []
        self.title = title
        self.saveplots = save
        self.labels = labels
        self.br_weights = br_weights
        self.signal_weights = signal_weights
        
        # Presence mask support
        self.use_presence_mask = use_presence_mask
        self.background_raw = background_raw
        self.signal_raw = signal_raw
        
        if use_presence_mask and background_raw is None:
            print("Warning: use_presence_mask=True but no raw data provided. Disabling presence mask.")
            self.use_presence_mask = False

    def _get_anomaly_scores(self, data, raw_data=None, batch_size=1024):
        """Get Mahalanobis anomaly scores for data in batches.
        
        Args:
            data: Normalized data
            raw_data: Raw (unnormalized) data for presence mask computation
            batch_size: Batch size for processing
        """
        n_samples = data.shape[0]
        scores = []
        
        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch = data[start_idx:end_idx]
                
                if isinstance(batch, np.ndarray):
                    batch_tensor = torch.tensor(batch, dtype=torch.float64, device=self.device)
                else:
                    batch_tensor = batch.to(self.device)
                
                # Compute presence mask if enabled (not for 2circuit or qmi)
                if self.qae_variant in ('2circuit', 'qmi'):
                    batch_scores = self.model.anomaly_score(batch_tensor)
                elif self.use_presence_mask and raw_data is not None:
                    raw_batch = raw_data[start_idx:end_idx]
                    if isinstance(raw_batch, np.ndarray):
                        raw_batch_tensor = torch.tensor(raw_batch, dtype=torch.float64, device=self.device)
                    else:
                        raw_batch_tensor = raw_batch.to(self.device)
                    presence_mask = compute_presence_mask(raw_batch_tensor)
                    batch_scores = self.model.anomaly_score(batch_tensor, presence_mask=presence_mask)
                else:
                    batch_scores = self.model.anomaly_score(batch_tensor)
                scores.append(batch_scores.cpu().numpy())
        
        return np.concatenate(scores, axis=0)

    def _get_block_scores(self, data, raw_data=None, batch_size=1024):
        """Get block scores (before Mahalanobis) for fitting background stats.
        
        Args:
            data: Normalized data
            raw_data: Raw (unnormalized) data for presence mask computation
            batch_size: Batch size for processing
        """
        n_samples = data.shape[0]
        scores = []
        
        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch = data[start_idx:end_idx]
                
                if isinstance(batch, np.ndarray):
                    batch_tensor = torch.tensor(batch, dtype=torch.float64, device=self.device)
                else:
                    batch_tensor = batch.to(self.device)
                
                # Compute presence mask if enabled (not for 2circuit or qmi)
                if self.qae_variant in ('2circuit', 'qmi'):
                    batch_scores = self.model.block_scores(batch_tensor)
                elif self.use_presence_mask and raw_data is not None:
                    raw_batch = raw_data[start_idx:end_idx]
                    if isinstance(raw_batch, np.ndarray):
                        raw_batch_tensor = torch.tensor(raw_batch, dtype=torch.float64, device=self.device)
                    else:
                        raw_batch_tensor = raw_batch.to(self.device)
                    presence_mask = compute_presence_mask(raw_batch_tensor)
                    batch_scores = self.model.block_scores(batch_tensor, presence_mask=presence_mask)
                else:
                    batch_scores = self.model.block_scores(batch_tensor)
                scores.append(batch_scores.cpu())
        
        return torch.cat(scores, dim=0)

    def calculate_loss(self):
        """Calculate Mahalanobis anomaly scores for background and signal."""
        # Fit background stats if needed
        if getattr(self, '_need_fit', False):
            print("Fitting background statistics on evaluation data...")
            bg_scores = self._get_block_scores(self.background, raw_data=self.background_raw)
            mu, precision = self.model.fit_background_stats(bg_scores, use_full_cov=True, ridge=1e-3)
            print(f"  Fitted mu: {mu.numpy()}")
            print(f"  Stats fitted on {len(bg_scores)} background samples")
        
        print("Calculating anomaly scores for background...")
        self.br_loss = self._get_anomaly_scores(self.background, raw_data=self.background_raw).tolist()
        print(f"  Background scores: min={min(self.br_loss):.4f}, max={max(self.br_loss):.4f}, mean={np.mean(self.br_loss):.4f}")
        
        self.signal_loss = []
        for i, sig_data in enumerate(self.signal):
            print(f"Calculating anomaly scores for signal {i+1}/{len(self.signal)}...")
            sig_raw = self.signal_raw[i] if self.signal_raw else None
            sig_scores = self._get_anomaly_scores(sig_data, raw_data=sig_raw).tolist()
            self.signal_loss.append(sig_scores)
            print(f"  Signal scores: min={min(sig_scores):.4f}, max={max(sig_scores):.4f}, mean={np.mean(sig_scores):.4f}")

    def ROC(self):
        """Plot ROC curves."""
        target_fpr = 1e-5
        tpr_at_target = []
        thresholds_at_target = []

        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), "--", label="diagonal")
        
        for j, batch in enumerate(self.signal_loss):
            sig_w = self.signal_weights[j]
            br_w = self.br_weights
            weights = np.concatenate((br_w, sig_w))
            truth = np.concatenate([np.zeros(len(self.br_loss)), np.ones(len(batch))])
            ROC_data = np.concatenate((self.br_loss, batch))
            
            fpr, tpr, thresholds = sk.roc_curve(truth, ROC_data, sample_weight=weights)
            auc = sk.roc_auc_score(truth, ROC_data)
            
            idx = np.argmin(np.abs(fpr - target_fpr))
            tpr_at_fpr = tpr[idx]
            tpr_at_target.append(tpr_at_fpr)
            thresholds_at_target.append(thresholds[idx])
            
            plt.plot(fpr, tpr, label=f"{self.labels[j]}: AUC={auc:.3f}, TPR={tpr_at_fpr*100:.2f}%")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1e-7, 1)
        plt.ylim(1e-7, 1)
        plt.title(f"{self.title} ROC")
        plt.vlines(target_fpr, 0, 1, colors="r", linestyles="dashed")
        plt.legend(loc="lower right")
        
        if self.saveplots:
            os.makedirs("outputs/plots", exist_ok=True)
            plt.savefig(f"outputs/plots/{self.title}_ROC.png", format="png", bbox_inches="tight")
            print(f"Saved ROC plot to outputs/plots/{self.title}_ROC.png")
        plt.show()
        
        print(f"\nTPR at FPR = {target_fpr} for each channel:")
        for label, tpr, threshold in zip(self.labels, tpr_at_target, thresholds_at_target):
            print(f"{label}: {tpr*100:.6f}%, Threshold = {threshold:.6f}")

    def plot_anomaly_scores(self):
        """Plot anomaly score distributions for background and each signal event."""
        if not self.br_loss or not self.signal_loss:
            print("Error: Must call calculate_loss() before plotting anomaly scores")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot background as histogram
        plt.hist(self.br_loss, bins=100, alpha=0.6, label='Background', 
                color='gray', density=True, histtype='step', linewidth=2)
        
        # Plot each signal overlaid
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for j, (batch, label) in enumerate(zip(self.signal_loss, self.labels)):
            color = colors[j % len(colors)]
            plt.hist(batch, bins=100, alpha=0.5, label=label, 
                    color=color, density=True, histtype='step', linewidth=2)
        
        plt.xlabel('Anomaly Score', fontsize=12)
        plt.ylabel('Normalized Density', fontsize=12)
        plt.title(f'{self.title} - Anomaly Score Distributions', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if self.saveplots:
            os.makedirs("outputs/plots", exist_ok=True)
            plt.savefig(
                f"outputs/plots/{self.title}_anomaly_scores.png",
                format="png",
                bbox_inches="tight",
            )
            print(f"Saved anomaly score plot to outputs/plots/{self.title}_anomaly_scores.png")
        plt.show()


def load_data(h5_path, start_idx=0, max_samples=None, return_raw=False):
    """Load data from H5 file using LazyH5Array.
    
    Args:
        h5_path: Path to H5 file
        start_idx: Starting index in the dataset
        max_samples: Maximum number of samples to load (None = all from start_idx)
        return_raw: If True, also return raw (unnormalized) data for presence mask
    
    Returns:
        If return_raw=False: normalized data array
        If return_raw=True: (normalized_data, raw_data) tuple
    """
    # Load normalized data
    data_loader = LazyH5Array(h5_path, norm=True)
    total_samples = len(data_loader)
    
    if max_samples is not None:
        end_idx = min(start_idx + max_samples, total_samples)
    else:
        end_idx = total_samples
    
    print(f"  Loading samples {start_idx:,} to {end_idx:,} ({end_idx - start_idx:,} samples)")
    normalized_data = np.array(data_loader[start_idx:end_idx], dtype=np.float64)
    
    if return_raw:
        # Also load raw data
        raw_loader = LazyH5Array(h5_path, norm=False)
        raw_data = np.array(raw_loader[start_idx:end_idx], dtype=np.float64)
        return normalized_data, raw_data
    
    return normalized_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Quantum VAE/QAE model')
    parser.add_argument('--model-type', choices=['vae', 'hybrid', 'extended', 'extended_qmi', 'qae'], default='vae',
                       help='Model type to evaluate: vae (quantum 56->19), hybrid (classical 56->19), extended (quantum 56->32), extended_qmi (QMI ordering), qae (trash qubits)')
    parser.add_argument('--model-path', default=None,
                       help='Path to model checkpoint (auto-detected if not set)')
    parser.add_argument('--title', default=None,
                       help='Title for plots (auto-generated if not set)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max background samples to evaluate (default: all 2M)')
    parser.add_argument('--no-presence-mask', action='store_true',
                       help='Disable presence masking for QAE evaluation')
    args = parser.parse_args()
    
    # Set torch default dtype
    torch.set_default_dtype(torch.float64)
    
    # Auto-detect model path if not provided
    if args.model_path is None:
        if args.model_type == 'vae':
            model_path = 'outputs/models/particle_quantum_vae_best.pt'
        elif args.model_type == 'hybrid':
            model_path = 'outputs/models/particle_hybrid_vae_best.pt'
        elif args.model_type == 'extended':
            model_path = 'outputs/models/particle_extended_quantum_vae_best.pt'
        elif args.model_type == 'extended_qmi':
            model_path = 'outputs/models/particle_quantum_extended_qmi_best.pt'
        else:
            # QAE - Prefer _final.pt (has background stats) over _best.pt
            final_path = 'outputs/models/quantum_qae_duffy_mixed14_final.pt'
            best_path = 'outputs/models/quantum_qae_duffy_mixed14_best.pt'
            if os.path.exists(final_path):
                model_path = final_path
                print(f"Using final checkpoint: {model_path}")
            else:
                model_path = best_path
                print(f"Using best checkpoint: {model_path}")
    else:
        model_path = args.model_path
    
    # Auto-generate title if not provided
    if args.title is None:
        title_map = {
            'vae': 'Quantum_VAE',
            'hybrid': 'Hybrid_VAE', 
            'extended': 'Extended_Quantum_VAE',
            'extended_qmi': 'Extended_Quantum_VAE_QMI',
            'qae': 'QAE_Model'
        }
        title = title_map.get(args.model_type, 'Model')
    else:
        title = args.title
    
    print(f"Evaluating {args.model_type.upper()} model from: {model_path}")
    
    background_path = '/global/cfs/cdirs/m2616/sagar/QiML/background_for_training.h5'
    
    signal_paths = [
        '/global/cfs/cdirs/m2616/sagar/QiML/Ato4l_lepFilter_13TeV.h5',
        # '/global/cfs/cdirs/m2616/sagar/QiML/leptoquark_LOWMASS_lepFilter_13TeV.h5',
        # '/global/cfs/cdirs/m2616/sagar/QiML/hChToTauNu_13TeV_PU20.h5',
        # '/global/cfs/cdirs/m2616/sagar/QiML/hToTauTau_13TeV_PU20.h5'
    ]
    
    # Evaluation config: use last 2M samples (samples 2M-4M)
    EVAL_START_IDX = 2_000_000
    EVAL_MAX_SAMPLES = args.max_samples if args.max_samples else 2_000_000
    
    # Determine if we need raw data for presence mask (QAE only)
    use_presence_mask = (args.model_type == 'qae') and (not args.no_presence_mask)
    
    # Load background data (last 2M samples for evaluation)
    print("Loading background data...")
    if use_presence_mask:
        X_test, X_test_raw = load_data(background_path, start_idx=EVAL_START_IDX, 
                                        max_samples=EVAL_MAX_SAMPLES, return_raw=True)
        print(f"  Also loaded raw data for presence mask")
    else:
        X_test = load_data(background_path, start_idx=EVAL_START_IDX, max_samples=EVAL_MAX_SAMPLES)
        X_test_raw = None
    print(f"Background shape: {X_test.shape}")
    
    # Load signal data
    print("Loading signal data...")
    signal_data = []
    signal_data_raw = [] if use_presence_mask else None
    for path in signal_paths:
        if os.path.exists(path):
            if use_presence_mask:
                data, data_raw = load_data(path, return_raw=True)
                signal_data.append(data)
                signal_data_raw.append(data_raw)
            else:
                data = load_data(path)
                signal_data.append(data)
            print(f"  Loaded {path}: {data.shape}")
        else:
            print(f"  Warning: {path} not found")
    
    # Run evaluation based on model type
    if args.model_type in ['vae', 'hybrid', 'extended', 'extended_qmi']:
        # VAE-style models use latent space anomaly scoring
        evaluation = Model_Evaluator(
            model_path,
            X_test,
            np.ones(len(X_test)),
            signal_data,
            [np.ones(len(data)) for data in signal_data],
            input_dim=X_test.shape[1],
            title=title,
            save=True,
            labels=signal_labels[:len(signal_data)],
            model_type=args.model_type
        )
        evaluation.calculate_loss('CKL')
        evaluation.ROC()
        evaluation.plot_anomaly_scores()
    else:
        # QAE evaluation
        evaluation = QAE_Evaluator(
            model_path,
            X_test,
            np.ones(len(X_test)),
            signal_data,
            [np.ones(len(data)) for data in signal_data],
            title=title,
            save=True,
            labels=signal_labels[:len(signal_data)],
            background_raw=X_test_raw,
            signal_raw=signal_data_raw,
            use_presence_mask=use_presence_mask
        )
        evaluation.calculate_loss()
        evaluation.ROC()
        evaluation.plot_anomaly_scores()
