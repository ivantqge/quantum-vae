#!/usr/bin/env python3
"""
Quantum encoding module for particle physics data.
"""

from typing import Tuple, Optional

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import h5py

#h5 lazy array loading
class LazyH5Array:
    """
    A lazy loading array that behaves like a JAX array but only loads data on demand.
    
    This class provides efficient access to large H5 datasets by loading data in chunks
    and caching them for repeated access.
    """
    
    def __init__(self, h5_file_path: str, dataset_key: str = "Particles", dtype=np.float64,
                 norm: bool = True):
        """
        Initialize the lazy loading array.

        Args:
            h5_file_path: Path to the H5 file
            dataset_key: Key of the dataset in the H5 file
            dtype: Data type for the output array
            norm: Whether to apply physics-aware normalization
            cartesian: Whether data is in Cartesian coordinates (affects normalization)
            skip_MET_eta: Whether MET eta is excluded from features
        """
        self.h5_file_path = h5_file_path
        self.dataset_key = dataset_key
        self.dtype = dtype
        self.norm = norm

        # Open the file to get shape, but don't load data
        with h5py.File(h5_file_path, 'r', swmr=False) as h5_file:
            self.dataset_shape = h5_file[dataset_key].shape
            self.shape = (self.dataset_shape[0], 56)

        # Keep track of loaded chunks
        self._cache = {}

    def __len__(self):
        """Return the number of samples."""
        return self.shape[0]

    def _process_chunk(self, data_chunk):
        """Transform a chunk of the raw data to the desired output format."""
        n_samples = data_chunk.shape[0]
        output = np.zeros((n_samples, 56), dtype=self.dtype)

        # MET features (indices 0-1)
        output[:, 0] = data_chunk[:, 0, 0]  # pt
        output[:, 1] = data_chunk[:, 0, 2]  # phi

        # Electron features (indices 2-13)
        for e in range(4):
            output[:, 2+e] = data_chunk[:, 1+e, 0]    # e pt
            output[:, 6+e] = data_chunk[:, 1+e, 1]    # e eta
            output[:, 10+e] = data_chunk[:, 1+e, 2]   # e phi

        # Muon features (indices 14-25)
        for m in range(4):
            output[:, 14+m] = data_chunk[:, 5+m, 0]   # mu pt
            output[:, 18+m] = data_chunk[:, 5+m, 1]   # mu eta
            output[:, 22+m] = data_chunk[:, 5+m, 2]   # mu phi

        # Jet features (indices 26-55)
        for j in range(10):
            output[:, 26+j] = data_chunk[:, 9+j, 0]   # jet pt
            output[:, 36+j] = data_chunk[:, 9+j, 1]   # jet eta
            output[:, 46+j] = data_chunk[:, 9+j, 2]   # jet phi

        # Apply physics-aware normalization if enabled
        if self.norm:
            output = self._normalize(output)

        return output

    def _normalize(self, features):
        """
        Apply physics-aware normalization to features for angle encoding.
        
        Normalization scheme:
        - pt: log(pt + 1) -> scale to [0, 1] -> map to [0, pi]
        - eta: scale from [-3, 3] to [-pi, pi]
        - phi: already in [-pi, pi], keep as is (just clamp)
        """        
        # Current layout: MET(0-1), e_pt(2-5), e_eta(6-9), e_phi(10-13),
        #                 mu_pt(14-17), mu_eta(18-21), mu_phi(22-25),
        #                 jet_pt(26-35), jet_eta(36-45), jet_phi(46-55)
        
        # Constants
        pt_log_min = 0.0
        pt_log_max = 8.0  # log(~3000) ≈ 8
        eta_max = 3.0
        
        def normalize_pt(pt):
            """pt: log(pt+1) -> [0,1] -> [0, pi]"""
            log_pt = np.log(np.maximum(pt, 1e-6) + 1.0)
            scaled = (log_pt - pt_log_min) / (pt_log_max - pt_log_min)
            scaled = np.clip(scaled, 0.0, 1.0)
            return scaled * np.pi
        
        def normalize_eta(eta):
            """eta: [-eta_max, eta_max] -> [-pi, pi]"""
            scaled = eta / eta_max
            scaled = np.clip(scaled, -1.0, 1.0)
            return scaled * np.pi
        
        def normalize_phi(phi):
            """phi: already in [-pi, pi], just clamp"""
            return np.clip(phi, -np.pi, np.pi)
        
        # MET normalization (indices 0-1: pt, phi)
        features[:, 0] = normalize_pt(features[:, 0])  # MET pt
        features[:, 1] = normalize_phi(features[:, 1])  # MET phi
        
        # Electron normalization
        features[:, 2:6] = normalize_pt(features[:, 2:6])  # pt
        features[:, 6:10] = normalize_eta(features[:, 6:10])  # eta
        features[:, 10:14] = normalize_phi(features[:, 10:14])  # phi
        
        # Muon normalization
        features[:, 14:18] = normalize_pt(features[:, 14:18])  # pt
        features[:, 18:22] = normalize_eta(features[:, 18:22])  # eta
        features[:, 22:26] = normalize_phi(features[:, 22:26])  # phi
        
        # Jet normalization
        features[:, 26:36] = normalize_pt(features[:, 26:36])  # pt
        features[:, 36:46] = normalize_eta(features[:, 36:46])  # eta
        features[:, 46:56] = normalize_phi(features[:, 46:56])  # phi
        
        return features

    def _get_chunk(self, start_idx, end_idx):
        """Load a chunk from the H5 file if not in cache."""
        chunk_key = (start_idx, end_idx)
        if chunk_key not in self._cache:
            with h5py.File(self.h5_file_path, 'r', swmr=False) as h5_file:
                raw_chunk = h5_file[self.dataset_key][start_idx:end_idx]
                self._cache[chunk_key] = self._process_chunk(raw_chunk)
        return self._cache[chunk_key]

    def __getitem__(self, idx):
        """Support array-like indexing but load data on demand."""
        if isinstance(idx, int):
            # Single item access
            chunk = self._get_chunk(idx, idx+1)
            return np.asarray(chunk[0], dtype=self.dtype)
        elif isinstance(idx, slice):
            # Slice access
            start = idx.start or 0
            stop = idx.stop or len(self)
            step = idx.step or 1

            if step != 1:
                # For non-unit steps, we need to load separate chunks
                indices = range(start, stop, step)
                result = np.zeros((len(indices), 56), dtype=self.dtype)
                for i, idx in enumerate(indices):
                    result[i] = self._get_chunk(idx, idx+1)[0]
                return np.asarray(result, dtype=self.dtype)
            else:
                # For unit steps, we can load the whole range at once
                return np.asarray(self._get_chunk(start, stop), dtype=self.dtype)
        elif isinstance(idx, tuple) and len(idx) == 2:
            # 2D indexing
            if isinstance(idx[0], int) and isinstance(idx[1], int):
                chunk = self._get_chunk(idx[0], idx[0]+1)
                return np.asarray(chunk[0, idx[1]], dtype=self.dtype)
            else:
                # Handle more complex slicing
                # (simplified implementation - would need to handle all cases)
                row_slice = idx[0]
                col_slice = idx[1]
                if isinstance(row_slice, slice):
                    start = row_slice.start or 0
                    stop = row_slice.stop or len(self)
                    chunk = self._get_chunk(start, stop)
                    return np.asarray(chunk[:, col_slice], dtype=self.dtype)
                else:
                    raise NotImplementedError("Complex indexing not fully implemented")
        else:
            raise IndexError("Unsupported indexing")