#!/usr/bin/env python3
"""
Quantum encoding module for particle physics data.
"""

from typing import Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as sk

import tensorflow as tf
import tensorflow.math as tfmath
import tensorflow.keras as keras
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    PReLU, Input, LSTM, Flatten, Concatenate, Dense, Conv2D, 
    TimeDistributed, MaxPooling2D, LeakyReLU, ReLU, Dropout, 
    BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import Precision
from tensorflow.keras.regularizers import l1, l2, l1_l2

import pennylane as qml


class LazyH5Array:
    """
    A lazy loading array that behaves like a JAX array but only loads data on demand.
    
    This class provides efficient access to large H5 datasets by loading data in chunks
    and caching them for repeated access.
    """
    
    def __init__(self, h5_file_path: str, dataset_key: str = "Particles", dtype=np.float64):
        """
        Initialize the lazy loading array.

        Args:
            h5_file_path: Path to the H5 file
            dataset_key: Key of the dataset in the H5 file
            dtype: Data type for the output array
        """
        self.h5_file_path = h5_file_path
        self.dataset_key = dataset_key
        self.dtype = dtype

        # Open the file to get shape, but don't load data
        with h5py.File(h5_file_path, 'r') as h5_file:
            self.dataset_shape = h5_file[dataset_key].shape
            # Cache the output shape for quick access
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

        return output

    def _get_chunk(self, start_idx, end_idx):
        """Load a chunk from the H5 file if not in cache."""
        chunk_key = (start_idx, end_idx)
        if chunk_key not in self._cache:
            with h5py.File(self.h5_file_path, 'r') as h5_file:
                raw_chunk = h5_file[self.dataset_key][start_idx:end_idx]
                self._cache[chunk_key] = self._process_chunk(raw_chunk)
        return self._cache[chunk_key]

    def __getitem__(self, idx):
        """Support array-like indexing but load data on demand."""
        if isinstance(idx, int):
            # Single item access
            chunk = self._get_chunk(idx, idx+1)
            return jnp.asarray(chunk[0])
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
                return jnp.asarray(result)
            else:
                # For unit steps, we can load the whole range at once
                return jnp.asarray(self._get_chunk(start, stop))
        elif isinstance(idx, tuple) and len(idx) == 2:
            # 2D indexing
            if isinstance(idx[0], int) and isinstance(idx[1], int):
                chunk = self._get_chunk(idx[0], idx[0]+1)
                return jnp.asarray(chunk[0, idx[1]])
            else:
                # Handle more complex slicing
                # (simplified implementation - would need to handle all cases)
                row_slice = idx[0]
                col_slice = idx[1]
                if isinstance(row_slice, slice):
                    start = row_slice.start or 0
                    stop = row_slice.stop or len(self)
                    chunk = self._get_chunk(start, stop)
                    return jnp.asarray(chunk[:, col_slice])
                else:
                    raise NotImplementedError("Complex indexing not fully implemented")
        else:
            raise IndexError("Unsupported indexing")


# Feature indexing constants
IDX = {
    "met":  (0, 2),        # [0:2] -> (pt, phi)
    "e_pt": slice(2, 6),   # 4
    "e_eta": slice(6, 10), # 4
    "e_phi": slice(10, 14),# 4
    "m_pt": slice(14, 18), # 4
    "m_eta": slice(18, 22),# 4
    "m_phi": slice(22, 26),# 4
    "j_pt": slice(26, 36), # 10
    "j_eta": slice(36, 46),# 10
    "j_phi": slice(46, 56) # 10
}


def met_slice(x):
    """Extract MET features from particle data."""
    a, b = IDX["met"]  # (pt, phi)
    return x[a:b]


def electrons_triplets(x):
    """Extract electron features from particle data."""
    return x[IDX["e_pt"]], x[IDX["e_eta"]], x[IDX["e_phi"]]  # (4,),(4,),(4,)


def muons_triplets(x):
    """Extract muon features from particle data."""
    return x[IDX["m_pt"]], x[IDX["m_eta"]], x[IDX["m_phi"]]


def jets_triplets(x):
    """Extract jet features from particle data."""
    return x[IDX["j_pt"]], x[IDX["j_eta"]], x[IDX["j_phi"]]


def build_scales(
    *,
    met_pt=1.0, met_phi=1.0,
    e_pt=1.0, e_eta=np.pi/10, e_phi=1.0,
    m_pt=1.0, m_eta=np.pi/10, m_phi=1.0,
    j_pt=1.0, j_eta=np.pi/10, j_phi=1.0
):
    """Build scaling parameters for quantum encoding."""
    return {
        "met":  {"pt": met_pt, "phi": met_phi},
        "ele":  {"pt": e_pt,   "eta": e_eta, "phi": e_phi},
        "mu":   {"pt": m_pt,   "eta": m_eta, "phi": m_phi},
        "jet":  {"pt": j_pt,   "eta": j_eta, "phi": j_phi},
    }


def make_xyz_group_qnode(n_qubits, device_name="lightning.gpu"):
    """
    Create a quantum circuit for encoding particle groups.
    
    Input: pt_vec, eta_vec, phi_vec (each length n_qubits), plus scalar scales s_pt,s_eta,s_phi.
    Encoding per qubit k: RX(s_eta*eta_k) -> RY(s_pt*pt_k) -> RZ(s_phi*phi_k)
    Entanglement: linear CNOT chain within the group only.
    Output: [X0,Y0,Z0, X1,Y1,Z1, ...]
    """
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="auto")
    def qnode(pt_vec, eta_vec, phi_vec, s_pt=1.0, s_eta=1.0, s_phi=1.0):
        for k in range(n_qubits):
            qml.RX(s_eta * eta_vec[k], wires=k)
            qml.RY(s_pt  * pt_vec[k],  wires=k)
            qml.RZ(s_phi * phi_vec[k], wires=k)
        for k in range(n_qubits - 1):
            qml.CNOT(wires=[k, k+1])
        obs = []
        for k in range(n_qubits):
            obs += [qml.expval(qml.PauliX(k)),
                    qml.expval(qml.PauliY(k)),
                    qml.expval(qml.PauliZ(k))]
        return obs

    return qnode


def make_met_1q(device_name="lightning.gpu"):
    """
    Create a quantum circuit for MET encoding.
    
    Input: pt, phi (scalars). Encoding: RY(s_pt*pt) -> RZ(s_phi*phi). No entanglement (1 qubit).
    Output: [Z, Y] (two numbers) which we map back to (pt, phi) slots.
    """
    dev = qml.device(device_name, wires=1)

    @qml.qnode(dev, interface="auto")
    def met_qnode(pt, phi, s_pt=1.0, s_phi=1.0):
        qml.RY(s_pt * pt, wires=0)
        qml.RZ(s_phi * phi, wires=0)
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(0))]

    return met_qnode


def make_group_circuits(device_name="lightning.gpu"):
    """Create all quantum circuits for different particle groups."""
    met_1q = make_met_1q(device_name)
    ele_4q = make_xyz_group_qnode(4, device_name)
    mu_4q  = make_xyz_group_qnode(4, device_name)
    jet_10q = make_xyz_group_qnode(10, device_name)
    return met_1q, ele_4q, mu_4q, jet_10q


def quantum_encode_particles(X, circuits=None, scales=None, batch_progress=False):
    """
    Quantum encode particle physics data.
    
    X: (N, 56)
    Returns: (N, 56), preserving original column layout.

    Mapping choices (documented):
      For electrons/muons/jets (one qubit per object):
        pt  <- ⟨Z⟩
        eta <- ⟨X⟩
        phi <- ⟨Y⟩
      For MET (one qubit total):
        pt  <- ⟨Z⟩
        phi <- ⟨Y⟩
    """
    if circuits is None:
        circuits = make_group_circuits()
    met_1q, ele_4q, mu_4q, jet_10q = circuits

    if scales is None:
        scales = build_scales()

    N = X.shape[0]
    out = np.empty((N, 56), dtype=float)

    for i in range(N):
        if batch_progress and (i % 100 == 0):
            print(f"Encoding {i}/{N}")

        x = X[i]

        # MET, using 1 qubit
        met_pt, met_phi = met_slice(x)
        met_Z, met_Y = met_1q(met_pt, met_phi, s_pt=scales["met"]["pt"], s_phi=scales["met"]["phi"])
        # Mapping pt<-Z, phi<-Y
        y_met = np.array([met_Z, met_Y])

        # Electrons, using 4 qubits
        e_pt, e_eta, e_phi = electrons_triplets(x)
        e_obs = np.array(
            ele_4q(
                e_pt, e_eta, e_phi,
                s_pt=scales["ele"]["pt"], s_eta=scales["ele"]["eta"], s_phi=scales["ele"]["phi"]
            )
        )  # [X0,Y0,Z0, X1,Y1,Z1, X2,Y2,Z2, X3,Y3,Z3]
        eX, eY, eZ = e_obs[0::3], e_obs[1::3], e_obs[2::3]

        # Muons, using 4 qubits
        m_pt, m_eta, m_phi = muons_triplets(x)
        m_obs = np.array(
            mu_4q(
                m_pt, m_eta, m_phi,
                s_pt=scales["mu"]["pt"], s_eta=scales["mu"]["eta"], s_phi=scales["mu"]["phi"]
            )
        )
        mX, mY, mZ = m_obs[0::3], m_obs[1::3], m_obs[2::3]

        # Jets, using 10 qubits
        j_pt, j_eta, j_phi = jets_triplets(x)
        j_obs = np.array(
            jet_10q(
                j_pt, j_eta, j_phi,
                s_pt=scales["jet"]["pt"], s_eta=scales["jet"]["eta"], s_phi=scales["jet"]["phi"]
            )
        )
        jX, jY, jZ = j_obs[0::3], j_obs[1::3], j_obs[2::3]

        # Stitch back to 56 in the original slot layout
        y = np.empty(56, dtype=float)
        a, b = IDX["met"]; y[a:b] = y_met

        # For groups: pt<-Z, eta<-X, phi<-Y
        y[IDX["e_pt"]]  = eZ
        y[IDX["e_eta"]] = eX
        y[IDX["e_phi"]] = eY

        y[IDX["m_pt"]]  = mZ
        y[IDX["m_eta"]] = mX
        y[IDX["m_phi"]] = mY

        y[IDX["j_pt"]]  = jZ
        y[IDX["j_eta"]] = jX
        y[IDX["j_phi"]] = jY

        out[i] = y

    return out


def encode_background_data(file_path, output_path='encoded_particles_objectwise.npy', 
                          device_name="lightning.gpu", batch_progress=True):
    """
    Complete pipeline for encoding background particle physics data.
    
    Args:
        file_path: Path to the H5 data file
        output_path: Path where to save the encoded data
        device_name: PennyLane device to use for quantum circuits
        batch_progress: Whether to show progress during encoding
    
    Returns:
        numpy array: The encoded particle data
    """
    print("Loading particle data...")
    background = LazyH5Array(file_path, "Particles")
    print(f"Loaded {len(background)} samples")
    
    circuits = make_group_circuits(device_name=device_name)
    
    scales = build_scales(
        met_pt=0.05, met_phi=1.0,
        e_pt=0.05, e_eta=np.pi/10, e_phi=1.0,
        m_pt=0.05, m_eta=np.pi/10, m_phi=1.0,
        j_pt=0.02, j_eta=np.pi/10, j_phi=1.0
    )
    
    print("Starting quantum encoding...")
    encoded = quantum_encode_particles(background, circuits, scales, batch_progress=batch_progress)
    
    print("Saving encoded data...")
    np.save(output_path, encoded)
    print(f"Saved encoded data with shape: {encoded.shape}")
    
    return encoded
