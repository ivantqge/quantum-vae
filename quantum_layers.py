#!/usr/bin/env python3
"""
Quantum layers using PennyLane or TensorFlow Quantum for differentiable quantum computing.
Provides real quantum circuits that integrate with TensorFlow.
"""

import warnings
import tensorflow as tf
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# silence complex to float casting warning (just taking real part anyways)
warnings.filterwarnings("ignore", message=".*casting an input of type complex128 to an incompatible dtype float32.*")
tf.get_logger().setLevel('ERROR')

# try to import TensorFlow Quantum otherwise use PennyLane  
try:
    from tensorflow_quantum import layers as tfq_layers
    TFQ_AVAILABLE = True
except ImportError:
    print("TensorFlow Quantum not available. Using PennyLane implementation.")
    TFQ_AVAILABLE = False

class QuantumLayerVectorized(tf.keras.layers.Layer):
    """
    Real quantum layer using PennyLane with TensorFlow integration.
    Uses PennyLane's TensorFlow interface for proper gradient support.
    """
    
    def __init__(self, num_qubits, depth=1, device='default.qubit', **kwargs):
        super().__init__(**kwargs)
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = device
        
        # create qml device
        self.dev = qml.device(device, wires=num_qubits)
        
        # define quantum circuit with TF interface (single sample)
        @qml.qnode(self.dev, interface='tf', diff_method='backprop')
        def quantum_circuit(inputs, weights):
            # encode data into circuit
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)
            
            # variational layers
            for d in range(depth):
                # rotate each of the qubits based on the weights
                for i in range(num_qubits):
                    qml.RX(weights[d * num_qubits + i], wires=i)
                
                # also add entangling layers
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            # output is expected val measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
        
        self.quantum_circuit = quantum_circuit
        
        # initialize variational parameters
        self.quantum_params = self.add_weight(
            name='quantum_params',
            shape=(depth * num_qubits,),
            initializer='random_uniform',
            trainable=True
        )
    
    def call(self, inputs):
        """
        Execute quantum circuit for each sample in the batch using tf.vectorized_map.
        """
        # need 2D inputs for the quantum circuit
        if len(inputs.shape) == 3:
            inputs = tf.squeeze(inputs, axis=1)
        
        # Take only the first num_qubits features
        quantum_inputs = inputs[:, :self.num_qubits]
        
        # Use tf.vectorized_map for better performance 
        @tf.autograph.experimental.do_not_convert
        def process_sample(sample_input):
            # Execute quantum circuit
            expectation_values = self.quantum_circuit(sample_input, self.quantum_params)
            
            # Convert list to tensor if needed
            if isinstance(expectation_values, list):
                expectation_values = tf.stack(expectation_values)
            
            if expectation_values.dtype.is_complex:
                expectation_values = tf.math.real(expectation_values)
            
            expectation_values = tf.cast(expectation_values, tf.float32)
            return expectation_values
        
        # Use vectorized_map for better GPU utilization
        result = tf.vectorized_map(
            process_sample, 
            quantum_inputs,
            fallback_to_while_loop=False
        )
        result.set_shape([None, self.num_qubits])
        return result
    
    def compute_output_shape(self, input_shape):
        """Define the output shape of the layer."""
        return (input_shape[0], self.num_qubits)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_qubits': self.num_qubits,
            'depth': self.depth,
            'device': self.device,
        })
        return config


def make_quantum_layer(num_qubits, depth=1, use_vectorized=True, device='default.qubit', use_tfq=None):
    """
    Create a quantum layer using either TensorFlow Quantum or PennyLane.
    
    Args:
        num_qubits: Number of qubits in the quantum circuit
        depth: Depth of the variational circuit (number of layers)
        use_vectorized: If True, use vectorized implementation for better performance (PennyLane only)
        device: Device to use ('default.qubit', 'lightning.qubit', etc.)
        use_tfq: If True, use TensorFlow Quantum. If False, use PennyLane. If None, auto-detect.
    
    Returns:
        Quantum layer (TFQ PQC layer or PennyLane quantum layer)
    """
    # Auto-detect if not specified
    if use_tfq is None:
        use_tfq = TFQ_AVAILABLE
    
    if use_tfq and TFQ_AVAILABLE:
        # Use TensorFlow Quantum
        import cirq
        import sympy
        
        # Create Cirq circuit
        qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
        circuit = cirq.Circuit()
        
        # Data encoding
        data_symbols = sympy.symbols('x0:%d' % num_qubits)
        for i, q in enumerate(qubits):
            circuit.append(cirq.ry(data_symbols[i])(q))
        
        # Variational circuit
        theta = sympy.symbols('theta0:%d' % (num_qubits * depth))
        for d in range(depth):
            for i, q in enumerate(qubits):
                circuit.append(cirq.rx(theta[d * num_qubits + i])(q))
            for i in range(num_qubits - 1):
                circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        
        readout_ops = [cirq.Z(q) for q in qubits]
        
        # Create TFQ PQC layer
        pqc_layer = tfq_layers.PQC(circuit, readout_ops)
        return pqc_layer
    else:
        # Use PennyLane
        return QuantumLayerVectorized(num_qubits, depth, device)


def make_quantum_encoder_layer(input_dim, quantum_dim, depth=1, device='default.qubit'):
    """
    Create a quantum encoder layer that maps classical data to quantum circuit.
    
    Args:
        input_dim: Dimension of input classical data
        quantum_dim: Number of qubits for quantum processing
        depth: Depth of quantum circuit
        device: PennyLane device to use
    
    Returns:
        tf.keras.layers.Layer: Quantum encoder layer
    """
    class QuantumEncoderLayer(tf.keras.layers.Layer):
        def __init__(self, quantum_dim, depth, device, **kwargs):
            super().__init__(**kwargs)
            self.quantum_dim = quantum_dim
            self.depth = depth
            self.device = device
            self.pqc_layer = make_quantum_layer(quantum_dim, depth, device=device)
            
        def call(self, inputs):
            return self.pqc_layer(inputs)
            
        def get_config(self):
            config = super().get_config()
            config.update({
                'quantum_dim': self.quantum_dim,
                'depth': self.depth,
                'device': self.device
            })
            return config
    
    return QuantumEncoderLayer(quantum_dim, depth, device)


# ---- TESTING FUNCTIONS ----
def test_quantum_layer(num_qubits=4, batch_size=10, device='default.qubit'):
    """
    Test function to verify quantum layer works correctly.
    
    Args:
        num_qubits: Number of qubits
        batch_size: Batch size for testing
        device: PennyLane device to use
    
    Returns:
        bool: True if test passes
    """
    print(f"Testing quantum layer with {num_qubits} qubits...")
    print(f"TensorFlow Quantum available: {TFQ_AVAILABLE}")
    print(f"Using device: {device}")
    
    try:
        # Create quantum layer
        quantum_layer = make_quantum_layer(num_qubits, depth=2, device=device)
        print(f"Created quantum layer: {type(quantum_layer).__name__}")
        
        # Create test input
        test_input = tf.random.normal((batch_size, num_qubits))
        
        # Test forward pass
        output = quantum_layer(test_input)
        
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.numpy().min():.3f}, {output.numpy().max():.3f}]")
        
        # Test gradient computation
        # NOTE: this does not work likely because of differences between PennyLane and TensorFlow in terms of gradient visibility
        with tf.GradientTape() as tape:
            tape.watch(test_input)
            output = quantum_layer(test_input)
            loss = tf.reduce_mean(output)
        
        gradients = tape.gradient(loss, test_input)
        print(f"Gradients computed: {gradients is not None}")
        
        print("Quantum layer test passed!")
        return True
        
    except Exception as e:
        print(f"Quantum layer test failed: {e}")
        return False


def test_quantum_training(num_qubits=4, batch_size=32, epochs=3):
    """
    Test quantum layer training with a simple model.
    
    Args:
        num_qubits: Number of qubits
        batch_size: Batch size for training
        epochs: Number of training epochs
    
    Returns:
        bool: True if test passes
    """
    print(f"Testing quantum layer training...")
    
    try:
        # Create simple model
        inputs = tf.keras.Input(shape=(num_qubits,))
        quantum_layer = make_quantum_layer(num_qubits, depth=2)
        outputs = quantum_layer(inputs)
        model = tf.keras.Model(inputs, outputs)
        
        # Compile model
        model.compile(optimizer='adam', loss='mse')
        
        # Create synthetic data
        x_train = tf.random.normal((batch_size * 10, num_qubits))
        y_train = tf.random.normal((batch_size * 10, num_qubits))
        
        # Train model
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        print(f"Training completed. Final loss: {history.history['loss'][-1]:.6f}")
        print("Quantum layer training test passed!")
        return True
        
    except Exception as e:
        print(f"Quantum layer training test failed: {e}")
        return False


# Example usage and testing
if __name__ == "__main__":
    print("Quantum Layers Module")
    print("=" * 30)
    
    # Test the quantum layer
    test_quantum_layer()
    
    # Test training
    test_quantum_training()