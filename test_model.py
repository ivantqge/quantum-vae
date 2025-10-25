#!/usr/bin/env python3
"""
Test script to verify quantum VAE model training works correctly.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from quantum_encoding import LazyH5Array
from vae_model import create_quantum_vae


def create_test_data(n_samples=1000, n_features=56):
    """
    Create synthetic test data for model testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
    
    Returns:
        numpy.ndarray: Synthetic test data
    """
    print(f"Creating synthetic test data: {n_samples} samples, {n_features} features")
    
    # Create realistic particle physics-like data
    np.random.seed(42)
    
    # MET features (indices 0-1)
    data = np.zeros((n_samples, n_features))
    data[:, 0] = np.random.exponential(50, n_samples)  # MET pt
    data[:, 1] = np.random.uniform(-np.pi, np.pi, n_samples)  # MET phi
    
    # Electron features (indices 2-13)
    for i in range(4):
        data[:, 2+i] = np.random.exponential(30, n_samples)  # e pt
        data[:, 6+i] = np.random.uniform(-2.5, 2.5, n_samples)  # e eta
        data[:, 10+i] = np.random.uniform(-np.pi, np.pi, n_samples)  # e phi
    
    # Muon features (indices 14-25)
    for i in range(4):
        data[:, 14+i] = np.random.exponential(25, n_samples)  # mu pt
        data[:, 18+i] = np.random.uniform(-2.5, 2.5, n_samples)  # mu eta
        data[:, 22+i] = np.random.uniform(-np.pi, np.pi, n_samples)  # mu phi
    
    # Jet features (indices 26-55)
    for i in range(10):
        data[:, 26+i] = np.random.exponential(40, n_samples)  # jet pt
        data[:, 36+i] = np.random.uniform(-3, 3, n_samples)  # jet eta
        data[:, 46+i] = np.random.uniform(-np.pi, np.pi, n_samples)  # jet phi
    
    # Add some zeros to simulate missing particles
    zero_mask = np.random.random((n_samples, n_features)) < 0.3
    data[zero_mask] = 0
    
    print(f"Data shape: {data.shape}")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Non-zero values: {np.count_nonzero(data) / data.size * 100:.1f}%")
    
    return data


def test_model_creation():
    """Test if the quantum VAE model can be created successfully."""
    print("\n" + "="*50)
    print("Testing Model Creation")
    print("="*50)
    
    try:
        # Create model
        model = create_quantum_vae(
            input_dim=56,
            h_dim_1=32,
            h_dim_2=4,
            latent_dim=3,
            steps_per_epoch=10,
            cycle_length=5,
            min_beta=0.1,
            max_beta=0.8
        )
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer)
        
        # Build the model with a test input to get parameter counts
        test_input = tf.random.normal((1, 56))
        _ = model(test_input)
        
        print("✅ Model created and compiled successfully!")
        print(f"Model summary:")
        print(f"  Encoder: {model.encoder.count_params()} parameters")
        print(f"  Decoder: {model.decoder.count_params()} parameters")
        try:
            print(f"  Total: {model.count_params()} parameters")
        except:
            print(f"  Total: Model not fully built yet")
        
        return model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None


def test_forward_pass(model, test_data):
    """Test if the model can perform forward pass."""
    print("\n" + "="*50)
    print("Testing Forward Pass")
    print("="*50)
    
    try:
        # Test forward pass
        batch_size = 10
        test_batch = test_data[:batch_size]
        
        print(f"Input shape: {test_batch.shape}")
        
        # Forward pass
        outputs = model(test_batch)
        
        print("✅ Forward pass successful!")
        print(f"Output keys: {list(outputs.keys())}")
        print(f"z_mean shape: {outputs['z_mean'].shape}")
        print(f"z_log_var shape: {outputs['z_log_var'].shape}")
        print(f"reconstruction shape: {outputs['reconstruction'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False


def test_training_step(model, test_data):
    """Test if the model can perform a training step."""
    print("\n" + "="*50)
    print("Testing Training Step")
    print("="*50)
    
    try:
        # Test training step
        batch_size = 32
        test_batch = test_data[:batch_size]
        
        print(f"Training batch shape: {test_batch.shape}")
        
        # Single training step
        loss_dict = model.train_step(test_batch)
        
        print("✅ Training step successful!")
        print(f"Loss components:")
        for key, value in loss_dict.items():
            print(f"  {key}: {float(value):.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False


def test_short_training(model, test_data, epochs=3):
    """Test short training session."""
    print("\n" + "="*50)
    print(f"Testing Short Training ({epochs} epochs)")
    print("="*50)
    
    try:
        # Prepare data
        train_data = test_data[:800]
        val_data = test_data[800:]
        
        print(f"Training data: {train_data.shape}")
        print(f"Validation data: {val_data.shape}")
        
        # Train for a few epochs
        history = model.fit(
            x=train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        print("✅ Short training successful!")
        print(f"Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
        
        return history
        
    except Exception as e:
        print(f"❌ Short training failed: {e}")
        return None


def plot_test_results(history):
    """Plot training results."""
    if history is None:
        return
        
    print("\n" + "="*50)
    print("Plotting Results")
    print("="*50)
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Total loss
        axes[0, 0].plot(history.history['loss'], label='Training')
        axes[0, 0].plot(history.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Reconstruction loss
        axes[0, 1].plot(history.history['reco_loss'], label='Training')
        axes[0, 1].plot(history.history['val_reco_loss'], label='Validation')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # KL loss
        axes[1, 0].plot(history.history['kl_loss'], label='Training')
        axes[1, 0].plot(history.history['val_kl_loss'], label='Validation')
        axes[1, 0].set_title('KL Divergence Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Beta value
        axes[1, 1].plot(history.history['beta'], label='Beta')
        axes[1, 1].set_title('Beta Annealing')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Beta')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('test_training_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Results plotted and saved to 'test_training_results.png'")
        
    except Exception as e:
        print(f"❌ Plotting failed: {e}")


def test_quantum_layer_parameter_updates(model, test_data):
    """
    Test if the quantum layer parameters are actually being updated during training.
    
    Args:
        model: Compiled VAE model
        test_data: Test data for training
    
    Returns:
        bool: True if quantum parameters are updated
    """
    print(f"\n{'='*50}")
    print("Testing Quantum Layer Parameter Updates")
    print(f"{'='*50}")
    
    try:
        # Find the quantum layer in the encoder
        quantum_layer = None
        quantum_wrapper = None
        
        for layer in model.encoder.layers:
            if hasattr(layer, 'pqc_layer'):  # QuantumWrapper
                quantum_wrapper = layer
                quantum_layer = layer.pqc_layer
                break
            elif hasattr(layer, 'quantum_params'):  # Direct quantum layer
                quantum_layer = layer
                break
        
        if quantum_layer is None:
            print("❌ No quantum layer found in encoder!")
            print("Available layers:")
            for i, layer in enumerate(model.encoder.layers):
                print(f"  {i}: {type(layer).__name__}")
            return False
        
        print(f"Found quantum wrapper: {type(quantum_wrapper).__name__ if quantum_wrapper else 'None'}")
        print(f"Found quantum layer: {type(quantum_layer).__name__}")
        
        # Get initial quantum parameters
        initial_params = quantum_layer.quantum_params.numpy().copy()
        print(f"Initial quantum params shape: {initial_params.shape}")
        print(f"Initial quantum params range: [{initial_params.min():.6f}, {initial_params.max():.6f}]")
        print(f"Initial quantum params mean: {initial_params.mean():.6f}")
        print(f"Initial quantum params: {initial_params}")
        
        # Create a small training batch
        train_batch = test_data[:32]
        
        # Perform a few training steps
        print("\nPerforming 5 training steps...")
        for step in range(5):
            with tf.GradientTape() as tape:
                # Forward pass
                z_mean, z_log_var, z = model.encoder(train_batch)
                reconstruction = model.decoder(z)
                
                # Compute loss
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.square(train_batch - reconstruction), axis=1)
                )
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
                )
                total_loss = reconstruction_loss + 0.1 * kl_loss
            
            # Compute gradients and apply them
            gradients = tape.gradient(total_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            print(f"Step {step+1}: Loss = {total_loss.numpy():.6f}")
        
        # Get final quantum parameters
        final_params = quantum_layer.quantum_params.numpy()
        print(f"\nFinal quantum params range: [{final_params.min():.6f}, {final_params.max():.6f}]")
        print(f"Final quantum params mean: {final_params.mean():.6f}")
        print(f"Final quantum params: {final_params}")
        
        # Check if parameters changed
        param_change = np.abs(final_params - initial_params)
        max_change = param_change.max()
        mean_change = param_change.mean()
        
        print(f"Max parameter change: {max_change:.6f}")
        print(f"Mean parameter change: {mean_change:.6f}")
        
        # Parameters should change if gradients are flowing
        if max_change > 1e-6:  # Threshold for meaningful change
            print("✅ Quantum layer parameters updated during training!")
            print("✅ Gradients are flowing through the quantum layer!")
            return True
        else:
            print("❌ Quantum layer parameters did not change significantly!")
            print("❌ Gradients may not be flowing through the quantum layer!")
            return False
            
    except Exception as e:
        print(f"❌ Parameter update test failed: {e}")
        return False


def main():
    """Main test function."""
    print("Quantum VAE Model Testing")
    print("="*50)
    
    # Create test data
    test_data = create_test_data(n_samples=1000, n_features=56)
    
    # Test model creation
    model = test_model_creation()
    if model is None:
        print("❌ Cannot proceed without working model")
        return 1
    
    # Test forward pass
    if not test_forward_pass(model, test_data):
        print("❌ Cannot proceed without working forward pass")
        return 1
    
    # Test quantum layer parameter updates
    if not test_quantum_layer_parameter_updates(model, test_data):
        print("❌ Quantum layer parameters not updating properly")
        return 1
    
    # Test short training
    history = test_short_training(model, test_data, epochs=5)
    if history is None:
        print("❌ Training failed")
        return 1
    
    # Plot results
    plot_test_results(history)
    
    print("\n" + "="*50)
    print("🎉 All tests passed! Model is ready for training.")
    print("="*50)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
