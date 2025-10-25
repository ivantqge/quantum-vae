#!/usr/bin/env python3
"""
Training script for Quantum VAE model.
Handles data loading, model training, and saving.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

from quantum_encoding import LazyH5Array
from vae_model import create_quantum_vae


def load_data(file_path, max_samples=None):
    """
    Load particle physics data from H5 file.
    
    Args:
        file_path: Path to H5 file
        max_samples: Maximum number of samples to load (None for all)
    
    Returns:
        numpy.ndarray: Loaded data
    """
    print(f"Loading data from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load data using lazy loading
    data_loader = LazyH5Array(file_path, "Particles")
    
    if max_samples is not None:
        print(f"Loading {max_samples} samples...")
        data = data_loader[:max_samples]
    else:
        print(f"Loading all {len(data_loader)} samples...")
        data = data_loader[:]
    
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    
    return data


def create_callbacks(stop_patience=15, lr_patience=10, model_dir="models"):
    """
    Create training callbacks.
    
    Args:
        stop_patience: Patience for early stopping
        lr_patience: Patience for learning rate reduction
        model_dir: Directory to save model checkpoints
    
    Returns:
        list: List of callbacks
    """
    os.makedirs(model_dir, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=stop_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=lr_patience,
            verbose=1,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks


def plot_training_history(history, save_path="training_history.png"):
    """
    Plot training history.
    
    Args:
        history: Training history object
        save_path: Path to save the plot
    """
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history plot saved to {save_path}")


def train_model(data, model_config, training_config, output_dir="outputs"):
    """
    Train the quantum VAE model.
    
    Args:
        data: Training data
        model_config: Model configuration dictionary
        training_config: Training configuration dictionary
        output_dir: Output directory for saving results
    
    Returns:
        tuple: (trained_model, training_history)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear any existing sessions
    tf.keras.backend.clear_session()
    
    # Create model
    print("Creating quantum VAE model...")
    model = create_quantum_vae(**model_config)
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=training_config['learning_rate'])
    model.compile(optimizer=optimizer)
    
    # Print model summaries
    print("\nEncoder Summary:")
    model.encoder.summary()
    print("\nDecoder Summary:")
    model.decoder.summary()
    
    # Create callbacks
    callbacks = create_callbacks(
        stop_patience=training_config['stop_patience'],
        lr_patience=training_config['lr_patience'],
        model_dir=os.path.join(output_dir, 'models')
    )
    
    # Calculate steps per epoch
    steps_per_epoch = len(data) // training_config['batch_size']
    model.steps_per_epoch = steps_per_epoch
    
    print(f"\nTraining Configuration:")
    print(f"  Data shape: {data.shape}")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Epochs: {training_config['epochs']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        x=data,
        validation_split=training_config['validation_split'],
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )
    
    # Save final model
    model_path = os.path.join(output_dir, 'final_model.h5')
    model.save_weights(model_path)
    print(f"Final model weights saved to {model_path}")
    
    # Plot training history
    plot_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    return model, history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Quantum VAE model')
    
    # Data arguments
    parser.add_argument('--data', '-d', 
                       default='data/background_for_training.h5',
                       help='Path to training data H5 file')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to use for training')
    
    # Model arguments
    parser.add_argument('--h-dim-1', type=int, default=32,
                       help='First hidden layer dimension')
    parser.add_argument('--h-dim-2', type=int, default=4,
                       help='Quantum layer dimension (number of qubits)')
    parser.add_argument('--latent-dim', type=int, default=3,
                       help='Latent space dimension')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--stop-patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--lr-patience', type=int, default=10,
                       help='Learning rate reduction patience')
    
    # Beta annealing arguments
    parser.add_argument('--cycle-length', type=int, default=10,
                       help='Cycle length for beta annealing')
    parser.add_argument('--min-beta', type=float, default=0.1,
                       help='Minimum beta value')
    parser.add_argument('--max-beta', type=float, default=0.8,
                       help='Maximum beta value')
    
    # Output arguments
    parser.add_argument('--output-dir', default='outputs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("Quantum VAE Training")
    print("=" * 50)
    
    try:
        # Load data
        data = load_data(args.data, args.max_samples)
        
        # Model configuration
        model_config = {
            'input_dim': data.shape[1],
            'h_dim_1': args.h_dim_1,
            'h_dim_2': args.h_dim_2,
            'latent_dim': args.latent_dim,
            'cycle_length': args.cycle_length,
            'min_beta': args.min_beta,
            'max_beta': args.max_beta
        }
        
        # Training configuration
        training_config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'validation_split': args.validation_split,
            'stop_patience': args.stop_patience,
            'lr_patience': args.lr_patience
        }
        
        # Train model
        model, history = train_model(data, model_config, training_config, args.output_dir)
        
        print("\nTraining completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error during training: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
