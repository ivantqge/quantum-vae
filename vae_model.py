#!/usr/bin/env python3
"""
Variational Autoencoder (VAE) model with quantum layers.
Contains encoder, decoder, and VAE model implementations.
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.backend as K
from quantum_layers import make_quantum_layer


class Sampling(keras.layers.Layer):
    """Sampling layer for VAE reparameterization trick."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def create_quantum_encoder(input_dim, h_dim_1, h_dim_2, latent_dim, l2_factor=1e-3):
    """
    Create quantum-enhanced encoder for VAE.
    
    Args:
        input_dim: Input data dimension
        h_dim_1: First hidden layer dimension
        h_dim_2: Quantum layer dimension (number of qubits)
        latent_dim: Latent space dimension
        l2_factor: L2 regularization factor
    
    Returns:
        keras.Model: Encoder model
    """
    inputs = keras.Input(shape=(input_dim,))
    
    # Classical preprocessing layer
    x = layers.Dense(
        h_dim_1,
        kernel_initializer=keras.initializers.HeNormal(seed=None),
        bias_initializer=keras.initializers.Zeros(),
        kernel_regularizer=l1_l2(l1=0, l2=l2_factor),
        name="dense1"
    )(inputs)
    
    # Quantum layer - wrap in a custom layer to handle KerasTensor properly
    class QuantumWrapper(tf.keras.layers.Layer):
        def __init__(self, quantum_dim, **kwargs):
            super().__init__(**kwargs)
            self.quantum_dim = quantum_dim
            self.pqc_layer = make_quantum_layer(quantum_dim, depth=3, use_vectorized=True)
        
        def call(self, inputs):
            # Take only the first quantum_dim features for the quantum layer
            quantum_input = inputs[:, :self.quantum_dim]
            
            if hasattr(self.pqc_layer, 'quantum_params'):  # PennyLane layer
                # PennyLane layers expect (batch, features) format
                return self.pqc_layer(quantum_input)
            else:  # TensorFlow Quantum layer
                # TFQ expects (batch, 1, features) format
                quantum_input_expanded = tf.expand_dims(quantum_input, axis=1)
                return self.pqc_layer(quantum_input_expanded)
        
        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.quantum_dim)
    
    quantum_wrapper = QuantumWrapper(h_dim_2)
    x = quantum_wrapper(x)
    
    # Latent space layers
    z_mean = layers.Dense(
        latent_dim, 
        name='z_mean',
        kernel_initializer=keras.initializers.HeNormal(seed=None),
        bias_initializer=keras.initializers.Zeros(),
        kernel_regularizer=l1_l2(l1=0, l2=l2_factor)
    )(x)
    
    z_logvar = layers.Dense(
        latent_dim, 
        name='z_log_var',
        kernel_initializer=keras.initializers.Zeros(),
        bias_initializer=keras.initializers.Zeros(),
        kernel_regularizer=l1_l2(l1=0, l2=l2_factor)
    )(x)
    
    z = Sampling()([z_mean, z_logvar])
    encoder = keras.Model(inputs, [z_mean, z_logvar, z], name='encoder')
    return encoder


def create_decoder(input_dim, h_dim_1, h_dim_2, latent_dim, l2_factor=1e-3):
    """
    Create decoder for VAE.
    
    Args:
        input_dim: Output data dimension
        h_dim_1: First hidden layer dimension
        h_dim_2: Second hidden layer dimension
        latent_dim: Latent space dimension
        l2_factor: L2 regularization factor
    
    Returns:
        keras.Model: Decoder model
    """
    inputs = keras.Input(shape=(latent_dim,))
    
    x = layers.Dense(
        h_dim_2,
        activation='relu',
        kernel_initializer=keras.initializers.HeNormal(seed=None),
        bias_initializer=keras.initializers.Zeros(),
        kernel_regularizer=l1_l2(l1=0, l2=l2_factor)
    )(inputs)
    
    x = layers.Dense(
        h_dim_1,
        activation='relu',
        kernel_initializer=keras.initializers.HeNormal(seed=None),
        bias_initializer=keras.initializers.Zeros(),
        kernel_regularizer=l1_l2(l1=0, l2=l2_factor)
    )(x)
    
    z = layers.Dense(
        input_dim,
        kernel_initializer=keras.initializers.HeNormal(seed=None),
        bias_initializer=keras.initializers.Zeros(),
        kernel_regularizer=l1_l2(l1=0, l2=l2_factor)
    )(x)
    
    decoder = keras.Model(inputs, z, name='decoder')
    return decoder


class QuantumVAE(keras.Model):
    """
    Quantum-enhanced Variational Autoencoder with cyclical beta annealing.
    """
    
    def __init__(self, encoder, decoder, steps_per_epoch=3125, cycle_length=10, 
                 min_beta=0.1, max_beta=0.85, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.beta_tracker = keras.metrics.Mean(name="beta")
        self.steps_per_epoch = steps_per_epoch
        self.cycle_length = tf.cast(cycle_length, tf.float32)
        self.min_beta = tf.cast(min_beta, tf.float32)
        self.max_beta = tf.cast(max_beta, tf.float32)
        self.beta = tf.Variable(min_beta, dtype=tf.float32)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.beta_tracker,
        ]

    def cyclical_annealing_beta(self, epoch):
        """Cyclical beta annealing schedule."""
        cycle = tf.floor(1.0 + epoch / self.cycle_length)
        x = tf.abs(epoch / self.cycle_length - cycle + 1)
        return self.min_beta + (self.max_beta - self.min_beta) * tf.minimum(x, 1.0)

    def train_step(self, data):
        """Training step with cyclical beta annealing."""
        # Get the current epoch number
        epoch = tf.cast(self.optimizer.iterations / self.steps_per_epoch, tf.float32)

        # Update beta
        self.beta.assign(self.cyclical_annealing_beta(epoch))
        
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Masked reconstruction loss (handle zero values)
            mask = K.cast(K.not_equal(data, 0), K.floatx())
            reconstruction_loss = tf.reduce_mean(
                keras.losses.mse(mask * data, mask * reconstruction)
            )
            reconstruction_loss *= (1 - self.beta)

            # KL divergence loss
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= self.beta

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reco_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "beta": self.beta,
        }

    def test_step(self, data):
        """Validation step."""
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        
        mask = K.cast(K.not_equal(data, 0), K.floatx())
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mse(mask * data, mask * reconstruction)
        )
        reconstruction_loss *= (1 - self.beta)

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= self.beta

        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reco_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "beta": self.beta,
        }

    def call(self, data):
        """Forward pass."""
        z_mean, z_log_var, x = self.encoder(data)
        reconstruction = self.decoder(x)
        return {
            "z_mean": z_mean,
            "z_log_var": z_log_var,
            "reconstruction": reconstruction
        }


def create_quantum_vae(input_dim, h_dim_1=32, h_dim_2=4, latent_dim=3, 
                      steps_per_epoch=3125, cycle_length=10, 
                      min_beta=0.1, max_beta=0.8):
    """
    Create a complete quantum VAE model.
    
    Args:
        input_dim: Input data dimension
        h_dim_1: First hidden layer dimension
        h_dim_2: Quantum layer dimension (number of qubits)
        latent_dim: Latent space dimension
        steps_per_epoch: Steps per epoch for beta annealing
        cycle_length: Cycle length for beta annealing
        min_beta: Minimum beta value
        max_beta: Maximum beta value
    
    Returns:
        QuantumVAE: Complete quantum VAE model
    """
    encoder = create_quantum_encoder(input_dim, h_dim_1, h_dim_2, latent_dim)
    decoder = create_decoder(input_dim, h_dim_1, h_dim_2, latent_dim)
    
    vae = QuantumVAE(
        encoder, decoder, 
        steps_per_epoch=steps_per_epoch,
        cycle_length=cycle_length,
        min_beta=min_beta,
        max_beta=max_beta
    )
    
    return vae
