"""
Collections of some classical autoencoders to compare to
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
from keras.initializers import Constant
from keras.layers import PReLU
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


class AucCallback(keras.callbacks.Callback):
    """Computes and saves the auc for given test background and signal data for every epoch"""

    def __init__(self, model, x_test_bg, x_test_signal):
        """Creates the callback object

        Args:
          model (keras model): model that will be trained
          x_test_bg (array): test data for the background events (events the model is trained on)
          x_test_signal (array): test data for the signal events (events the model should tag as anomalous)"""
        self.model = model
        self.x_test_bg = x_test_bg
        self.x_test_signal = x_test_signal
        self.keras_metric = tf.keras.metrics.Mean("jaccard_score")
        self.epoch = 0
        self.hist = []

    def on_epoch_end(self, batch, logs=None):
        """Callback that is executed on the end of every epoch. Computes AUC and saves it"""
        self.epoch += 1
        self.keras_metric.reset_state()
        recon_bg = self.model.predict(self.x_test_bg)
        recon_signal = self.model.predict(self.x_test_signal)
        bce_background = tf.keras.losses.binary_crossentropy(self.x_test_bg, recon_bg, axis=(1,2,3)).numpy()
        bce_signal = tf.keras.losses.binary_crossentropy(self.x_test_signal, recon_signal, axis=(1,2,3)).numpy()
        y_true = np.append(np.zeros(len(bce_background)), np.ones(len(bce_signal)))
        y_pred = np.append(bce_background, bce_signal)
        auc = roc_auc_score(y_true, y_pred)
        self.hist.append(auc)

class Convolutional_Autoencoder_12x12(Model):
    """Convolutional Autoencoder model specifically for comparasion with Quantum models with a small number of parameters"""
    def __init__(self, latent_dim):
        """Create the model with given latenspace. Will have around 3000 parameters

        Args:
          latent_dim (int): number of latent neurons"""
        super(Convolutional_Autoencoder_12x12, self).__init__()

        self.encoder = tf.keras.Sequential([
          layers.Input(shape=(12, 12, 1)),
          layers.Conv2D(4, kernel_size=4, strides=1, activation='relu', padding='same'),
          layers.Conv2D(4, kernel_size=4, strides=1, activation='relu', padding='same'),
          layers.AveragePooling2D(pool_size=2),
          layers.Conv2D(2, kernel_size=4, strides=1, activation='relu', padding='same'),
          layers.Conv2D(2, kernel_size=4, strides=1, activation='relu', padding='same'),
          layers.Flatten(),
          # Exclude last dense layer for less parameters
          # layers.Dense(36*1, activation='relu'),
          layers.Dense(latent_dim, activation='relu')])

        self.decoder = tf.keras.Sequential([
          layers.Dense(36*1, activation='relu'),
          layers.Reshape((6, 6, 1)),
          layers.Conv2D(2, kernel_size=4, strides=1, activation='relu', padding='same'),
          layers.Conv2D(2, kernel_size=4, strides=1, activation='relu', padding='same'),
          layers.UpSampling2D(size=2),
          layers.Conv2D(4, kernel_size=4, strides=1, activation='relu', padding='same'),
          layers.Conv2D(4, kernel_size=4, strides=1, activation='relu', padding='same'),
          layers.Conv2D(1, kernel_size=4, strides=1, activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Convolutional_Autoencoder_Large(Model):
    """Larger conv ae model for 40x40 datset with a large amount of parameters.
       The model mostly follows:
       Finke, Thorben, et al.'Autoencoders for unsupervised anomaly detection
       in high energy physics.' Journal of High Energy Physics 2021.6 (2021): 1-32."""
    def __init__(self, latent_dim):
        super(Convolutional_Autoencoder_Large, self).__init__()

        self.encoder = tf.keras.Sequential([
          layers.Input(shape=(40, 40, 1)),
          layers.Conv2D(10, kernel_size=4, strides=1, padding='same'),
          PReLU(alpha_initializer=Constant(value=0.25)),
          layers.Conv2D(10, kernel_size=4, strides=1, padding='same'),
          PReLU(alpha_initializer=Constant(value=0.25)),
          layers.AveragePooling2D(pool_size=2),
          layers.Conv2D(5, kernel_size=4, strides=1, padding='same'),
          PReLU(alpha_initializer=Constant(value=0.25)),
          layers.Conv2D(5, kernel_size=4, strides=1, padding='same'),
          PReLU(alpha_initializer=Constant(value=0.25)),
          layers.Flatten(),
          layers.Dense(400),
          PReLU(alpha_initializer=Constant(value=0.25)),
          layers.Dense(100),
          PReLU(alpha_initializer=Constant(value=0.25)),
          layers.Dense(latent_dim),
          PReLU(alpha_initializer=Constant(value=0.25))])

        self.decoder = tf.keras.Sequential([
          layers.Dense(100),
          PReLU(alpha_initializer=Constant(value=0.25)),
          layers.Dense(400),
          PReLU(alpha_initializer=Constant(value=0.25)),
          layers.Reshape((20, 20, 1)),
          layers.Conv2D(5, kernel_size=4, strides=1, padding='same'),
          PReLU(alpha_initializer=Constant(value=0.25)),
          layers.Conv2D(5, kernel_size=4, strides=1, padding='same'),
          PReLU(alpha_initializer=Constant(value=0.25)),
          layers.UpSampling2D(size=2),
          layers.Conv2D(10, kernel_size=4, strides=1, padding='same'),
          PReLU(alpha_initializer=Constant(value=0.25)),
          layers.Conv2D(1, kernel_size=4, strides=1, activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Sampling(layers.Layer):
    """Sampling layer for a varaiational autoencoder"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_dim = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch_dim, latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    """Convolutional variational autoencoder"""
    def __init__(self,  latent_dim, input_size, **kwargs):
        """Create the vae with variable latent dimension and input image size

        Args:
          latent_dim (int): dimension of latent space
          input_size (int): width of input images (must be dividable by four)"""
        super(VAE, self).__init__(**kwargs)

        encoder_inputs = keras.Input(shape=(input_size, input_size, 1))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(input_size // 4 * input_size // 4 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((input_size // 4, input_size // 4, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, x):
        _, latent_z = self.encoder(x)
        reconstruction = self.decoder(latent_z)
        return reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
              tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )
            # compute kl divergence
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
