from typing import Optional, Union, Tuple, Dict

from quple.models.generative import QGAN

import tensorflow as tf
from tensorflow.keras import Model

class QWGAN(QGAN):
    """Quantum Wasserstein Generative Adversarial Network (QWGAN)
    """    
    def __init__(self, generator:Model,
                 discriminator:Model,
                 latent_dim:Optional[Union[int, Tuple]]=None,
                 n_disc:int=3,
                 epochs:int=100, 
                 batch_size:int=32,
                 optimizer:Optional[Union[str, Dict]]=None,
                 optimizer_kwargs:Optional[Dict]=None,
                 name:str='QGAN',
                 random_state:Optional[int]=None,
                 checkpoint_dir:Optional[str]=None,
                 checkpoint_interval:int=10,
                 checkpoint_max_to_keep:Optional[int]=None):
        """ Creates a QGAN model equipped with a generator and a discriminator neural networks.
            The Wasserstein distance is used as the loss function.
        Args:
            generator: tensorflow.keras.Model
                A keras model representing the generator neural network. It can contain
                both classical and quantum layers.
            discriminator: cirq.Circuit or quple.QuantumCircuit instace
                A keras model representing the discriminator neural network. It can contain
                both classical and quantum layers.
            latent_dim: (Optional) int or tuple of int
                Dimensions of the latent space. It specifies the shape of the input noise
                (drawn from a normal distribution) passed to the generator neural network.
            n_disc: (Optional) int, default=3
                Number of discriminator iterations per generator iteration in each epoch.
            epochs: int, default=100
                Number of epochs for the training.
            batch_size: int, default=10
                Number of training examples used in one iteration. Note that the true batch
                size is batch_size * n_disc so that training samples of size batch_size is
                passed to the discriminator in each discriminator step.
            name: (Optional) string
                Name given to the model.
            random_state: (Optional) int
                The random state used at the beginning of a training
                for reproducible result.
            checkpoint_dir: Optional[string]
                The path to a directory in which to write checkpoints. If None,
                no checkpoint will be saved.
            checkpoint_interval: int, default=10
                Number of epochs between each checkpoint.
            checkpoint_max_to_keep: (Optional) int
        """
        super().__init__(generator=generator,
                 discriminator=discriminator,
                 latent_dim=latent_dim,
                 n_disc=n_disc,
                 epochs=epochs,
                 batch_size=batch_size,
                 optimizer=optimizer,
                 optimizer_kwargs=optimizer_kwargs,
                 name=name,
                 random_state=random_state,
                 checkpoint_dir=checkpoint_dir,
                 checkpoint_interval=checkpoint_interval,
                 checkpoint_max_to_keep=checkpoint_max_to_keep)

    @tf.function
    def D_loss(self, real_output, fake_output):
        """Compute discriminator loss."""
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
    
    @tf.function
    def G_loss(self, fake_output):
        """Compute generator loss."""
        return -tf.reduce_mean(fake_output)

    @tf.function
    def gradient_penalty(self, f, x_real, x_fake):
        alpha = tf.random.uniform([self.batch_size, 1], 0., 1.)
        diff = x_fake - x_real
        inter = x_real + (alpha * diff)
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = f(inter)
        grad = t.gradient(pred, [inter])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1))
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp    