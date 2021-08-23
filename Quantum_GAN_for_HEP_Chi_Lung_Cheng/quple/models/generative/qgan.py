import os
from typing import Union, Optional, Callable, Dict, Tuple
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Optimizer

from quple.models import AbstractModel

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from IPython.display import clear_output
except ImportError:
    clear_output = None

class QGAN(AbstractModel):
    """Quantum Generative Adversarial Network (QGAN)
    """    
    def __init__(self, generator:Model, 
                 discriminator:Model,
                 latent_dim:Optional[Union[int, Tuple]]=None,
                 n_disc:int=1,
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
            The modified Jensen-Shanon divergence is used as the loss function.
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
                Number of checkpoints to keep. If None, all checkpoints are kept.
        """
        self.set_random_state(random_state)
        
        self.start_epoch = 0
        self.n_disc = n_disc
        self.epochs = epochs
        self.batch_size = batch_size

        if not isinstance(generator, Model):
            raise ValueError("generator must be a tf.keras.Model instance")
        self.G = generator
        if not isinstance(discriminator, Model):
            raise ValueError("discriminator must be a tf.keras.Model instance")            
        self.D = discriminator
        
        if latent_dim is None:
            try:
                self.latent_dim = self.G.input_shape[1:]
            except:
                raise RuntimeError("cannot infer input shape from generator")
        else:
            self.latent_dim = tuple(latent_dim)
        self.z_batch_shape = (self.batch_size,) + self.latent_dim            

        self.G_optimizer, self.D_optimizer = self._get_optimizers(optimizer, optimizer_kwargs)
        
        self.loss_function = self._get_loss_function()
        
        self.visualization_interval = None
        self.image_shape = None
        self.n_image_to_show = 0
        
        print('Summary of Generator')
        self.G.summary()
        
        print('Summary of Discriminator')
        self.D.summary()
        
        super().__init__(name=name, random_state=random_state,
                         checkpoint_dir=checkpoint_dir,
                         checkpoint_interval=checkpoint_interval,
                         checkpoint_max_to_keep=checkpoint_max_to_keep)

    def _create_checkpoint(self):
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                         generator_optimizer=self.G_optimizer,
                                         discriminator_optimizer=self.D_optimizer,
                                         generator=self.G,
                                         discriminator=self.D)
        return checkpoint
        
        
    def _get_loss_function(self):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
    @staticmethod
    def _get_optimizers(optimizer:Optional[Union[str, Dict]]=None,
                        optimizer_kwargs:Optional[Dict]=None):
        if isinstance(optimizer, str):
            optimizer = {"generator": optimizer, "discriminator": optimizer}
        elif isinstance(optimizer, dict):
            if ("generator" not in optimizer) or ("discriminator" not in optimizer):
                raise ValueError("optimizer passed as dictionary must contain both  "
                                 "`generator` and `discriminator` keys")
        else:
            raise ValueError("could not interpret optimizer: {}".format(optimizer))
        if optimizer_kwargs is None:
            optimizer_kwargs = {"generator": {}, "discriminator": {}}
        elif isinstance(optimizer_kwargs, dict):
            if ("generator" not in optimizer_kwargs) and ("discriminator" not in optimizer_kwargs):
                optimizer_kwargs = {"generator": optimizer_kwargs, "discriminator": optimizer_kwargs}
            elif ("generator" not in optimizer_kwargs) or ("discriminator" not in optimizer_kwargs):
                raise ValueError("optimizer_kwargs passed as dictionary must contain both  "
                                 "`generator` and `discriminator` keys")
        resolved = {}
        for key in ["generator", "discriminator"]:
            identifier = optimizer[key]
            config = optimizer_kwargs[key]
            if isinstance(identifier, Optimizer):
                resolved[key] = identifier
            elif isinstance(identifier, dict):
                identifier.update(kwargs)
                resolved[key] = tf.keras.optimizers.get(identifier)
            elif isinstance(identifier, str):
                identifier = {"class_name": str(identifier), "config": config}
                resolved[key] = tf.keras.optimizers.get(identifier)
            else:
                raise ValueError("could not interpret identifier: {}".format(identifier))
        generator_optimizer = resolved["generator"]
        discriminator_optimizer = resolved["discriminator"]
        return generator_optimizer, discriminator_optimizer
                        
    
    @tf.function
    def D_loss(self, real_output, fake_output):
        """Compute discriminator loss."""
        real_loss = self.loss_function(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_function(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss
    
    @tf.function
    def G_loss(self, fake_output):
        """Compute generator loss."""
        return self.loss_function(tf.ones_like(fake_output), fake_output)
    
    @tf.function
    def train_step_1v1(self, x_real):
        """Training step for one epoch with 1 generator step and 1 discriminator step
        """
        z = tf.random.normal(self.z_batch_shape)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            x_fake_ = self.G(z, training=True)
            x_fake = tf.reshape(x_fake_, tf.shape(x_real))
            real_output = self.D(x_real, training=True)
            fake_output = self.D(x_fake, training=True)
            gen_loss = self.G_loss(fake_output)
            disc_loss = self.D_loss(real_output, fake_output)
        grad_gen = gen_tape.gradient(gen_loss, self.G.trainable_variables)
        grad_disc = disc_tape.gradient(disc_loss, self.D.trainable_variables)
        self.G_optimizer.apply_gradients(zip(grad_gen, self.G.trainable_variables))
        self.D_optimizer.apply_gradients(zip(grad_disc, self.D.trainable_variables))  
        return gen_loss, disc_loss
    
    @tf.function 
    def G_step(self):
        """Perform one training step for generator"""
        # using Gaussian noise with mean 0 and width 1 as generator input
        z = tf.random.normal(self.z_batch_shape)
        with tf.GradientTape() as t:
            x_fake = self.G(z, training=True)
            fake_output = self.D(x_fake, training=True)
            loss = self.G_loss(fake_output)
        grad = t.gradient(loss, self.G.trainable_variables)
        self.G_optimizer.apply_gradients(zip(grad, self.G.trainable_variables))
        return loss
    
    @tf.function
    def D_step(self, x_real):
        """Perform one training step for discriminator
        
            Arguments:
                x_real: Numpy array, tf.Tensor / any inputs accepted by tensorflow
                    Input from the real data.
        """
        # using Gaussian noise with mean 0 and width 1 as generator input
        z = tf.random.normal(self.z_batch_shape)
        with tf.GradientTape() as t:
            x_fake_ = self.G(z, training=True)
            x_fake = tf.reshape(x_fake_, tf.shape(x_real))
            real_output = self.D(x_real, training=True)
            fake_output = self.D(x_fake, training=True)
            cost = self.D_loss(real_output, fake_output)
        grad = t.gradient(cost, self.D.trainable_variables)
        self.D_optimizer.apply_gradients(zip(grad, self.D.trainable_variables))
        return cost    
    
    @tf.function
    def train_step_nv1(self, x_real):
        """Training step for one epoch with 1 generator step and n_disc discriminator step"""
        for i in range(self.n_disc):
            x_real_batch = tf.gather(x_real, i)
            d_loss = self.D_step(x_real_batch)
        g_loss = self.G_step()
        return g_loss, d_loss  
    
    def generate_samples(self, batch_size:int, shape:Optional[Tuple[int]]=None):
        """Generates sample using random inputs
        
            Arguments:
                batch_size: int
                    Number of samples to generate.
                shape: (Optional) tuple of int
                    Reshape the output to the given shape.
        """
        z_batch_shape = (batch_size,) + self.latent_dim
        z = tf.random.normal(z_batch_shape)
        samples = self.G(z, training=False)  
        if shape is not None:
            shape = (batch_size,) + shape
            samples = tf.reshape(samples, shape)
        return samples
    
    def predict(self, x):
        """Get predicted output from discriminator
        
            Arguments:
                x: Numpy array, tf.Tensor / any inputs accepted by tensorflow
                    Input data.
            Returns:
                tf.Tensor representing the predicted output from discriminator
        """
        return self.D(x, training=False)

    def train(self, x):
        """Train a GAN model
        
            Arguments:
                x: Numpy array, tf.Tensor / any inputs accepted by tensorflow
                    Input data.        
        """
        self._train_preprocess()
        dataset = self.prepare_dataset(x, batch_size=self.batch_size * self.n_disc,
                                       seed=self.random_state)
        g_metric = tf.keras.metrics.Mean()
        d_metric = tf.keras.metrics.Mean()
        
        self.g_loss_arr = []
        self.d_loss_arr = []
        self.epoch_arr = []
        
        input_shape = x.shape[1:]
        if self.n_disc == 1:
            train_step = self.train_step_1v1
            input_batch_shape = (self.batch_size,) + input_shape
        else:
            train_step = self.train_step_nv1
            input_batch_shape = (self.n_disc, self.batch_size) + input_shape

        for epoch in range(self.epochs):
            for step, x_batch_train_ in enumerate(dataset):
                # for restoring dataset at checkpoint
                if self.start_epoch > epoch:
                    continue
                x_batch_train = tf.reshape(x_batch_train_, input_batch_shape)
                gen_loss, disc_loss = train_step(x_batch_train)
                g_metric(gen_loss)
                d_metric(disc_loss)
                
            self.g_loss_arr.append(g_metric.result().numpy())
            self.d_loss_arr.append(d_metric.result().numpy())
            self.epoch_arr.append(epoch)
            
            self._train_post_epoch(epoch)
            g_metric.reset_states()
            d_metric.reset_states()
        
        self._train_postprocess()
        
    def _train_post_epoch(self, epoch:int, *args, **kwargs):
        super()._train_post_epoch(epoch)
        if self.visualization_interval and ((epoch + 1) % self.visualization_interval == 0):
            self.display_loss_and_image(self.g_loss_arr, self.d_loss_arr, self.epoch_arr)
        
    def enable_visualization(self, image_shape:Tuple[int], n_image:int=16, interval:int=1):
        """Enable visualization of loss curve and generated images
        
            Arguments:
                image_shape: tuple of int
                    Dimensions of the image of the form (rows, cols).
                n_image: int, default=16
                    Number of images to show.
                interval: int, default=1
                    Number of epochs between each update.
        """
        self.image_shape = image_shape
        self.visualization_interval = interval
        self.n_image_to_show = n_image
    
    def disable_visualization(self):
        self.visualization_interval = None
    
    def display_loss_and_image(self, g_loss, d_loss, epochs):
        if clear_output is not None:
            clear_output(wait=True)
        fig = plt.figure(figsize=(16,9))
        size = max(self.n_image_to_show, 1)
        rows = ( size // 4 ) + 1
        gs = gridspec.GridSpec(ncols=8, nrows=rows, figure=fig)
        epoch = epochs[-1]
        # plot loss curve
        ax_loss = plt.subplot(gs[:,:4])
        ax_loss.set_xlim(0, 1.1*epoch)
        ax_loss.plot(epochs, g_loss, label="Generator")
        ax_loss.plot(epochs, d_loss, label="Discriminator")
        ax_loss.set_xlabel('Epoch', fontsize=20)
        ax_loss.set_ylabel('Loss', fontsize=20)
        ax_loss.grid(True)
        ax_loss.legend(fontsize=15)        
        if (self.image_shape is not None):
            images = self.generate_samples(self.n_image_to_show, self.image_shape)
            for i in range(images.shape[0]):
                ax = plt.subplot(gs[i//4, 4 + i%4])
                plt.imshow(images[i])
        if self.checkpoint_dir:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            image_path = os.path.join(self.checkpoint_dir, 'image_at_epoch_{:04d}.png'.format(epoch))
            plt.savefig(image_path)
        plt.show()