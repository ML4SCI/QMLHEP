import numpy as np 
import sympy as sp
import tensorflow as tf
import tensorflow_quantum as tfq
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from .swap_test_utils import quantum_state_overlap

class EntangledQGAN():

  """
  Model for Entangled Quantum Generative Adversarial Networks
  Inspiration: https://arxiv.org/abs/2105.00080
  
  """

  def __init__(self, generator_model, discriminator_model,
               use_sampled=False,backend=None,name='QGAN_Model'):
    
    self.d_loss = []
    self.g_loss = []
    self.param_history = []
    self.state_overlap_l = []
    self.generator_model = generator_model
    self.discriminator_model = discriminator_model
    self.use_sampled = use_sampled
  
  def train(self,real_data_inputs, generator_data_inputs, batch_size, g_epochs, d_epochs, n_episodes):
    """
    Function for training the model
    
    Arguments: real_data_inputs, generator_data_inputs(random_data(size=real data)), batch_size, 
                g_epochs = number of generator steps in one train step
                d_epochs = number of discriminator steps in one train step
    
    Returns: g_loss(generator loss), d_loss(discriminator loss), 
             param_history(values of swap test parameters for each step(will remain same if set to be not trainable))
             state_overlap_l(overlap of two states(real and fake) at each step)

    """
    
    for episode in range(n_episodes):
      self.param_history.append(self.discriminator_model.get_layer('Swap_Test_Layer').get_weights())
      d_history = self.discriminator_model.fit(x=[real_data_inputs,generator_data_inputs],
                                          y=[tf.zeros_like(real_data_inputs,dtype=tf.dtypes.float32),
                                             tf.zeros_like(real_data_inputs,dtype=tf.dtypes.float32),
                                             tf.zeros_like(real_data_inputs,dtype=tf.dtypes.float32)],
                                             epochs=d_epochs,batch_size=batch_size,verbose=0)
      self.d_loss.append(d_history.history['loss'])
      updated_fidelity_test_params = np.array(self.discriminator_model.get_layer('Swap_Test_Layer').get_weights())
      self.generator_model.get_layer('Swap_Test_Layer').set_weights(np.reshape(updated_fidelity_test_params,(-1,8)))

      g_history = self.generator_model.fit(x=[real_data_inputs,generator_data_inputs],
                                      y=[tf.zeros_like(real_data_inputs,dtype=tf.dtypes.float32),
                                        tf.zeros_like(real_data_inputs,dtype=tf.dtypes.float32)],
                                        epochs=g_epochs,batch_size=batch_size,verbose=0)
      self.g_loss.append(g_history.history['loss'])

      state_overlap = quantum_state_overlap(self.generator_model,real_data_inputs,generator_data_inputs)
      self.state_overlap_l += [state_overlap]
      print(f'Step = {episode}')
      print(f'discriminator_loss={self.d_loss[-1]}')
      print(f'generator_loss={self.g_loss[-1]}')
      print(f'overlap={state_overlap}')
      print('-'*50) 

    return self.g_loss,self.d_loss,self.param_history,self.state_overlap_l   
  
  def plot_loss(gen_loss,disc_loss,epochs):
    """
    Function for plotting loss of discriminator and generator
    
    Arguments: gen_loss(generator loss), disc_loss(discriminator loss), epochs(list of range=number of train steps)

    """
    
    fig = plt.figure(figsize=(16,9))
    gs = gridspec.GridSpec(ncols=8, nrows=8, figure=fig)
    epochs = [i for i in range(epochs)]
    epoch = epochs[-1]
    # plot loss curve
    ax_loss = plt.subplot(gs[:,:4])
    ax_loss.set_xlim(0, 1.1*epoch)
    ax_loss.plot(epochs, np.mean(gen_loss,axis=1), label="Generator")
    ax_loss.plot(epochs, np.mean(disc_loss,axis=1), label="Discriminator")
    ax_loss.set_xlabel('Epoch', fontsize=20)
    ax_loss.set_ylabel('Loss', fontsize=20)
    ax_loss.grid(True)
    ax_loss.legend(fontsize=15)

  def create_images(self,real_data,random_data):
    """
    Function for generating images in inference mode

    Arguments: real_data(real data)
              random_data(random data)

    Returns: X_pca_inv_transform(final images similar to real images)

    """
    intermediate_output = self.generator_model.get_layer('Swap_Test_Layer').input[1]
    generator_model_1 = tf.keras.models.Model(inputs=[self.generator_model.input],outputs=[intermediate_output])
    print("Generating samples")
    samples = generator_model_1.predict([real_data,random_data])
    print("Applying PCA to real data")
    flattened_real_data = real_data.reshape(-1,real_data.shape[1]*real_data.shape[2])
    pca = PCA(n_components=4)
    pca.fit(flattened_real_data)
    print("Applying inverse PCA to generated samples")
    X_pca_inv_transform = pca.inverse_transform([tf.keras.backend.flatten(samples_) for samples_ in samples])
    X_pca_inv_transform = np.reshape(X_pca_inv_transform,real_data.shape)
    print("Generated Images:")
    fig = plt.figure(figsize=(26,18))
    gs = gridspec.GridSpec(ncols=8, nrows=8, figure=fig)
    for i in range(8):
      ax = plt.subplot(gs[i//4, 4 + i%4])
      plt.imshow(X_pca_inv_transform[i])
    
    return X_pca_inv_transform