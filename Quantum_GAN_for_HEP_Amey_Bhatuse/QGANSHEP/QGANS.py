import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class QGAN():

  """
  Class for creating the GANS model
  reference: https://gitlab.cern.ch/clcheng/quple/-/blob/master/quple/models/generative/qgan.py

  """

  def __init__(self,discriminator,generator,disc_optimizer,gen_optimizer,generator_loss='negative_binary_cross_entropy'):
    self.generator_model = generator
    self.discriminator_model = discriminator
    self.d_opt = disc_optimizer
    self.g_opt = gen_optimizer
    self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    if generator_loss=='negative_binary_cross_entropy':
        self.generator_loss = self.generator_loss_1
    if generator_loss=='inverted_label':
        self.generator_loss = self.generator_loss_2
    self.gen_loss_ = []
    self.disc_loss_ = []
    self.epochs_ = []

  def prepare_dataset(self,data,batch_size,seed=None,drop_remainder=True,buffer_size=100):
    buffer_size =len(data[0])
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.shuffle(buffer_size=buffer_size,seed=seed,reshuffle_each_iteration=True)
    ds = ds.batch(batch_size,drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

  def train_preprocess(self,random_state):
    tf.random.set_seed(random_state)
    np.random.seed(random_state)
  
  @tf.function
  def generator_loss_1(self,fake_output):
    return -self.loss(tf.zeros_like(fake_output),fake_output)

  @tf.function
  def generator_loss_2(self,fake_output):
    return self.loss(tf.ones_like(fake_output),fake_output)

  @tf.function
  def discriminator_loss(self,real_output,fake_output):
    real_loss = self.loss(tf.ones_like(real_output),real_output)
    fake_loss = self.loss(tf.zeros_like(fake_output),fake_output)
    return real_loss + fake_loss
  
  @tf.function
  def train_step_1v1(self,x_real,batch_size):
    """
    Training step for one epoch with 1 generator step and 1 discriminator step

    """

    fake_data_shape = (batch_size,) + self.generator_model.input_shape[1:]
    z = tf.random.normal(shape=fake_data_shape)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      x_fake_ = self.generator_model(z, training=True)
      real_output = self.discriminator_model(x_real, training=True)
      fake_output = self.discriminator_model(x_fake_, training=True)
      gen_loss = self.generator_loss(fake_output)
      disc_loss = self.discriminator_loss(real_output, fake_output)
    grad_gen = gen_tape.gradient(gen_loss, self.generator_model.trainable_variables)
    grad_disc = disc_tape.gradient(disc_loss, self.discriminator_model.trainable_variables)
    self.g_opt.apply_gradients(zip(grad_gen, self.generator_model.trainable_variables))
    self.d_opt.apply_gradients(zip(grad_disc, self.discriminator_model.trainable_variables))  
    return gen_loss, disc_loss

  @tf.function
  def train_step_nv1(self,x_real,n_disc,batch_size):
    """
    Training step for one epoch with n generator steps and 1 discriminator step

    """

    for i in range(n_disc):
      x_real_batch = tf.gather(x_real,i)
      d_loss = self.discriminator_step(x_real_batch,batch_size)
    g_loss = self.generator_step(batch_size)
    return g_loss, d_loss

  @tf.function
  def train_step_1vn(self,x_real,n_gen,batch_size):
    """
    Training step for one epoch with 1 generator step and n discriminator steps

    """

    for i in range(n_gen):
      g_loss = self.generator_step(batch_size)
    d_loss = self.discriminator_step(x_real,batch_size)
    return g_loss, d_loss 

  @tf.function
  def discriminator_step(self,x_real,batch_size):
    fake_data_shape = (batch_size,) + self.generator_model.input_shape[1:]
    z = tf.random.normal(shape=fake_data_shape)
    with tf.GradientTape() as gradient_tape:
      real_output = self.discriminator_model(x_real,training = True)
      fake_input = self.generator_model(z, training = True) 
      fake_output = self.discriminator_model(fake_input,training = True)
      cost = self.discriminator_loss(real_output,fake_output)
    grad = gradient_tape.gradient(cost,self.discriminator_model.trainable_variables)
    self.d_opt.apply_gradients(zip(grad,self.discriminator_model.trainable_variables))
    return cost

  @tf.function
  def generator_step(self,batch_size):
    fake_data_shape = (batch_size,) + self.generator_model.input_shape[1:]
    z = tf.random.normal(shape=fake_data_shape)
    with tf.GradientTape() as gradient_tape:
      fake_input = self.generator_model(z,training=True)
      fake_output = self.discriminator_model(fake_input,training= True) #
      loss = self.generator_loss(fake_output)
    grad = gradient_tape.gradient(loss,self.generator_model.trainable_variables)
    self.g_opt.apply_gradients(zip(grad,self.generator_model.trainable_variables))
    return loss
  
  def train_qgans(self,x,epochs,batch_size,seed=1024,n_disc=1,n_gen=1):
    """
    Function for training the model

    Arguments: x(real data), epochs(total train steps), batch_size, seed(random seed),
               n_disc = number of discriminator steps in one train step
               n_gen = number of generator steps in one train step

    Returns: gen_loss_(generator loss), disc_loss_(discriminator loss), epochs_(list with range=number of train steps)

    """

    input_shape = x.shape[1:]
    self.train_preprocess(seed)
    data = self.prepare_dataset(data=x,batch_size=batch_size*n_disc,seed=seed)
    g_metric = tf.keras.metrics.Mean()
    d_metric = tf.keras.metrics.Mean()
    for epoch in range(epochs):
      for step,training_batch_data_ in enumerate(data):
        if( n_disc == 1 and n_gen == 1):
          input_batch_shape = (batch_size,) + input_shape
          training_batch_data = tf.reshape(training_batch_data_,input_batch_shape)
          gen_loss,disc_loss = self.train_step_1v1(x_real=training_batch_data, batch_size=batch_size)
        if n_disc > 1 and n_gen == 1:
          input_batch_shape = (n_disc, batch_size) + input_shape
          training_batch_data = tf.reshape(training_batch_data_,input_batch_shape)        
          gen_loss,disc_loss = self.train_step_nv1(x_real=training_batch_data, batch_size=batch_size,n_disc=n_disc)
        if n_gen > 1 and n_disc == 1:
          input_batch_shape = (batch_size,) + input_shape
          training_batch_data = tf.reshape(training_batch_data_,input_batch_shape)
          gen_loss,disc_loss = self.train_step_1vn(x_real=training_batch_data, batch_size=batch_size,n_gen=n_gen)
        g_metric(gen_loss)
        d_metric(disc_loss)
      self.gen_loss_.append(g_metric.result().numpy())
      self.disc_loss_.append(d_metric.result().numpy())
      self.epochs_.append(epoch)
      print("Epoch:{} ;   generator_loss:{} ;   discriminator_loss:{}".format(epoch,g_metric.result().numpy(),d_metric.result().numpy()))
  
      g_metric.reset_state()
      d_metric.reset_state()
    return self.gen_loss_,self.disc_loss_,self.epochs_

  def create_images(self,batch_size, shape=None):
    """ 
    Generates sample using random inputs
        
    Arguments:
            batch_size (int): Number of samples to generate.
            shape (Optional) (tuple of int): Reshape the output to the given shape.
    
    Returns: samples(images similar to real images)

    """

    print("Generating random data...")    
    z_batch_shape = (batch_size,) + self.generator_model.input_shape[1:]
    z = tf.random.normal(z_batch_shape)
    print("Fetching images from generator...")
    samples = self.generator_model(z,training = False)
    print("Generated Images:")
    fig = plt.figure(figsize=(26,18))
    gs = gridspec.GridSpec(ncols=8, nrows=8, figure=fig)
    for i in range(8):
      ax = plt.subplot(gs[i//4, 4 + i%4])
      plt.imshow(samples[i,:,:,0])

    return samples
  
  def plot_loss(self,gen_loss,disc_loss,epochs):
    """
    Function for plotting loss of discriminator and generator
    
    Arguments: gen_loss(generator loss), disc_loss(discriminator loss), epochs(list of range=number of train steps)

    """
    fig = plt.figure(figsize=(16,9))
    gs = gridspec.GridSpec(ncols=8, nrows=8, figure=fig)
    epoch = epochs[-1]
    # plot loss curve
    ax_loss = plt.subplot(gs[:,:4])
    ax_loss.set_xlim(0, 1.1*epoch)
    ax_loss.plot(epochs, gen_loss, label="Generator")
    ax_loss.plot(epochs, disc_loss, label="Discriminator")
    ax_loss.set_xlabel('Epoch', fontsize=20)
    ax_loss.set_ylabel('Loss', fontsize=20)
    ax_loss.grid(True)
    ax_loss.legend(fontsize=15)

  def predict(self, x):
    """
    Function for predicting the truth value of samples, i.e. classifying into real and fake classes

    """
    return self.discriminator_model(x, training=False)