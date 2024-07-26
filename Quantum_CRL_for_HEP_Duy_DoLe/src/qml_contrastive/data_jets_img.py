import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import h5py


f3 = h5py.File("../../data/quark-gluon/quark-gluon_train-set_n793900.hdf5","r")
f2 = h5py.File("../../data/quark-gluon/quark-gluon_test-set_n10000.hdf5","r")
f = h5py.File("../../data/quark-gluon/quark-gluon_test-set_n139306.hdf5","r")
x_train = f3.get('X_jets')
y_train = f3.get('y')

x_val = f2.get('X')
y_val = f2.get('y')

x_test = f2.get('X')
y_test = f2.get('y')
x_val_ones = x_val[y_val[()]==1]
x_val = x_val[y_val[()]==0]

div1 = np.max(x_val, axis=(1,2)).reshape((x_val.shape[0],1,1,3))
div1[div1 == 0] = 1
x_val = x_val / div1
div2 = np.max(x_val_ones, axis=(1,2)).reshape((x_val_ones.shape[0],1,1,3))
div2[div2 == 0] = 1
x_val_ones = x_val_ones / div2

x_test = x_val
x_test_ones = x_val_ones

def crop(x, channel, crop_fraction):
    return f.image.central_crop(x[:,:,:,channel].reshape(x.shape[0],125,125,1), crop_fraction)
def crop_and_resize(x, channel, scale, crop_fraction=0.8,meth="bilinear"):
    cropped = tf.image.central_crop(x[:,:,:,channel].reshape(x.shape[0],125,125,1), crop_fraction)
    return tf.image.resize(cropped, (scale,scale), method=meth).numpy()
def simple_resize(x, channel, scale, meth="bilinear"):
    return tf.image.resize(x[:,:,:,channel].reshape((x.shape[0],125,125,1)), (scale,scale), method=meth).numpy()
batch_size = 20
num_batches = x_train.shape[0]//batch_size

events = num_batches*batch_size
fnew = h5py.File("QG_train_normalized", "w")
dsetx = fnew.create_dataset("X", (events,125,125,3), dtype='f')
dsety = fnew.create_dataset("y", (events,), dtype='i')
 


for i in range(int(num_batches)):
    y = y_train[i * batch_size: (i + 1) * batch_size]
    x = x_train[i * batch_size: (i + 1) * batch_size]

    div1 = np.max(x, axis=(1,2)).reshape((x.shape[0],1,1,3))
    div1[div1 == 0] = 1
    x = x / div1

    dsety[i * batch_size: (i + 1) * batch_size] = y
    dsetx[i * batch_size: (i + 1) * batch_size] = x
    print("batch ",i,"/",num_batches, end="\r")
    
fnew.close()