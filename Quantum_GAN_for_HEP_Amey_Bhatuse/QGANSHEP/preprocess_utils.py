import numpy as np 
import sympy as sp
import tensorflow as tf

def inputs_preprocess(inputs,filter_shape,stride,input_rows,input_cols,input_channels,padding="same"):

    """
    Processes the input images and returns patches from input
    images according to filter size and stride

    """
    
    kernel_size = (1, 1) + filter_shape + (1,)
    strides = (1, 1) + stride + (1,)
    padding = padding.upper()
    batchsize = lambda x: tf.gather(tf.shape(x), 0)

    # planes = number of channels
    planes = input_channels
    rows = input_rows
    cols = input_cols
    depth = 1
    reshaped_input_ = lambda x: tf.reshape(x, shape=(batchsize(x), rows, cols, planes))

    # change input order to (batchsize, depth, rows, cols)
    transposed_input = lambda x: tf.transpose(reshaped_input_(x), [0, 3, 1, 2])
    reshaped_input = lambda x: tf.reshape(transposed_input(x), 
                                              shape=(batchsize(x), planes, rows, cols, depth))

    # get patches from input images to encode in the quantum circuits
    input_patches = lambda x: tf.extract_volume_patches(reshaped_input(x),
                                            ksizes=kernel_size, strides=strides, padding=padding)
    return input_patches(inputs) 
    
def get_output_shape(input_shape,filter_shape,stride,padding='same'):

    """
    Returns:
      output shape for given input shape  
      
    """

    if (input_shape[0] % stride[0] == 0):
        pad_along_height = max(filter_shape[0] - stride[0], 0)
    else:
        pad_along_height = max(filter_shape[0] - (input_shape[0] % stride[0]), 0)
    if (input_shape[1] % stride[1] == 0):
        pad_along_width = max(filter_shape[1] - stride[1], 0)
    else:
        pad_along_width = max(filter_shape[1] - (input_shape[1] % stride[1]), 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    paddings = tf.constant([[pad_top,pad_bottom],[pad_left,pad_right]])
    rows = input_shape[0]+paddings[0][0]+paddings[0][1]
    cols = input_shape[1]+paddings[1][0]+paddings[1][1]
    padded_shape = tf.TensorShape([rows,cols])
    new_rows = np.ceil(float(padded_shape[0] - filter_shape[0] + 1) / float(stride[0]))
    new_cols = np.ceil(float(padded_shape[1] - filter_shape[1] + 1) / float(stride[1]))
    return tf.TensorShape([int(new_rows), int(new_cols)])

def crop_images(data,dimensions):
    """
    Arguments: 
       data(ndarray) - input images/data)
       dimensions(tuple) - required dimension of images 

    Returns: 
       cropped images/data
    """
    img_size = dimensions[0]
    max_val_pix = np.argmax(np.mean(data[:, :, :], axis=0))
    center = [int(max_val_pix/data.shape[1]), max_val_pix%data.shape[1]]
    data = data[:, (center[0]-int(img_size/2)):(center[0]+int(img_size/2)), (center[1]-int(img_size/2)):(center[1]+int(img_size/2))]
    return data
