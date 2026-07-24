from math import ceil
import tensorflow as tf

def get_paddings(input_shape, filter_shape, strides):
    """
    Arguments:
        input_shape:
            `(rows, cols)`
        filter_shape:
            `(rows, cols)`
        strides:
            `(rows, cols)`
    """
    assert len(input_shape) == 2
    out_height = ceil(float(input_shape[0]) / float(strides[0]))
    out_width  = ceil(float(input_shape[1]) / float(strides[1]))
    if (input_shape[0] % strides[0] == 0):
        pad_along_height = max(filter_shape[0] - strides[0], 0)
    else:
        pad_along_height = max(filter_shape[0] - (input_shape[0] % strides[0]), 0)
    if (input_shape[1] % strides[1] == 0):
        pad_along_width = max(filter_shape[1] - strides[1], 0)
    else:
        pad_along_width = max(filter_shape[1] - (input_shape[1] % strides[1]), 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return tf.constant([[pad_top, pad_bottom,], [pad_left, pad_right]])

def get_padded_shape(input_shape, filter_shape, strides, padding="same"):
    """
    Arguments:
        input_shape:
            `(rows, cols)`
        filter_shape:
            `(rows, cols)`
        strides:
            `(rows, cols)`
    """
    assert len(input_shape) == 2
    if padding.lower() == "same":
        paddings = get_paddings(input_shape, filter_shape, strides)
        rows = input_shape[0]+paddings[0][0]+paddings[0][1]
        cols = input_shape[1]+paddings[1][0]+paddings[1][1]
    elif padding.lower() == "valid":
        rows = input_shape[0]
        cols = input_shape[1]
    else:
        raise ValueError("invalid padding: {}".format(padding))
        
    return tf.TensorShape([rows, cols])

# reference: https://cs231n.github.io/convolutional-networks/
def get_output_shape(input_shape, filter_shape, strides, padding="same"):
    """
    Arguments:
        input_shape:
            `(rows, cols)`
        filter_shape:
            `(rows, cols)`
        strides:
            `(rows, cols)`
    """
    assert len(input_shape) == 2
    padded_shape = get_padded_shape(input_shape, filter_shape, strides, padding)
    new_rows = ceil(float(padded_shape[0] - filter_shape[0] + 1) / float(strides[0]))
    new_cols = ceil(float(padded_shape[1] - filter_shape[1] + 1) / float(strides[1]))
    return tf.TensorShape([new_rows, new_cols])