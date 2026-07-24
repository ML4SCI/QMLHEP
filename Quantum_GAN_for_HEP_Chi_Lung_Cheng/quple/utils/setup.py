import numpy as np
import random

def set_global_random_seed(seed:int=None, tf_seed=True):
    """Set global random seed for various python modules such as random, numpy and tensorflow
    """
    random.seed(seed)
    np.random.seed(seed)
    if tf_seed:
        import tensorflow as tf
        tf.random.set_seed(seed)