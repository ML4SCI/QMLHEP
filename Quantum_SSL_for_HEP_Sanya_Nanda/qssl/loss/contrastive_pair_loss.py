import tensorflow as tf
from qssl.config import Config

class Losses:
    @staticmethod
    def quantum_fidelity_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    @staticmethod
    def contrastive_pair_loss(margin=Config.MARGIN):
        def loss(y_true, dist):
            y_true = tf.cast(y_true, tf.float32)
            square_dist = tf.square(dist)
            margin_square = tf.square(tf.maximum(margin - dist, 0))
            return tf.reduce_mean(y_true * square_dist + (1 - y_true) * margin_square)
        return loss