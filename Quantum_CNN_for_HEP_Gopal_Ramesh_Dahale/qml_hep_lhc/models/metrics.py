import tensorflow as tf


# Custom accuracy metric for quantum circuits
@tf.function
def custom_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    y_pred = tf.map_fn(lambda x: 1.0 if x >= 0 else -1.0, y_pred)
    return tf.keras.backend.mean(tf.keras.backend.equal(y_true, y_pred))


class qAUC(tf.keras.metrics.AUC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_ = tf.clip_by_value(y_pred, -1, 1)
        y_pred_ = (y_pred + 1) / 2
        y_true = (y_true + 1) / 2
        super().update_state(y_true, y_pred_, sample_weight=sample_weight)
