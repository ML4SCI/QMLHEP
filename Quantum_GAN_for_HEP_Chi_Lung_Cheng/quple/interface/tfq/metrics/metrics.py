import tensorflow as tf

class qAUC(tf.keras.metrics.AUC):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_ = (y_pred+1)/2
        super().update_state(y_true, y_pred_, sample_weight=sample_weight)