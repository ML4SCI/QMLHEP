from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Layer
from tensorflow.keras.regularizers import l2


class BottleneckResidual(Layer):
    """
    Bottleneck Residual Layer for Resent model
    """
    def __init__(self,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

        super(BottleneckResidual, self).__init__()

        # Configuration
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.strides = strides
        self.batch_normalization = batch_normalization
        self.conv_first = conv_first

        # Layers
        self.conv = Conv2D(num_filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding='same',
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))

        self.batch_norm = BatchNormalization() if batch_normalization else None
        self.activation = Activation(
            activation) if activation is not None else None

    def call(self, input_tensor):
        """
        Forward pass
        
        Args:
          input_tensor: the input tensor
        
        Returns:
          The output of the convolutional layer.
        """
        x = input_tensor
        if self.conv_first:
            x = self.conv(x)
            if self.batch_normalization:
                x = self.batch_norm(x)
            if self.activation is not None:
                x = self.activation(x)
        else:
            if self.batch_normalization:
                x = self.batch_norm(x)
            if self.activation is not None:
                x = self.activation(x)
            x = self.conv(x)
        return x
