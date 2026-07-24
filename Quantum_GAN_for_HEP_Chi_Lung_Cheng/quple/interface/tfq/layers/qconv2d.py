import numpy as np

import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import constraints, initializers, regularizers

from quple.interface.tfq.layers import PQC
from quple.interface.tfq.tf_utils import get_output_shape

class QConv2D(PQC):
    """Quantum Convolution 2D Layer.
    Quantum convolution uses a quantum filter as the basic building block. 
    It replaces the classical filter by a Parameterised Quantum Circuit (PQC) 
    which scan across the local regions (the receptive field) of an image. 
    A quantum filter transforms the pixel values in the local region into the 
    quantum states of its data circuit via a suitable feature map, hence projecting
    the data into a higher dimensional quantum (Hilbert) space. The quantum states 
    are then propagated to the kernel circuit of the quantum filter which undergoes a 
    sequence parameterised gate operations and outputs the expectation value of a 
    given measurement operator.
    """
    def __init__(self, kernel_circuit,
                 data_circuit,
                 operators,
                 kernel_size,
                 filters=1,
                 strides=1,
                 padding='same',
                 initializer=None,
                 regularizer=None,
                 constraint=None,
                 parameter_sharing=True,
                 trainable=True,       
                 seed=None,
                 name=None, **kwargs):
        """Instantiate this layer.
        
        Arguments:
            kernel_circuit: `cirq.Circuit` or `quple.QuantumCircuit` containing 
                `sympy.Symbols` that will be used as the model which will be fed with 
                quantum data from the data circuit.
            data_circuit: `cirq.Circuit` or `quple.QuantumCircuit` containing 
                `sympy.Symbols` that maps the input patches in this layer to the quantum
                data of the circuit which will be fed to the model circuit.
            operators: `cirq.PauliSum` or Python `list` of `cirq.PauliSum` objects
                used as observables at the end of the kernel circuit.
            filters: Integer, the dimensionality of the output space (i.e. the number
              of filters in the convolution).
            kernel_size: An integer or tuple/list of n integers, specifying the
              length of the convolution window.
            strides: An integer or tuple/list of n integers,
              specifying the stride length of the convolution.
              Specifying any stride value != 1 is incompatible with specifying
              any `dilation_rate` value != 1.
            padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
              `"valid"` means no padding. `"same"` results in padding with zeros
              evenly to the left/right or up/down of the input such that output has the
              same height/width dimension as the input. `"causal"` results in causal
              (dilated) convolutions, e.g. `output[t]` does not depend on `input[t+1:]`.
            initializer: Optional `tf.keras.initializer` object to specify how the
                symbols in `kernel_circuit` should be initialized when creating
                the managed variables. If None, defaults to initialize from a
                random uniform distribution between 0 and 2*pi. 
            regularizer: Optional `tf.keras.regularizer` object applied to the
                managed variables parameterizing `kernel_circuit`.
            constraint: Optional `tf.keras.constraint` object applied to the
                managed variables parameterizing `kernel_circuit`.
            parameter_sharing: Boolean. Whether to share parameters for quantum filters
            across different local regions of the input data.
            trainable: Boolean, whether the layer's variables should be trainable.
            seed: Optional integer. Controls the random seed of the weight initializer.
            name: Optional string. Name of the layer.
        """
        self.rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.parameter_sharing = parameter_sharing
        super(QConv2D, self).__init__(kernel_circuit, data_circuit, operators,
                                      initializer=initializer,
                                      regularizer=regularizer,
                                      constraint=constraint,
                                      trainable=trainable, 
                                      seed=seed,
                                      name=name, **kwargs)
        
    def _validate_init(self):
        if not all(self.kernel_size):
            raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                           'Received: %s' % (self.kernel_size,))

        if not all(self.strides):
            raise ValueError('The argument `strides` cannot contains 0(s). '
                           'Received: %s' % (self.strides,))
        kernel_size = self.kernel_size[0]*self.kernel_size[1]
        num_input_symbols = len(self._base_input_symbols_list)
        if num_input_symbols != kernel_size:
            raise ValueError(f"The kernel size (={kernel_size}) must match the number of "
                             f"input symbols in the data circuit (={num_input_symbols})")
            
    def _validate_build(self, input_shape):
        pass
            
    def build(self, input_shape):
        """Keras build function."""
        assert len(input_shape) in [3, 4]
        if len(input_shape) == 3:
            self.input_rows = input_shape[1]
            self.input_cols = input_shape[2]
            self.input_channels = 1
        else:
            self.input_rows = input_shape[1]
            self.input_cols = input_shape[2]
            self.input_channels = input_shape[3]
        output_shape = get_output_shape(input_shape[1:3], self.kernel_size, self.strides, self.padding)
        self.output_rows = output_shape[0]
        self.output_cols = output_shape[1]
        
        # kernel shape = (filters, input_channels, circuit_params)
        if self.parameter_sharing:
            self.kernel_shape = tf.TensorShape([self.filters, self.input_channels, self._symbols.shape[0]])
        else:
            self.kernel_shape = tf.TensorShape([self.filters, self.input_channels, 
                                               self.output_rows,
                                               self.output_cols,
                                               self._symbols.shape[0]])

        self.kernel = self.add_weight(
                        name='kernel',
                        shape=self.kernel_shape,
                        initializer=self.initializer,
                        regularizer=self.regularizer,
                        constraint=self.constraint,
                        trainable=True,
                        dtype=self.dtype)

        super().build(input_shape)
            
    def _init_weights(self):
        """Initialize trainable weights
        """
        # do not initialize weight before building
        pass
    
    def _get_input_resolver(self):
        base_resolver = super()._get_input_resolver()
        kernel_size = (1, 1) + self.kernel_size + (1,)
        strides = (1, 1) + self.strides + (1,)
        padding = self.padding.upper()
        batchsize = lambda x: tf.gather(tf.shape(x), 0)
        # planes = number of channels
        planes = self.input_channels
        rows = self.input_rows
        cols = self.input_cols
        depth = 1
        reshaped_input_ = lambda x: tf.reshape(x, shape=(batchsize(x), rows, cols, planes))
        # change input order to (batchsize, depth, rows, cols)
        transposed_input = lambda x: tf.transpose(reshaped_input_(x), [0, 3, 1, 2])
        reshaped_input = lambda x: tf.reshape(transposed_input(x), 
                                              shape=(batchsize(x), planes, rows, cols, depth))
        input_patches = lambda x: tf.extract_volume_patches(reshaped_input(x),
            ksizes=kernel_size, strides=strides, padding=padding)
        resolved_input = lambda x: base_resolver(input_patches(x))
        return resolved_input
    
    def call(self, inputs):
        """Keras call function."""
        batchsize = tf.gather(tf.shape(inputs), 0)
        depth = self.input_channels
        rows = self.output_rows
        cols = self.output_cols
        resolved_inputs__ = self._input_resolver(inputs)
        resolved_inputs_ = tf.reshape(resolved_inputs__, [batchsize, depth, 
                                                          self.output_rows, 
                                                          self.output_cols,
                                                          self._num_formulas])
        # change to (depth, batchsize, rows, cols, symbols)
        resolved_inputs = tf.transpose(resolved_inputs_, [1, 0, 2, 3, 4])
        # total number of circuit = filters*depth*batchsize*rows*cols
        circuit_size = tf.reduce_prod([self.filters, batchsize, depth, rows, cols])
        # tile inputs to (filters, depth, batchsize, rows, cols, symbols)
        tiled_up_inputs_ = tf.tile([resolved_inputs], [self.filters, 1, 1, 1, 1, 1])
        # reshape inputs to (circuit_size, symbols)
        tiled_up_inputs = tf.reshape(tiled_up_inputs_, (circuit_size, tf.shape(tiled_up_inputs_)[-1]))
        if self.parameter_sharing:
            # tile size for weights = batchsize*rows*cols
            tile_size = tf.reduce_prod([batchsize, rows, cols])
            tiled_up_weights__ = tf.tile([self.kernel], [tile_size, 1, 1, 1])
            # change to (filters, depth, batchsize*rows*cols, weight_symbols)
            tiled_up_weights_ = tf.transpose(tiled_up_weights__, [1, 2, 0, 3])
        else:
            # tile size for weights = batchsize
            # weight now has shape (batchsize, filters, depth, rows, cols, weight_symbols)
            tiled_up_weights__ = tf.tile([self.kernel], [batchsize, 1, 1, 1, 1, 1])
            # change to (filters, depth, batchsize, rows, cols, weight_symbols)
            tiled_up_weights_ = tf.transpose(tiled_up_weights__, [1, 2, 0, 3, 4, 5])
        # reshape to (circuit_size, weight_symbols)
        tiled_up_weights = tf.reshape(tiled_up_weights_, (circuit_size, tf.shape(tiled_up_weights_)[-1]))
        tiled_up_parameters = tf.concat([tiled_up_inputs, tiled_up_weights], 1)
        
        tiled_up_data_circuit = tf.tile(self._data_circuit, [circuit_size])
        tiled_up_model = tf.tile(self._model_circuit, [circuit_size])
        model_appended = self._append_layer(tiled_up_data_circuit, append=tiled_up_model)
        tiled_up_operators = tf.tile(self._operators, [circuit_size, 1])
        # this is disabled to make autograph compilation easier.
        # pylint: disable=no-else-return
        if self._analytic:
            result = self._executor(model_appended,
                                   symbol_names=self._all_symbols,
                                   symbol_values=tiled_up_parameters,
                                   operators=tiled_up_operators)
        else:
            tiled_up_repetitions = tf.tile(self._repetitions,
                                           [circuit_batch_dim, 1])
            result =  self._executor(model_appended,
                                    symbol_names=self._all_symbols,
                                    symbol_values=tiled_up_parameters,
                                    operators=tiled_up_operators,
                                    repetitions=tiled_up_repetitions)
        reshaped_output = tf.reshape(result, 
                          (self.filters, self.input_channels, batchsize, self.output_rows, self.output_cols))
        summed_output = tf.reduce_mean(reshaped_output, axis=1)
        final_output = tf.transpose(summed_output, [1, 2, 3, 0])
        return tf.reshape(final_output, (batchsize, self.output_rows, self.output_cols, self.filters))
        # pylint: enable=no-else-return