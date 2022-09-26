import numpy as np 
import sympy as sp
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
from .preprocess_utils import inputs_preprocess, get_output_shape
from .demo_circuits import pqc_circuit_for_conv

class QConv2D_layer(tf.keras.layers.Layer):

    """
    Layer performs the convolutions similar to classical convolutional layers with parameterised quantum circuits as kernel
    
    """

    def __init__(self,circuit_layers,filters,filter_shape,stride,seed,conv_circuit=None,parameter_sharing=True,padding='same',conv_id='',name='Quantum_Convolutional_Layer_with_padding'):
        super(QConv2D_layer,self).__init__(name=name+conv_id)
        self.layers = circuit_layers
        self.filters = filters
        self.parameter_sharing = parameter_sharing
        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding
        self.main_name = name
        self.qubits = cirq.GridQubit.rect(1, filter_shape[0]*filter_shape[1])
        self.observables = tfq.convert_to_tensor([cirq.Z(self.qubits[-1])])
        if conv_circuit is None:
            self.circuit, self.input_symbols, self.param_symbols = pqc_circuit_for_conv(self.qubits,layers=self.layers)
        else:
            self.circuit, self.input_symbols, self.param_symbols = conv_circuit
        self.model_circuit = tfq.convert_to_tensor([self.circuit])
        self.all_symbols = np.concatenate((self.input_symbols,self.param_symbols),axis=0)
        self.initializer = tf.keras.initializers.RandomUniform(0, 2 * np.pi, seed=seed)


    def build(self,input_shape):
        if len(input_shape) == 3:
            self.input_rows = input_shape[1]
            self.input_cols = input_shape[2]
            self.input_channels = 1
        else:
            self.input_rows = input_shape[1]
            self.input_cols = input_shape[2]
            self.input_channels = input_shape[3]
        output_shape = get_output_shape(input_shape[1:3], self.filter_shape, self.stride, self.padding)
        self.output_rows = output_shape[0]
        self.output_cols = output_shape[1]
        if self.parameter_sharing:
            self.kernel_shape = tf.TensorShape([self.filters, self.input_channels, len(self.param_symbols)])
        else:
            self.kernel_shape = tf.TensorShape([self.filters, self.input_channels, 
                                               self.output_rows,
                                               self.output_cols,
                                               len(self.param_symbols)])
        self.symbol_names = tfq.util.get_circuit_symbols(tfq.from_tensor(self.model_circuit)[0])
        self.kernel = self.add_weight(
                        name='kernel',
                        shape=self.kernel_shape,
                        initializer=self.initializer,
                        trainable=True,
                        dtype=self.dtype)
        self.inputs_preprocess_ = inputs_preprocess


    """ 
    reference : https://gitlab.cern.ch/clcheng/quple/-/blob/master/quple/interface/tfq/layers/qconv2d.py  

    """  
    def call(self,inputs):
        batchsize = tf.gather(tf.shape(inputs), 0)
        depth = self.input_channels
        rows = self.output_rows
        cols = self.output_cols

        input_patches = self.inputs_preprocess_(inputs,self.filter_shape,self.stride,self.input_rows,self.input_cols,self.input_channels,padding=self.padding)
        inputs = tf.reshape(input_patches, [batchsize, depth, 
                                      self.output_rows, 
                                      self.output_cols,
                                      len(self.input_symbols)])

        # change to (depth, batchsize, rows, cols, symbols)
        inputs = tf.transpose(inputs, [1, 0, 2, 3, 4])
        # total number of circuit = filters*depth*batchsize*rows*cols
        circuit_size = tf.reduce_prod([self.filters, batchsize, depth, rows, cols])
        # tile inputs to (filters, depth, batchsize, rows, cols, symbols)
        tiled_up_inputs = tf.tile([inputs], [self.filters, 1, 1, 1, 1, 1])
        # reshape inputs to (circuit_size, symbols)
        tiled_up_inputs = tf.reshape(tiled_up_inputs, (circuit_size, tf.shape(tiled_up_inputs)[-1]))


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
        

        # tiled_up_data_circuit = tf.tile(self._data_circuit, [circuit_size])
        tiled_up_circuits = tf.tile(self.model_circuit, [circuit_size])
        tiled_up_operators = tf.tile([self.observables], [circuit_size, 1])
    

        result = tfq.layers.Expectation()(tiled_up_circuits,
                                    symbol_names=self.symbol_names,
                                    symbol_values=tiled_up_parameters,
                                    operators=tiled_up_operators)

        reshaped_output = tf.reshape(result,(self.filters, self.input_channels, batchsize, self.output_rows, self.output_cols))
        summed_output = tf.reduce_mean(reshaped_output, axis=1)
        final_output = tf.transpose(summed_output, [1, 2, 3, 0])
        return tf.reshape(final_output, (batchsize, self.output_rows, self.output_cols, self.filters))