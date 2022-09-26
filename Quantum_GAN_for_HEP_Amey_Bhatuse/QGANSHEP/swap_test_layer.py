import numpy as np 
import sympy as sp
import tensorflow as tf
import cirq
from  .swap_test_utils import data_encoding_circuit, variational_swap_test_circuit, swap_test_op
import tensorflow_quantum as tfq


class SwapTestLayer(tf.keras.layers.Layer):

    """
    Layer outputs the fidelity between two states created using two data entries 
    """

    def __init__(self,swap_test_symbol_values,real_data_encoding_circuit=None,gen_data_encoding_circuit=None,use_sampled=False,name='Swap_Test_Layer'):
        super(SwapTestLayer,self).__init__(name=name)
        self.qubits = cirq.GridQubit.rect(8,1)
        self.input_symbols = sp.symbols('i_:'+str(swap_test_symbol_values.shape[0]))
        self.swap_test_symbols = sp.symbols('x_:'+str(len(self.qubits)/2)+'_:'+str(2))
        if real_data_encoding_circuit is not None:
            self.real_data_encoding_circuit, self.real_input_symbols = real_data_encoding_circuit
        else:
            self.real_data_encoding_circuit, self.real_input_symbols = data_encoding_circuit(self.qubits[:4],self.input_symbols[:4])
        if gen_data_encoding_circuit is not None:
            self.gen_data_encoding_circuit, self.gen_input_symbols = gen_data_encoding_circuit
        else:
            self.gen_data_encoding_circuit, self.gen_input_symbols = data_encoding_circuit(self.qubits[4:],self.input_symbols[4:])      
        self.fidelity_circuit,self.param_symbols = variational_swap_test_circuit(self.qubits[:4],self.qubits[4:],self.swap_test_symbols)
        self.param_symbols_values = swap_test_symbol_values
        self.operators = swap_test_op(self.qubits[:4],self.qubits[4:])
        self.use_sampled = use_sampled
        self.main_name = name
   
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'name':self.main_name,
            'param_symbol_names':np.reshape(self.param_symbols,(-1,2))
        })
        return config
  
    def build(self,input_shape):
        self.parameters = tf.Variable(self.param_symbols_values,
                                  trainable=True,
                                  dtype = tf.dtypes.float32)
        self.full_circuit = tfq.layers.AddCircuit()(self.real_data_encoding_circuit,append=self.fidelity_circuit)
        self.full_circuit = tfq.layers.AddCircuit()(self.gen_data_encoding_circuit,append=self.full_circuit)
        self.symbol_names = tfq.util.get_circuit_symbols(tfq.from_tensor(self.full_circuit)[0])
    
    def call(self, inputs):
        real_data_inputs = inputs[0]
        generated_data_inputs = inputs[1]
        batch_size = tf.shape(real_data_inputs)[0]
        full_circuit_batch = tf.repeat(self.full_circuit,repeats=batch_size,name=self.main_name+'-tiled_fidelity_circuits') 
        tiled_parameters = tf.tile(tf.expand_dims(self.parameters,0),multiples=[batch_size,1])
        joined_parameters = tf.concat([real_data_inputs,generated_data_inputs],axis=-1)
        joined_parameters = tf.concat([joined_parameters,tiled_parameters],axis=-1)
        if not self.use_sampled:
            return tfq.layers.Expectation()(full_circuit_batch,
                                      symbol_names=self.symbol_names,
                                      operators=self.operators,
                                      symbol_values=joined_parameters)  
        else:
            return tfq.layers.SampledExpectation()(full_circuit_batch,
                                      symbol_names=self.symbol_names,
                                      operators=self.operators,
                                      symbol_values=joined_parameters,
                                      repetitions=1000)               