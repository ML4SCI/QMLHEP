from tensorflow.keras.layers import Layer, Flatten, Activation
from qml_hep_lhc.ansatzes.utils import cluster_state_circuit
import cirq
import sympy as sp
import numpy as np
from tensorflow import random_uniform_initializer, Variable, constant, repeat, tile, shape, gather, pad
import tensorflow_quantum as tfq
from qml_hep_lhc.ansatzes import NQubit
from tensorflow import multiply, add
import tensorflow as tf


class NQubitPQC(Layer):
    def __init__(self,
                 n_qubits,
                 cluster_state=False,
                 observable=None,
                 n_layers=1,
                 sparse=False,
                 name='NQubitPQC'):

        super(NQubitPQC, self).__init__(name=name)

        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.cluster_state = cluster_state
        self.observable = observable
        self.sparse = sparse

        # Prepare qubits
        self.qubits = cirq.GridQubit.rect(1, self.n_qubits)

    def build(self, input_shape):
        self.n_inputs = np.prod(input_shape[1:])

        # Make n_inputs a multiple of 3 greater than or equal to n_inputs
        if self.sparse is False:
            if self.n_inputs % 3 != 0:
                self.n_inputs += (3 - (self.n_inputs % 3))

        circuit = cirq.Circuit()

        if self.cluster_state:
            circuit += cluster_state_circuit(self.qubits)

        # Sympy symbols for (wx + b) input
        self.num_in_symbols = self.n_inputs * self.n_layers * self.n_qubits
        in_shape = (self.n_layers, self.n_qubits, self.n_inputs)
        num_weights = self.num_in_symbols
        num_biases = self.num_in_symbols

        if self.sparse:
            self.num_in_symbols = self.n_layers * self.n_qubits * 3
            num_weights *= 3
            in_shape = (self.n_layers, self.n_qubits, 3)
            num_biases = self.num_in_symbols

        in_symbols = sp.symbols(f'w0:{self.num_in_symbols}')
        self.in_symbols = np.asarray(in_symbols).reshape(in_shape)

        var_circuit, obs = NQubit().build(self.qubits, self.n_layers,
                                          self.sparse, self.in_symbols)

        if self.observable is None:
            self.observable = obs

        circuit += var_circuit

        self.in_symbols = list(self.in_symbols.flat)

        # Initalize variational angles
        w_init = random_uniform_initializer(minval=-1, maxval=1)
        self.qweights = Variable(initial_value=w_init(shape=(1, num_weights),
                                                      dtype="float32"),
                                 trainable=True,
                                 name=self.name + "_qweights")

        b_init = random_uniform_initializer(minval=-0.1, maxval=0.1)
        self.qbiases = Variable(initial_value=b_init(shape=(1, num_biases),
                                                     dtype="float32"),
                                trainable=True,
                                name=self.name + "_qbiases")

        # Align Left
        circuit = cirq.align_left(circuit)

        # Define explicit symbol order
        symbols = [str(symb) for symb in self.in_symbols]
        self.indices = constant([symbols.index(a) for a in sorted(symbols)])

        # Define computation layer
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(
            circuit, self.observable)

    def call(self, input_tensor):
        batch_dim = shape(input_tensor)[0]
        x = Flatten()(input_tensor)

        # Pad input_tensor if not a multiple of 3
        if self.sparse is False:
            if x.shape[1] % 3 != 0:
                x = pad(x, [[0, 0], [0, 3 - x.shape[1] % 3]])

        tiled_up_circuits = repeat(self.empty_circuit,
                                   repeats=batch_dim,
                                   name=self.name + "_tiled_up_circuits")

        if self.sparse is False:
            tiled_up_inputs = tile(
                x, multiples=[1, self.n_layers * self.n_qubits])
        else:
            tiled_up_inputs = tile(
                x, multiples=[1, self.n_layers * self.n_qubits * 3])

        # Multiply by weights
        tiled_up_inputs = multiply(tiled_up_inputs,
                                   self.qweights,
                                   name=self.name +
                                   "_tiled_up_inputs_qweights")

        if self.sparse is False:
            # Add biases
            tiled_up_inputs = add(tiled_up_inputs,
                                  self.qbiases,
                                  name=self.name +
                                  "_tiled_up_inputs_qweights_qbiases")
        else:
            # Reshape to (batch,n_layers*n_qubits,3, n_inputs)
            tiled_up_inputs = tf.reshape(
                tiled_up_inputs,
                [batch_dim, self.n_layers * self.n_qubits, 3, self.n_inputs],
                name=self.name + "_reshaped_inputs")
            # Sum over each layer and qubit (w1*x1 + w2*x2 + ...)
            # The new shape is (batch, n_layers*n_qubits)
            tiled_up_inputs = tf.reduce_sum(tiled_up_inputs,
                                            axis=-1,
                                            name=self.name +
                                            "_tiled_up_inputs_reduced_sum")
            # Add biases
            tiled_up_inputs = tf.reshape(tiled_up_inputs, [batch_dim, -1])
            tiled_up_inputs = add(tiled_up_inputs,
                                  self.qbiases,
                                  name=self.name +
                                  "_tiled_up_inputs_qweights_qbiases")

        joined_vars = gather(tiled_up_inputs,
                             self.indices,
                             axis=1,
                             name=self.name + "_joined_vars")

        return self.computation_layer([tiled_up_circuits, joined_vars])
