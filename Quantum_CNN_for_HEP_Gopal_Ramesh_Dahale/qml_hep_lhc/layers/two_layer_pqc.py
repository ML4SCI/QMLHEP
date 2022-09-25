from tensorflow.keras.layers import Layer, Flatten
from qml_hep_lhc.ansatzes.utils import cluster_state_circuit
from qml_hep_lhc.encodings import AmplitudeMap
import cirq
import sympy as sp
import numpy as np
from tensorflow import random_uniform_initializer, Variable, constant, repeat, tile, shape, concat, gather, pad
from .utils import symbols_in_expr_map, resolve_formulas
import tensorflow_quantum as tfq
import tensorflow as tf


class TwoLayerPQC(Layer):
    def __init__(self,
                 n_qubits,
                 n_inputs,
                 feature_map,
                 ansatz,
                 cluster_state=False,
                 observable=None,
                 n_layers=1,
                 drc=False,
                 name='TwoLayerPQC'):

        super(TwoLayerPQC, self).__init__(name=name)

        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.cluster_state = cluster_state
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.drc = drc

        # Prepare qubits
        self.qubits = cirq.GridQubit.rect(1, self.n_qubits)

        # Observables
        self.observable = observable

        # Sympy symbols for encoding angles
        self.num_data_symbols = n_inputs
        self.num_in_symbols = self.num_data_symbols
        in_shape = (1, self.num_data_symbols)
        self.input_tile_size = 1

        if self.drc:
            in_shape = (self.n_layers, self.num_data_symbols)
            self.num_in_symbols *= self.n_layers
            self.input_tile_size = self.n_layers

        in_symbols = sp.symbols(f'x0:{self.num_in_symbols}')
        self.in_symbols = np.asarray(in_symbols).reshape(in_shape)

    def build(self, input_shape):
        # Define data and variational circuits
        data_circuit = cirq.Circuit()
        var_circuit = cirq.Circuit()

        # Prepare data circuit
        if self.cluster_state:
            data_circuit += cluster_state_circuit(self.qubits)

        data_circuit += self.feature_map.build(self.qubits, self.in_symbols[0])

        # Prepare model circuit
        if self.drc:
            var_circuit, data_symbols, var_symbols, obs = self.ansatz.build(
                self.qubits, self.feature_map, self.n_layers, self.drc,
                self.in_symbols[1:])
        else:
            var_circuit, data_symbols, var_symbols, obs = self.ansatz.build(
                self.qubits, self.feature_map, self.n_layers, self.drc)

        if self.observable is None:
            self.observable = obs

        # Initalize variational angles
        var_init = random_uniform_initializer(minval=-np.pi / 2,
                                              maxval=np.pi / 2)
        self.theta = Variable(initial_value=var_init(shape=(1,
                                                            len(var_symbols)),
                                                     dtype="float32"),
                              trainable=True,
                              name=self.name + "_thetas")

        # Flatten circuits
        data_circuit = cirq.align_left(data_circuit)
        var_circuit = cirq.align_left(var_circuit)
        data_circuit, expr_map = cirq.flatten(data_circuit)
        raw_in_symbols = symbols_in_expr_map(expr_map)
        data_expr = list(expr_map)
        data_expr_symbols = list(expr_map.values())

        # var_circuit = cirq.align_left(var_circuit)
        var_expr_symbols = var_symbols
        data_expr_symbols += data_symbols

        if not isinstance(self.feature_map, AmplitudeMap):
            lmbd_init = tf.ones(shape=(len(data_expr_symbols), ))
            self.lmbd = tf.Variable(initial_value=lmbd_init,
                                    dtype="float32",
                                    trainable=True,
                                    name=self.name + "_lambdas")

        # Non trainable symbols
        self.data_sym = Variable(initial_value=var_init(
            shape=(1, len(data_expr_symbols)), dtype="float32"),
                                 trainable=False,
                                 name=self.name + "_data_sym")

        # Define explicit symbol order and expression resolver
        symbols = [str(symb) for symb in var_expr_symbols + data_expr_symbols]
        self.indices = constant([symbols.index(a) for a in sorted(symbols)])
        self.input_resolver = resolve_formulas(data_expr, raw_in_symbols)

        # Define computation layer
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(
            data_circuit + var_circuit, self.observable)

    def call(self, input_tensor):
        batch_dim = shape(input_tensor)[0]
        x = Flatten()(input_tensor)
        # Pad input tensor to nearest power of 2 in case of amplitude encoding
        # Padded with one to avoid division by zero
        if isinstance(self.feature_map, AmplitudeMap):
            padding = self.num_data_symbols - x.shape[1]
            if padding:
                x = pad(x,
                        constant([[0, 0], [0, padding]]),
                        constant_values=1.0)
            x, _ = tf.linalg.normalize(x, axis=1)

        resolved_inputs = self.input_resolver(x)

        # Replace NaNs with zeros
        resolved_inputs = tf.where(tf.math.is_nan(resolved_inputs),
                                   tf.zeros_like(resolved_inputs),
                                   resolved_inputs)
        tiled_up_circuits = repeat(self.empty_circuit,
                                   repeats=batch_dim,
                                   name=self.name + "_tiled_up_circuits")
        tiled_up_thetas = tile(self.theta,
                               multiples=[batch_dim, 1],
                               name=self.name + "_tiled_up_thetas")
        tiled_up_inputs = tile(resolved_inputs,
                               multiples=[1, self.input_tile_size])

        if not isinstance(self.feature_map, AmplitudeMap):
            scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
            tiled_up_inputs = tf.keras.layers.Activation('tanh')(scaled_inputs)

        joined_vars = concat([tiled_up_thetas, tiled_up_inputs], axis=1)
        joined_vars = gather(joined_vars,
                             self.indices,
                             axis=1,
                             name=self.name + '_joined_vars')

        return self.computation_layer([tiled_up_circuits, joined_vars])
