# MIT License
#
# Copyright (c) 2021 Eraraya Ricardo Muten
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
from qcnn_drc.circuit_constructor import generate_circuit


class ReUploadingPQC(tf.keras.layers.Layer):
    def __init__(self, n_qubits, n_layers, input_size, use_entanglement=True, use_terminal_entanglement=True,
                    observables=None, name="re-uploading_PQC"):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.input_size = input_size
        self.use_entanglement = use_entanglement
        self.use_terminal_entanglement = use_terminal_entanglement
        self.main_name = name

        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        if observables is None:
            self.observables = [cirq.Z(self.qubits[-1])]  # Measure only the last qubit
        else:
            self.observables = observables  # Custom observables

        # Generate the data re-uploading circuit
        self.circuit, theta_symbols = generate_circuit(self.qubits, self.n_layers, self.input_size,
                                                       use_entanglement=self.use_entanglement,
                                                       use_terminal_entanglement=self.use_terminal_entanglement)

        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols]
        self.indices = tf.constant([sorted(symbols).index(a) for a in symbols])

        # Thetas (bias) initialization
        thetas_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        self.thetas = tf.Variable(
            initial_value=thetas_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name=self.main_name + "-thetas"
        )

        # Weights initialization
        w_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name=self.main_name + "-weights"
        )

        # Dummy inputs initialization
        # Using the empty circuits as hacks for ControlledPQC
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])

        self.computation_layer = tfq.layers.ControlledPQC(self.circuit, self.observables)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'input_size': self.input_size,
            'use_entanglement': self.use_entanglement,
            'use_terminal_entanglement': self.use_terminal_entanglement,
            'name': self.main_name,
        })
        return config

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # Flatten inputs (from 2D images to 1D array)
        inputs_flattened = tf.keras.layers.Flatten()(inputs)

        # Pad the inputs if it is not a multiple of 3
        padding = 3 - inputs_flattened.shape[1] % 3
        if padding % 3 != 0:
            inputs_flattened = tf.pad(inputs_flattened, tf.constant([[0, 0, ], [0, padding]]))

        # Repeat the inputs for every layer and qubit
        inputs_flattened = tf.tile(inputs_flattened, tf.constant([1, self.n_layers * self.n_qubits]))

        # Weight the inputs (x' = wx)
        inputs_weighted = tf.math.multiply(self.w, inputs_flattened, name=self.main_name + '-weighted_inputs')
        # Bias the inputs (x' = x + b)
        inputs_weighted_biased = tf.math.add(inputs_weighted, self.thetas,
                                             name=self.main_name + '-weighted_inputs_plus_bias')

        # Duplicate dummy inputs
        # One for every batch
        empty_circuit_batch = tf.repeat(self.empty_circuit, repeats=batch_size,
                                        name=self.main_name + '-tiled_up_empty_circuits')

        joined_params = tf.gather(inputs_weighted_biased, self.indices, axis=-1, name=self.main_name + '-joined_params')

        return self.computation_layer([empty_circuit_batch, joined_params])
