"""
Implementations of Quantum Autoencoders with tensorflow-quantum and cirq
"""

from itertools import product

import cirq
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow.keras.models import Model


class QAE_model(Model):
    """large QAE model with encoder and decoder
    """
    def __init__(self, data_qbits, latent_qbits, layers):
        """Create large QAE

        Args:
            data_qbits (int): number of qbits to upload data and use as encoder
            latent_qbits (int): number of latent qbits
            layers (int): number of layers to use for the pqc of encoder and decoder
        """
        super(QAE_model, self).__init__()
        self.latent_qbits = latent_qbits
        self.data_qbits = data_qbits
        self.num_layers = layers

        self.parameters = 0
        network_qbits = data_qbits + (data_qbits - latent_qbits)
        qbits = [cirq.GridQubit(0, i) for i in range(network_qbits + 1 + data_qbits)]

        self.model_circuit = self._build_circuit(qbits[:network_qbits], qbits[network_qbits:-1], data_qbits, latent_qbits, qbits[-1], self.num_layers)
        readout_operator = [cirq.Z(qbits[-1])]

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            tfq.layers.PQC(self.model_circuit, readout_operator),
        ])

    def _layer(self, qbits):
        """creates one layer with a fully entagled pqc

        Args:
            qbits (list): list of qbits to use for the layer

        Returns:
            cirq circuit
        """
        circ = cirq.Circuit()
        for i in range(len(qbits)):
            circ += cirq.ry(sympy.symbols(f"q{self.parameters}")).on(qbits[i])
            self.parameters += 1
        for i in range(len(qbits)):
            for j in range(i+1, len(qbits)):
                circ += cirq.CNOT(qbits[i], qbits[j])
        return circ


    def _build_circuit(self, network_qbits, reference_qbits, num_data_qbits, num_latent_qbits, swap_qbit, layers):
        """build the QAE circuit with encoder, decoder and swap test

        Args:
            network_qbits (list): list of qubits to use for encoder and decoder together
            reference_qbits (list): list of qubits to use as reference qbits
            num_data_qbits (int): the number of qubits to use for data
            num_latent_qbits (int): the number of qubits to use as latent space
            swap_qbit (int): the index of the swap qbit
            layers (int): number of layers for encoder and decoder

        Returns:
            cirq circuit
        """
        c = cirq.Circuit()
        for i in range(layers):
            c += self._layer(network_qbits[:num_data_qbits])
        for i in range(layers):
            c += self._layer(network_qbits[num_data_qbits - num_latent_qbits:])
        c += cirq.H(swap_qbit)
        for i in range(num_data_qbits):
            c += cirq.ControlledGate(sub_gate=cirq.SWAP, num_controls=1).on(swap_qbit, reference_qbits[i], network_qbits[num_data_qbits - num_latent_qbits:][i])
        c += cirq.H(swap_qbit)
        return c

    def call(self, x):
        return self.model(x)



class SQAE_model(Model):
    """SQAE model
    """
    def __init__(self, data_qbits, latent_qbits, layers):
        """Create SQAE

        Args:
            data_qbits (int): number of qbits to upload data and use as encoder
            latent_qbits (int): number of latent qbits
            layers (int): number of layers to use for the pqc of encoder and decoder
        """
        super(SQAE_model, self).__init__()
        self.latent_qbits = latent_qbits
        self.data_qbits = data_qbits
        self.num_layers = layers

        self.parameters = 0
        non_latent = data_qbits - latent_qbits
        qbits = [cirq.GridQubit(0, i) for i in range(data_qbits + non_latent + 1)]

        self.model_circuit = self._build_circuit(qbits[:data_qbits], qbits[data_qbits:-1], latent_qbits, qbits[-1], self.num_layers)
        readout_operator = [cirq.Z(qbits[-1])]

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            tfq.layers.PQC(self.model_circuit, readout_operator),
        ])

    def _layer(self, qbits):
        """creates one layer with a fully entagled pqc

        Args:
            qbits (list): list of qbits to use for the layer

        Returns:
            cirq circuit
        """
        circ = cirq.Circuit()
        for i in range(len(qbits)):
            circ += cirq.ry(sympy.symbols(f"q{self.parameters}")).on(qbits[i])
            self.parameters += 1
        for i in range(len(qbits)):
            for j in range(i+1, len(qbits)):
                circ += cirq.CNOT(qbits[i], qbits[j])
        return circ


    def _build_circuit(self, data_qbits, trash_qbits, num_latent_qbits, swap_qbit, layers):
        """build the SQAE circuit

        Args:
            data_qbits (list): list of qubits to use for encoder
            trash_qbits (list): list of qubits to use as trash states
            num_latent_qbits (int): the number of qubits to use as latent space
            swap_qbit (int): the index of the swap qbit
            layers (int): number of layers for encoder and decoder

        Returns:
            cirq circuit
        """
        c = cirq.Circuit()
        #encoder
        for i in range(layers):
            c += self._layer(data_qbits)
        c += cirq.H(swap_qbit)
        for i in range(len(trash_qbits)):
            c += cirq.ControlledGate(sub_gate=cirq.SWAP, num_controls=1).on(swap_qbit, trash_qbits[i], data_qbits[num_latent_qbits:][i])
        c += cirq.H(swap_qbit)
        return c

    def call(self, x):
        return self.model(x)
