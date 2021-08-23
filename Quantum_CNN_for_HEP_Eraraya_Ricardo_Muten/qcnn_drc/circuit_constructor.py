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
import sympy
import numpy as np


def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]


def entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops


def generate_circuit(qubits, n_layers, input_size, use_entanglement=True, use_terminal_entanglement=True):
    """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
    # Number of qubits
    n_qubits = len(qubits)

    # Number of padding
    # Zero-pad the inputs and params if it is not a multiple of 3
    padding = (3 - (input_size % 3)) % 3

    # Sympy symbols for weights and bias parameters
    params = sympy.symbols(f'theta(0:{(input_size + padding) * n_layers * n_qubits})')
    params = np.asarray(params).reshape((n_layers, n_qubits, (input_size + padding)))

    # Define circuit
    circuit = cirq.Circuit()
    for l in range(n_layers):
        for gate in range(int(np.ceil(input_size/3))):
            # Variational layer
            circuit += cirq.Circuit(
                one_qubit_rotation(q, params[l, i, gate * 3:(gate + 1) * 3]) for i, q in enumerate(qubits))

        # Entangling layer
        if n_qubits >= 2 and (l != (n_layers - 1) or n_layers == 1) and use_entanglement:
            circuit += entangling_layer(qubits)
        if n_qubits >= 2 and use_terminal_entanglement and (l == (n_layers - 1) and n_layers != 1):
            circuit += entangling_layer(qubits)

    return circuit, list(params.flat)
