import cirq
import sympy as sp
import numpy as np


class DoubleAngleMap:
    def __init__(self, activation='atan'):
        super().__init__()
        self.activation = getattr(sp, activation)

    def build(self, qubits, symbols):
        num_in_symbols = len(symbols)
        symbols = np.asarray(symbols).reshape((num_in_symbols))
        e_ops = [
            cirq.ry(sp.pi * self.activation(symbols[i]))(bit)
            for i, bit in enumerate(qubits)
        ]
        e_ops += [
            cirq.rz(sp.pi * self.activation(symbols[i]**2))(bit)
            for i, bit in enumerate(qubits)
        ]
        return e_ops
