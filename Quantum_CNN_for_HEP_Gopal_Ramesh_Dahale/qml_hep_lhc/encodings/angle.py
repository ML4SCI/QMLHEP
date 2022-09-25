import cirq
from qml_hep_lhc.utils import _import_class
import sympy as sp
import numpy as np


class AngleMap:
    def __init__(self, gate='rx'):
        valid_gates = ['rx', 'ry', 'rz']
        if gate not in valid_gates:
            raise ValueError('gate must be one of rx, ry, rz')
        self.gate = _import_class("cirq.{}".format(gate))

    def build(self, qubits, symbols):
        num_in_symbols = len(symbols)
        symbols = np.asarray(symbols).reshape((num_in_symbols))
        e_ops = [
            self.gate(sp.pi * symbols[index])(bit)
            for index, bit in enumerate(qubits)
        ]
        return e_ops
