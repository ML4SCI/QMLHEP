import cirq
from qml_hep_lhc.utils import _import_class
import sympy as sp
import numpy as np


class BasisMap:
    def __init__(self):
        super().__init__()

    def build(self, qubits, symbols):
        num_in_symbols = len(symbols)
        symbols = np.asarray(symbols).reshape((num_in_symbols))
        e_ops = [cirq.H(q) for q in qubits]
        e_ops += [
            cirq.Z(q)**(sp.GreaterThan(symbols[i], 0))
            for i, q in enumerate(qubits)
        ]

        e_ops += [cirq.H(q) for q in qubits]
        return e_ops
