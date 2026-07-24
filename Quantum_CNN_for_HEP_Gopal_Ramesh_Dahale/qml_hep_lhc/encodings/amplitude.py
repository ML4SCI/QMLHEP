from logging import warning
import cirq
import sympy as sp
import numpy as np
import warnings


class AmplitudeMap:
    def __init__(self):
        super().__init__()
        warnings.warn(
            "AmplitudeMap currently does not normalize the input unless padding is needed.\nUser must manually normalize the input."
        )

    def _beta(self, s, j, x):
        index_num = (2 * j - 1) * (2**(s - 1))
        index_den = (j - 1) * (2**s)

        num_start = index_num
        num_end = index_num + 2**(s - 1)

        den_start = index_den
        den_end = index_den + 2**(s)

        if ((num_start >= num_end) or (den_start >= den_end)):
            return 0 * x[num_start]

        num = (np.sum(np.abs(x[num_start:num_end])**2))**0.5
        den = (np.sum(np.abs(x[den_start:den_end])**2))**0.5

        beta = 2 * sp.asin(num / den)
        return beta

    def _locate_x(self, curr_j, prev_j, length):
        curr_bin = bin(curr_j)[2:].zfill(length)
        prev_bin = bin(prev_j)[2:].zfill(length)
        return [
            i for i, (x, y) in enumerate(zip(curr_bin, prev_bin)) if x != y
        ]

    def build(self, qubits, symbols):
        num_in_symbols = len(symbols)
        symbols = np.asarray(symbols).reshape((num_in_symbols))

        n = len(qubits)
        ae_ops = []
        count = 0
        ae_ops += [cirq.ry(self._beta(n, 1, symbols))(qubits[0])]
        count += 1
        for i in range(1, n):
            # We can have at most i control bits
            # Total possibilities is therefore 2^i
            controls = 2**i

            control_qubits = [qubits[c] for c in range(i + 1)]
            ae_ops += [
                cirq.ControlledGate(
                    sub_gate=cirq.ry(self._beta(n - i, controls, symbols)),
                    num_controls=len(control_qubits) - 1)(*control_qubits)
            ]
            count += 1
            for j in range(1, controls):
                for loc in self._locate_x(controls - j - 1, controls - j, i):
                    ae_ops += [cirq.X(qubits[loc])]

                ae_ops += [
                    cirq.ControlledGate(sub_gate=cirq.ry(
                        self._beta(n - i, controls - j, symbols)),
                                        num_controls=len(control_qubits) -
                                        1)(*control_qubits)
                ]
                count += 1

            for k in range(i):
                ae_ops += [cirq.X(qubits[k])]

        return ae_ops
