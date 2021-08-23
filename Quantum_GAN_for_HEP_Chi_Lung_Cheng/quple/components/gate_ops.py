from typing import Sequence
import numpy as np

import cirq
from cirq import ops


def RXX(theta: float) -> ops.XXPowGate:
    """The XX Ising coupling gate, a native two-qubit operation in ion traps.

    A rotation around the XX axis in the two-qubit bloch sphere.

    The gate implements the following unitary:

        exp(-i θ XX) = [ cos(θ)   0        0       -isin(θ)]
                       [ 0        cos(θ)  -isin(θ)  0      ]
                       [ 0       -isin(θ)  cos(θ)   0      ]
                       [-isin(θ)  0        0        cos(θ) ]

    Args:
        theta: float, sympy.Basic
        	The rotation angle in radians.

    Returns:
        RXX gate with a rotation of `theta` angle.
    """
    return ops.XXPowGate(exponent=theta * 2 / np.pi, global_shift=-0.5)

def RYY(theta: float) -> ops.YYPowGate:
    """The YY Ising coupling gate

    A rotation around the YY axis in the two-qubit bloch sphere.

    The gate implements the following unitary:

        exp(-i θ YY) = [ cos(θ)   0        0       sin(θ)]
                       [ 0        cos(θ)  -isin(θ)  0      ]
                       [ 0       -isin(θ)  cos(θ)   0      ]
                       [sin(θ)  0        0        cos(θ) ]

    Args:
        theta: float, sympy.Basic
        	The rotation angle in radians.

    Returns:
        RYY gate with a rotation of `theta` angle.
    """
    return ops.YYPowGate(exponent=theta * 2 / np.pi, global_shift=-0.5)

def RZZ(theta: float) -> ops.ZZPowGate:
    """The ZZ Ising coupling gate

    A rotation around the ZZ axis in the two-qubit bloch sphere.

    The gate implements the following unitary:

        exp(-i θ ZZ) = [ exp(iθ/2)     0        0           0     ]
                       [    0      exp(-iθ/2)   0           0     ]
                       [    0          0     exp(-iθ/2)     0     ]
                       [    0          0        0       exp(iθ/2) ]

    Args:
        rads: float, sympy.Basic
        	The rotation angle in radians.

    Returns:
        RZZ gate with a rotation of `theta` angle.
    """
    return ops.ZZPowGate(exponent=theta * 2 / np.pi, global_shift=-0.5)    

class CompositeGate:
    pass


class PauliRotation(CompositeGate):
    def __init__(self, pauli_string, theta, global_shift=False):
        self.paulis = pauli_string[::-1]
        self.theta = theta
        self.indices = [i for i, pauli in enumerate(self.paulis) if pauli != 'I']
        self.global_shift = global_shift

    @staticmethod
    def change_basis(*qubits:Sequence[int],
                     pauli_string:Sequence[str], inverse=False) -> None:
        # do not change basis if only first order pauli operator
        if len(pauli_string) == 1:
            return
        operations = []
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X':
                operations.append(ops.H(qubits[i]))
            elif pauli == 'Y':
                if inverse:
                    operations.append(ops.rx(-np.pi / 2)(qubits[i]))
                else: 
                    operations.append(ops.rx(np.pi / 2)(qubits[i]))
        return operations 

    def __call__(self, *qubits):
        if not self.indices:
            return None

        operations = []

        operations += PauliRotation.change_basis(*qubits, pauli_string=self.paulis)

        qpairs = [(qubits[i], qubits[i+1]) for i in range(len(qubits)-1)]

        for qpair in qpairs:
            operations.append(cirq.CX(*qpair))
        
        # do not switch to RZ gate if only first order pauli operator
        if len(self.paulis) == 1:
            if self.global_shift:
                if self.paulis[0] == 'Z':
                    operations.append(ops.ZPowGate(self.theta)(qubits[-1]))
                elif self.paulis[0] == 'X':
                    operations.append(ops.XPowGate(self.theta)(qubits[-1]))
                elif self.paulis[0] == 'Y':
                    operations.append(ops.YPowGate(self.theta)(qubits[-1]))             
            else:
                if self.paulis[0] == 'Z':
                    operations.append(ops.rz(self.theta)(qubits[-1]))
                elif self.paulis[0] == 'X':
                    operations.append(ops.rx(self.theta)(qubits[-1]))
                elif self.paulis[0] == 'Y':
                    operations.append(ops.ry(self.theta)(qubits[-1]))    
        else:
            if self.global_shift:
                operations.append(ops.ZPowGate(self.theta)(qubits[-1]))
            else:
                operations.append(ops.rz(self.theta)(qubits[-1]))

        for qpair in qpairs[::-1]:
            operations.append(cirq.CX(*qpair))
        
        operations += PauliRotation.change_basis(*qubits, pauli_string=self.paulis, inverse=True)
        
        return operations