import cirq
from quple.components.gate_ops import RXX, RYY, RZZ

class CirqInterface(ABC):
    __GATE_MAPPING__ = {
        "H": cirq.H, # Hadamard gate
        "I": cirq.I,  # one-qubit Identity gate
        "S": cirq.S, # Clifford S gate
        "T": cirq.T, # non-Clifford T gate
        'X': cirq.X, # Pauli-X gate
        "Y": cirq.Y, # Pauli-Y gate
        "Z": cirq.Z, # Pauli-Z gate
        "PauliX": cirq.X, # Pauli-X gate
        "PauliY": cirq.Y, # Pauli-Y gate
        "PauliZ": cirq.Z, # Pauli-Z gate
        "CX": cirq.CX, # Controlled-NOT gate
        "CNOT": cirq.CNOT, # Controlled-NOT gate
        "CZ": cirq.CZ, # Controlled-Z gate
        "XX": cirq.XX, # tensor product of two X gates (X parity gate)
        "YY": cirq.YY, # tensor product of two Y gates (Y parity gate)
        "ZZ": cirq.ZZ, # tensor product of two Z gates (Z parity gate)
        "XPowGate": cirq.XPowGate, # rotation along X axis with extra phase factor
        "YPowGate": cirq.YPowGate, # rotation along Y axis with extra phase factor 
        "ZPowGate": cirq.ZPowGate, # rotation along Z axis with extra phase factor
        "XXPowGate": cirq.XXPowGate, # X parity gate raised to some power
        "YYPowGate": cirq.YYPowGate, # Y parity gate raised to some power
        "ZZPowGate": cirq.ZZPowGate, # Z parity gate raised to some power
        "MS": cirq.ms, # Mølmer–Sørensen gate == RXX (A rotation around the XX axis in the two-qubit bloch sphere)
        "RXX": RXX, # XX Ising coupling gate (A rotation around the XX axis in the two-qubit bloch sphere)
        "RXX": RYY, # YY Ising coupling gate (A rotation around the YY axis in the two-qubit bloch sphere)
        "RXX": RZZ, # ZZ Ising coupling gate (A rotation around the ZZ axis in the two-qubit bloch sphere)                        
        "RX": cirq.rx, # rotation along X axis
        "RY": cirq.ry, # rotation along Y axis
        "RZ": cirq.rz, # rotation along Z axis
        "CCNOT": cirq.CCNOT, # Toffoli gate
        "CCX": cirq.CCX, # Toffoli gate
        "Toffoli": cirq.TOFFOLI, # Toffoli gate
        "SWAP": cirq.SWAP, # SWAP gate
        "CSWAP": cirq.CSWAP, # Controlled SWAP gate
        "ISWAP": cirq.ISWAP, # ISWAP gate
        "RISWAP": cirq.riswap, #Rotation ISWAP gate (X⊗X + Y⊗Y)
        "FSim": cirq.FSimGate, # Fermionic simulation gate
        "Fredkin": cirq.FREDKIN, # Controlled SWAP gate
        "CXPowGate": cirq.CXPowGate, # Controlled Power of an X gate
        "CZPowGate": cirq.CZPowGate, # Controlled Power of an Z gate
        "CNOTPowGate": cirq.CXPowGate, # Controlled Power of an X gate
    }