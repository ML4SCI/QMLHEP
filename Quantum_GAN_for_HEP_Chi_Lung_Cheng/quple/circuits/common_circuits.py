from quple import QuantumCircuit

def construct_bell_circuit():
    """
    Quantum circuit to create the Bell state:
    |\Phi ^{+}\rangle ={\frac  {1}{{\sqrt  {2}}}}(|0\rangle _{A}\otimes |0\rangle _{B}+|1\rangle _{A}\otimes |1\rangle _{B})
    """
    cq = QuantumCircuit(2, name='BellCircuit')
    cq.H(0)
    cq.CNOT((0,1))
    return cq

bell_circuit = construct_bell_circuit()
