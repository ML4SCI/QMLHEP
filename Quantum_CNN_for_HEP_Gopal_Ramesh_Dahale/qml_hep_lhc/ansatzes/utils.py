import cirq


def one_qubit_unitary(qubit, symbols):
    """
		Make a Cirq circuit enacting a rotation of the bloch sphere about the X,
		Y and Z axis, that depends on the values in `symbols`.
		
		Args:
			qubit: The qubit to apply the unitary to.
			symbols: a list of 3 symbols, each of which is either 0 or 1.
		
		Returns:
			A circuit with a single qubit and three gates.
		"""
    return cirq.Circuit(
        cirq.X(qubit)**symbols[0],
        cirq.Y(qubit)**symbols[1],
        cirq.Z(qubit)**symbols[2])


def cz_entangling_circuit(qubits):
    """
		Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
		
		Args:
			qubits: The qubits to entangle.
		
		Returns:
			A list of CZ gates.
		"""
    if len(qubits) == 1:
        return []
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[-1], qubits[0])] if len(qubits) != 2 else [])
    return cz_ops


def cnot_entangling_circuit(qubits):
    """
		Returns a layer of CNOT entangling gates on `qubits` (arranged in a circular topology).
		
		Args:
			qubits: The qubits to entangle.
		
		Returns:
			A list of CNOT gates.
		"""
    if len(qubits) == 1:
        return []
    cnot_ops = [cirq.CNOT(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cnot_ops += ([cirq.CNOT(qubits[-1], qubits[0])]
                 if len(qubits) != 2 else [])
    return cnot_ops


def cluster_state_circuit(qubits):
    """
		Return a cluster state on the qubits in `qubits`
		
		Args:
				qubits: The qubits to use in the circuit.
		
		Returns:
				A circuit that creates a cluster state.
		"""
    ops = [cirq.H(q) for q in qubits]
    if len(qubits) == 1:
        return ops
    ops += [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    ops += ([cirq.CZ(qubits[-1], qubits[0])] if len(qubits) != 2 else [])
    return ops


def two_qubit_unitary(qubits, symbols):
    """
		Make a Cirq circuit that creates an arbitrary two qubit unitary.
		
		Args:
			qubits: a list of two qubits
			symbols: a list of 15 symbols, each of which is a float between 0 and 2pi.
		
		Returns:
			A circuit with a two qubit unitary.
		"""

    circuit = cirq.Circuit()
    circuit += one_qubit_unitary(qubits[0], symbols[0:3])
    circuit += one_qubit_unitary(qubits[1], symbols[3:6])
    circuit += [cirq.ZZ(*qubits)**symbols[6]]
    circuit += [cirq.YY(*qubits)**symbols[7]]
    circuit += [cirq.XX(*qubits)**symbols[8]]
    circuit += one_qubit_unitary(qubits[0], symbols[9:12])
    circuit += one_qubit_unitary(qubits[1], symbols[12:])
    return circuit


def two_qubit_pool(source_qubit, sink_qubit, symbols):
    """
		Make a Cirq circuit to do a parameterized 'pooling' operation, which
		attempts to reduce entanglement down from two qubits to just one.
		
		Args:
				source_qubit: the qubit that is being measured
				sink_qubit: the qubit that will be measured
				symbols: a list of 6 symbols, each of which is either 'X', 'Y', or 'Z'.
		
		Returns:
				A circuit that performs a two-qubit pooling operation.
	"""
    pool_circuit = cirq.Circuit()
    sink_basis_selector = one_qubit_unitary(sink_qubit, symbols[0:3])
    source_basis_selector = one_qubit_unitary(source_qubit, symbols[3:6])
    pool_circuit.append(sink_basis_selector)
    pool_circuit.append(source_basis_selector)
    pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
    pool_circuit.append(sink_basis_selector**-1)
    return pool_circuit
