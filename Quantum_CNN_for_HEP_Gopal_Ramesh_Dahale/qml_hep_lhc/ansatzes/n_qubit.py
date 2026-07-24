import cirq


class NQubit:
    def __init__(self) -> None:
        super().__init__()

    def __single_qubit_rot(self, qubit, symbols, sparse):
        if sparse:
            return [
                cirq.Z(qubit)**symbols[0],
                cirq.Y(qubit)**symbols[1],
                cirq.Z(qubit)**symbols[2]
            ]
        return [[
            cirq.Z(qubit)**symbols[i],
            cirq.Y(qubit)**symbols[i + 1],
            cirq.Z(qubit)**symbols[i + 2]
        ] for i in range(0, len(symbols), 3)]

    def build(self, qubits, n_layers, sparse, in_symbols=None):
        observable = [cirq.Z(qubits[-1])]

        circuit = cirq.Circuit()
        for l in range(n_layers):
            circuit += cirq.Circuit(
                self.__single_qubit_rot(q, in_symbols[l, i], sparse)
                for i, q in enumerate(qubits))

            # Alternate CZ entangling circuit
            if (l & 1):
                circuit += [
                    cirq.CZ(q0, q1)
                    for q0, q1 in zip(qubits[1::2], qubits[2::2] + [qubits[0]])
                ]
            else:
                circuit += [
                    cirq.CZ(q0, q1)
                    for q0, q1 in zip(qubits[0::2], qubits[1::2])
                ]

        return circuit, observable
