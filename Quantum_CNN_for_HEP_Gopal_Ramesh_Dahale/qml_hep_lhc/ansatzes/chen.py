from .utils import cnot_entangling_circuit, one_qubit_unitary
import cirq
import sympy as sp
import numpy as np


class Chen:
    """
	Ansatz based on
	
	S. Y. C. Chen, T. C. Wei, C.Zhang, H. Yu and S. Yoo, 
	Quantum convolutional neural networks for high energy 
	physics data analysis, Phys. Rev. Res. \textbf{4} (2022) no.1, 013231
	doi:10.1103/PhysRevResearch.4.013231 
	"""

    def __init__(self) -> None:
        super().__init__()

    def build(self, qubits, feature_map, n_layers, drc, in_symbols=None):
        """
		Builds the circuit for the Chen ansatz.
		
		Args:
			qubits: the qubits to use
			feature_map: the feature map to use.
			n_layers: number of layers in the circuit
			drc: boolean, whether to use the re-encoding layer
			in_symbols: the input symbols to the circuit.
		
		Returns:
			The circuit, the symbols, and the observable.
				"""
        # Observables
        observable = [cirq.Z(qubits[-1])]

        n_qubits = len(qubits)
        # Sympy symbols for variational angles
        var_symbols = sp.symbols(f'θ0:{3*n_qubits*n_layers}')
        var_symbols = np.asarray(var_symbols).reshape((n_layers, n_qubits, 3))

        data_symbols = []

        circuit = cirq.Circuit()
        for l in range(n_layers):
            circuit += cnot_entangling_circuit(qubits)
            circuit += cirq.Circuit([
                one_qubit_unitary(q, var_symbols[l, i])
                for i, q in enumerate(qubits)
            ])
            # Re-encoding layer
            if drc and (l < n_layers - 1):
                data_circuit, expr = cirq.flatten(
                    feature_map.build(qubits, in_symbols[l]))
                circuit += data_circuit
                data_symbols += list(expr.values())

        return circuit, data_symbols, list(var_symbols.flat), observable
