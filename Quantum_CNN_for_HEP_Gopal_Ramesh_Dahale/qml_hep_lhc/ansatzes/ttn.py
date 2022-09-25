import cirq
import sympy as sp
import numpy as np
import warnings


class TTN:
    """
	Ansatz based on
	"""
    def __init__(self) -> None:
        super().__init__()

    def _block(self, qubits, symbols):

        assert len(qubits) == 2
        assert len(qubits) == len(symbols)

        return cirq.Circuit(
            cirq.Y(qubits[0])**symbols[0],
            cirq.Y(qubits[1])**symbols[1], cirq.CNOT(qubits[0], qubits[1]))

    def _compute_indices(self, qubits, n_block_qubits):
        n_qubits = len(qubits)

        if n_block_qubits % 2 != 0:
            raise ValueError(
                f"n_block_qubits must be an even integer; got {n_block_qubits}"
            )

        if n_block_qubits < 2:
            raise ValueError(
                f"number of qubits in each block must be larger than or equal to 2; got n_block_qubits = {n_block_qubits}"
            )

        if n_block_qubits > n_qubits:
            raise ValueError(
                f"n_block_qubits must be smaller than or equal to the number of qubits; "
                f"got n_block_qubits = {n_block_qubits} and number of qubits = {n_qubits}"
            )

        if not np.log2(n_qubits / n_block_qubits).is_integer():
            warnings.warn(
                f"The number of qubits should be n_block_qubits times 2^n; got n_qubits/n_block_qubits = {n_qubits/n_block_qubits}"
            )

        n_qubits = 2**(int(np.log2(
            len(qubits) / n_block_qubits))) * n_block_qubits
        n_layers = int(np.log2(n_qubits // n_block_qubits)) + 1

        layers = [[
            qubits[i] for i in range(
                x + 2**(j - 1) * n_block_qubits // 2 - n_block_qubits // 2,
                x + n_block_qubits // 2 + 2**(j - 1) * n_block_qubits // 2 -
                n_block_qubits // 2,
            )
        ] + [
            qubits[i] for i in range(
                x + 2**(j - 1) * n_block_qubits // 2 +
                2**(j - 1) * n_block_qubits // 2 - n_block_qubits // 2,
                x + 2**(j - 1) * n_block_qubits // 2 + n_block_qubits // 2 +
                2**(j - 1) * n_block_qubits // 2 - n_block_qubits // 2,
            )
        ] for j in range(1, n_layers + 1)
                  for x in range(0, n_qubits -
                                 n_block_qubits // 2, 2**(j - 1) *
                                 n_block_qubits)]

        return layers

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
        n_block_qubits = 2
        n_params_block = 2
        ind_gates = self._compute_indices(qubits, n_block_qubits)
        n_blocks = int(2**int(np.log2(n_qubits / n_block_qubits)) * 2 - 1)
        assert len(ind_gates) == n_blocks

        # Sympy symbols for variational angles
        var_symbols = sp.symbols(f'θ0:{n_layers*n_params_block*n_blocks}')
        var_symbols = np.asarray(var_symbols).reshape(
            (n_layers, n_blocks, n_params_block))

        data_symbols = []

        circuit = cirq.Circuit()
        for l in range(n_layers):
            circuit += cirq.Circuit([
                self._block(w, var_symbols[l, idx])
                for idx, w in enumerate(ind_gates)
            ])
            # Re-encoding layer
            if drc and (l < n_layers - 1):
                data_circuit, expr = cirq.flatten(
                    feature_map.build(qubits, in_symbols[l]))
                circuit += data_circuit
                data_symbols += list(expr.values())

        return circuit, data_symbols, list(var_symbols.flat), observable
