import cirq
import sympy as sp
import numpy as np


class Farhi:
    """
    Ansatz based on
    
    Farhi, Edward and Hartmut Neven. 
    “Classification with Quantum Neural Networks on Near Term Processors.” 
    arXiv: Quantum Physics (2018): n. pag.
    """
    def __init__(self) -> None:
        super().__init__()

    def _generate_gates(self):
        """
        > The function `_generate_gates` returns a list of gates to be applied to the qubits in the
        layer
        
        Returns:
          The gates that will be used in the circuit.
        """
        if self.layer_type == 'alt-xz':
            return [cirq.XX, cirq.ZZ]
        if self.layer_type == 'all-x':
            return [cirq.XX, cirq.XX]
        if self.layer_type == 'all-z':
            return [cirq.ZZ, cirq.ZZ]

    def build(self, qubits, feature_map, n_layers, drc, in_symbols=None):
        """
        Builds the circuit for the Farhi et al ansatz.
        
        Args:
          qubits: the qubits we're using
          feature_map: the feature map to use.
          n_layers: number of layers in the circuit
          drc: whether to use the re-encoding layer
          in_symbols: The input symbols to the circuit.
        
        Returns:
          The circuit, the symbols, and the observable.
        """
        n_qubits = len(qubits)
        readout = cirq.GridQubit(-1, -1)
        self.layer_type = 'alt-xz'

        # Observables
        observable = [cirq.Z(readout)]

        # Sympy symbols for variational angles
        var_symbols = sp.symbols(f'θ0:{n_qubits*n_layers}')
        var_symbols = np.asarray(var_symbols).reshape((n_qubits, n_layers))

        circuit = cirq.Circuit()
        gates = self._generate_gates()
        num_gates = len(gates)

        data_symbols = []

        # Prepare the readout qubit
        circuit.append(cirq.X(readout))
        circuit.append(cirq.H(readout))

        for l in range(n_layers):
            for index, qubit in enumerate(qubits):
                circuit.append(gates[l % num_gates](
                    qubit, readout)**var_symbols[index, l])
            # Re-encoding layer
            if drc and (l < n_layers - 1):
                data_circuit, expr = cirq.flatten(
                    feature_map.build(qubits, in_symbols[l]))
                circuit += data_circuit
                data_symbols += list(expr.values())

        # Finally, prepare the readout qubit.
        circuit.append(cirq.H(readout))

        return circuit, data_symbols, list(var_symbols.flat), observable
