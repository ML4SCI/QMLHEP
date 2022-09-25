import cirq
import sympy as sp
import numpy as np
from .utils import two_qubit_pool, two_qubit_unitary

NUM_CONV_SYMBOLS = 15
NUM_POOL_SYMBOLS = 6


class Cong:
    """
    Ansatz based on

    Cong, I., Choi, S. & Lukin, M.D. Quantum convolutional neural networks. 
    Nat. Phys. 15, 1273–1278 (2019). https://doi.org/10.1038/s41567-019-0648-8
    """

    def __init__(self) -> None:
        super().__init__()

    def _quantum_conv_circuit(self, bits, symbols):
        """
        Quantum Convolution Layer. Return a Cirq circuit with the 
        cascade of `two_qubit_unitary` applied to all pairs of 
        qubits in `qubits`.
        
        Args:
            qubits: a list of qubits
            symbols: a list of symbols that will be used to represent the qubits.
        
        Returns:
            A circuit with the two qubit unitary applied to the first two qubits, 
            then the second two qubits, then the third two qubits, then the 
            first and last qubits.
        """
        circuit = cirq.Circuit()
        for first, second in zip(bits[0::2], bits[1::2]):
            circuit += two_qubit_unitary([first, second], symbols)
        for first, second in zip(bits[1::2], bits[2::2] + [bits[0]]):
            circuit += two_qubit_unitary([first, second], symbols)
        return circuit

    def _quantum_pool_circuit(self, source_bits, sink_bits, symbols):
        """
        A layer that specifies a quantum pooling operation.
        A Quantum pool tries to learn to pool the relevant information from two
        qubits onto 1.
        
        Args:
            source_bits: the qubits that will be used as the input to the pooling layer
            sink_bits: the qubits that will be measured at the end of the circuit
            symbols: a list of symbols that will be used to label the qubits in the circuit.
        
        Returns:
            A circuit with the two qubit pool gates applied to each pair of source and sink bits.
        """
        circuit = cirq.Circuit()
        for source, sink in zip(source_bits, sink_bits):
            circuit += two_qubit_pool(source, sink, symbols)
        return circuit

    def build(self, qubits, feature_map, n_layers, drc, in_symbols=None):
        """
        Builds the circuit for the Cong ansatz.
        
        Args:
          qubits: the qubits that the circuit will be built on
          feature_map: The feature map to use.
          n_layers: number of layers in the circuit
          drc: Whether to use DRC.
          in_symbols: The input symbols to the circuit.
        
        Returns:
          The circuit, the symbols, and the observable.
        """

        # Observables
        observable = [cirq.Z(qubits[-1])]

        # Sympy symbols for variational angles
        n_qubits = len(qubits)
        num_layers = int(np.log2(n_qubits))

        if n_layers > num_layers:
            print(
                f"Warning: n_layers > {num_layers}."
                f"There can be at most {num_layers} layers with {n_qubits} qubits."
                f"n_layers will be set to {num_layers}.")
            n_layers = num_layers

        var_symbols = sp.symbols(
            f'θ0:{n_layers*(NUM_CONV_SYMBOLS + NUM_POOL_SYMBOLS)}')
        var_symbols = np.asarray(var_symbols).reshape(n_layers, -1)

        data_symbols = []

        circuit = cirq.Circuit()
        n = n_qubits
        for l in range(n_layers):
            conv_start = n - (n // 2**l)
            pool_source_start = conv_start
            pool_source_end = n - (n // 2**(l + 1))
            pool_sink_start = pool_source_end

            if l == (n_layers - 1):
                pool_source_end = n - 1
                pool_sink_start = pool_source_end

            circuit += self._quantum_conv_circuit(
                qubits[conv_start:], var_symbols[l, :NUM_CONV_SYMBOLS])

            circuit += self._quantum_pool_circuit(
                qubits[pool_source_start:pool_source_end],
                qubits[pool_sink_start:], var_symbols[l, NUM_CONV_SYMBOLS:])

            # Re-encoding layer
            if drc and (l < n_layers - 1):
                data_circuit, expr = cirq.flatten(
                    feature_map.build(qubits, in_symbols[l]))
                circuit += data_circuit
                data_symbols += list(expr.values())

        return circuit, data_symbols, list(var_symbols.flat), observable
