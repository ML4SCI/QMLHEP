from typing import Sequence
import sympy as sp
import numpy as np

import quple
from quple import TemplateCircuitBlock

class PauliBlock(TemplateCircuitBlock):
    def __init__(self, pauli_string:str, encoding_map=None, global_shift=False):
        self.paulis = pauli_string[::-1]
        self.indices = [i for i, pauli in enumerate(self.paulis) if pauli != 'I']
        self.global_shift = global_shift
        self._num_block_qubits = len(self.paulis)
        self.encoding_map = encoding_map
    
    @staticmethod
    def change_basis(circuit:'quple.ParameterisedCircuit', qubits:Sequence[int],
                     pauli_string:Sequence[str], inverse=False) -> None:
        # do not change basis if only first order pauli operator
        if len(pauli_string) == 1:
            return
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X':
                circuit.H(qubits[i])
            elif pauli == 'Y':
                circuit.RX(-np.pi / 2 if inverse else np.pi / 2, qubits[i])            

    def build(self, circuit:'quple.ParameterisedCircuit', qubits:Sequence[int]):
        # Check if no effective Pauli operations
        if not self.indices:
            return None
        
        PauliBlock.change_basis(circuit, qubits, self.paulis)
        
        qubits_to_entangle = tuple(qubits[i] for i in self.indices)
        # prepare encoded parameters
        from quple.data_encoding import EncodingCircuit
        if isinstance(circuit, EncodingCircuit):
            encoded_value = circuit.encode_parameters(list(qubits), 
                                                      encoding_map=self.encoding_map)            
        elif isinstance(circuit, quple.ParameterisedCircuit):
            encoded_value = circuit.new_param(size=1)
        else:
            raise ValueError('Unknown circuit type: {}'.format(type(circuit)))
        circuit.entangle(qubits_to_entangle)
        params = (encoded_value, qubits[-1])
        
        # do not switch to RZ gate if only first order pauli operator
        if len(self.paulis) == 1:
            if self.global_shift:
                if self.paulis[0] == 'Z':
                    circuit.ZPowGate(*params)
                elif self.paulis[0] == 'X':
                    circuit.XPowGate(*params)
                elif self.paulis[0] == 'Y':
                    circuit.YPowGate(*params)                
            else:
                if self.paulis[0] == 'Z':
                    circuit.RZ(*params)
                elif self.paulis[0] == 'X':
                    circuit.RX(*params)
                elif self.paulis[0] == 'Y':
                    circuit.RY(*params)      
        else:
            if self.global_shift:
                circuit.ZPowGate(*params)
            else:
                circuit.RZ(*params)
        circuit.entangle(qubits_to_entangle, inverse=True)
        
        PauliBlock.change_basis(circuit, qubits, self.paulis, inverse=True)       
    
    @property
    def num_block_qubits(self) -> int:
        return self._num_block_qubits