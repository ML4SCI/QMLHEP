from typing import Sequence, Optional, Union, List, Callable, Tuple
import sympy as sp
import numpy as np

import cirq
import quple
from quple import TemplateCircuitBlock, ParameterisedCircuit

class ParameterisedBlock(TemplateCircuitBlock):
    def __init__(self, copies:int=1,
                       rotation_blocks:Optional[Union[str, cirq.Gate, Callable, 'TemplateCircuitBlock',
                                            List[str],List[cirq.Gate],List[Callable],
                                            List['TemplateCircuitBlock']]] =None,
                       entanglement_blocks:Optional[Union[str, cirq.Gate, Callable, 'TemplateCircuitBlock',
                                            List[str],List[cirq.Gate],List[Callable],
                                            List['TemplateCircuitBlock']]] =None,
                       entangle_strategy:Optional[Union[str,List[str], Callable[[int,int],List[Tuple[int]]],
                                            List[Callable[[int,int],List[Tuple[int]]]]]]=None):
        self.copies = copies
        self.rotation_blocks = rotation_blocks
        self.entanglement_blocks = entanglement_blocks
        self.entangle_strategy = entangle_strategy
        self._num_block_qubits = 2
        
    def build(self, circuit:'quple.ParameterisedCircuit', qubits:Sequence[int]):

        block = ParameterisedCircuit(circuit.qr.get(qubits), copies=self.copies,
                                     rotation_blocks=self.rotation_blocks,
                                     entanglement_blocks=self.entanglement_blocks,
                                     parameter_index=circuit._parameter_index)
        circuit.append(block)   
    
    @property
    def num_block_qubits(self) -> int:
        return self._num_block_qubits