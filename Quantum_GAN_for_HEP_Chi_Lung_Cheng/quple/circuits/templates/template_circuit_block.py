from abc import ABC, abstractmethod
from typing import List, Sequence

class TemplateCircuitBlock(ABC):
    
    @abstractmethod
    def build(self, circuit:'quple.ParameterisedCircuit', qubits:Sequence[int]):
        '''Builds the circuit block involving the specified qubits
        '''
        raise NotImplementedError
    
    @property
    @abstractmethod
    def num_block_qubits(self) -> int:
        """Returns the number of qubits in the circuit block
        """