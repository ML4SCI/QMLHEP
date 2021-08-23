from abc import ABC, abstractmethod
from enum import Enum
import cirq   

class QuantumPlatform(Enum):
	CIRQ = 1
	QISKIT = 2
	BRACKET = 3
    
    