from quple._version import __version__
from quple.components.descriptors import *
from quple.circuits.qubit_register import QubitRegister
from quple.circuits.quantum_circuit import QuantumCircuit
from quple.circuits.templates.template_circuit_block import TemplateCircuitBlock
from quple.circuits.parameterised_circuit import ParameterisedCircuit
from quple.circuits.templates.pauli_block import PauliBlock
from quple.circuits.templates.parameterised_block import ParameterisedBlock
from quple.circuits import variational_circuits
from quple.circuits.common_circuits import *
from quple.utils.utils import *

MAX_WORKERS = 8
CV_NJOBS = -1