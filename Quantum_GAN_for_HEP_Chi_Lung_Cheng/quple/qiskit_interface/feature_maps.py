from typing import Optional, List, Union, Callable
import numpy as np
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import PauliFeatureMap 
from qiskit import QuantumCircuit, QuantumRegister
from quple.components.interaction_graphs import interaction_graph
from qiskit.aqua.components.feature_maps import FeatureMap
import operator
import functools

class FeatureMaps01(FeatureMap):
    """
    Mapping data with the second order expansion followed by entangling gates.
    Refer to https://arxiv.org/pdf/1804.11326.pdf for details.
    """

    CONFIGURATION = {
        'name': 'MultiVariableMap',
        'description': 'Second order expansion for feature map',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'Second_Order_Expansion_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
                },
                'entangler_map': {
                    'type': ['array', 'null'],
                    'default': None
                },
                'entanglement': {
                    'type': 'string',
                    'default': 'full',
                    'oneOf': [
                        {'enum': ['full', 'linear']}
                    ]
                }
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits, depth=2, degree=1):
        """Constructor.
        Args:
            num_qubits (int): number of qubits
            depth (int): the number of repeated circuits
            entangler_map (list[list]): describe the connectivity of qubits, each list describes
                                        [source, target], or None for full entanglement.
                                        Note that the order is the list is the order of
                                        applying the two-qubit gate.
            entanglement (str): ['full', 'linear'], generate the qubit connectivitiy by predefined
                                topology
            data_map_func (Callable): a mapping function for data x
        """
        #self.validate(locals())
        super().__init__()
        self._num_qubits = num_qubits
        self._feature_dimension = num_qubits
        self._depth = depth
        self._support_parameterized_circuit = False
        self._support_parameterized_circuit = False
        self._degree = degree

    def _build_circuit_template(self):
        return True

    def construct_circuit(self, x, qr=None, inverse=False):
        #logger.info("8 qubits 26 variable feature map is used")
        """
        Construct the second order expansion based on given data.
        Args:
            x (numpy.ndarray): 1-D to-be-transformed data.
            qr (QauntumRegister): the QuantumRegister object for the circuit, if None,
                                  generate new registers with name q.
            inverse (bool): whether or not inverse the circuit
        Returns:
            QuantumCircuit: a quantum circuit transform data x.
        """

        if qr is None:
            qr = QuantumRegister(self._num_qubits, name='q')

        qc = QuantumCircuit(qr)

        entangler_map = []
        for i in range(0, self._num_qubits - 1, 2):
            entangler_map.append([i, i + 1])
        for i in range(0, self._num_qubits - 2, 2):
            entangler_map.append([i + 1, i + 2])

        for d in range(self._depth):
            for i in range(self._num_qubits):
                qc.h(qr[i])
                qc.rz(x[i], qr[i])
                if self._degree>0:
                   qc.ry(x[i]**self._degree,qr[i])
            for src, targ in entangler_map:
                qc.cx(qr[src], qr[targ])
                qc.rz(((x[src] + x[targ])**self._degree)/np.pi, qr[targ])
                qc.cx(qr[src], qr[targ])
        if inverse:
            qc = qc.inverse()
        return qc

class FeatureMapDev(PauliFeatureMap):
    def pauli_evolution(self, pauli_string, time):
        """Get the evolution block for the given pauli string."""
        # for some reason this is in reversed order
        pauli_string = pauli_string[::-1]

        # trim the pauli string if identities are included
        trimmed = []
        indices = []
        for i, pauli in enumerate(pauli_string):
            if pauli != 'I':
                trimmed += [pauli]
                indices += [i]

        evo = QuantumCircuit(len(pauli_string))

        if len(trimmed) == 0:
            return evo

        def basis_change(circuit, inverse=False):
            # do not change basis if only first order pauli operator
            if len(pauli_string) == 1:
                return
            for i, pauli in enumerate(pauli_string):
                if pauli == 'X':
                    circuit.h(i)
                elif pauli == 'Y':
                    circuit.rx(-np.pi / 2 if inverse else np.pi / 2, i)

        def cx_chain(circuit, inverse=False):
            num_cx = len(indices) - 1
            for i in reversed(range(num_cx)) if inverse else range(num_cx):
                circuit.cx(indices[i], indices[i + 1])

        basis_change(evo)
        cx_chain(evo)
        if len(pauli_string) == 1:
            if pauli_string[0] == 'Z':
                evo.rz(time, indices[-1])
            elif pauli_string[0] == 'X':
                evo.rx(time, indices[-1])
            elif pauli_string[0] == 'Y':
                evo.ry(time, indices[-1])
        else:
            evo.rz(time, indices[-1])
        cx_chain(evo, inverse=True)
        basis_change(evo, inverse=True)
        return evo
    
    def get_entangler_map(self, rep_num: int, 
                          block_num: int, num_block_qubits: int) :
        strategy = self.entanglement
        if isinstance(strategy, str):
            return interaction_graph[strategy](self.num_qubits, num_block_qubits)
        elif callable(strategy):
            return strategy(self.num_qubits, num_block_qubits)
        elif isinstance(strategy, list):
            if all(isinstance(strat, str) for strat in strategy):
                return interaction_graph[strategy[block_num]](self.num_qubits, num_block_qubits)
            elif all(callable(strat) for strat in strategy):
                return strategy[block_num](self.num_qubits, num_block_qubits)
        else:
            raise ValueError('invalid entangle strategy: {}'.format(strategy))


class FeatureMaps02(PauliFeatureMap):

    def __init__(self,
                 feature_dimension: Optional[int] = None,
                 reps: int = 2,
                 entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = 'full',
                 paulis: Optional[List[str]] = None,
                 data_map_func: Optional[Callable[[np.ndarray], float]] = None,
                 parameter_prefix: str = 'x',
                 insert_barriers: bool = False,
                 degree: int= 2
                 ) -> None:

        super().__init__(feature_dimension=feature_dimension,
                         reps=reps,
                         entanglement=entanglement,
                         paulis=paulis,
                         data_map_func=data_map_func,
                         parameter_prefix=parameter_prefix,
                         insert_barriers=insert_barriers)

        self._degree = degree


    def _custom_data_map_func(self, x):
        if len(x) == 1:
            coeff = x[0]
        else:
            coeff = functools.reduce(lambda m, n: (m + n)._apply_operation(operator.pow, self._degree)/(np.pi), x)
        return coeff

    def pauli_block(self, pauli_string):
        """Get the Pauli block for the feature map circuit."""
        params = ParameterVector('_', length=len(pauli_string))
        
        time = self._custom_data_map_func(np.asarray(params))
        return self.pauli_evolution(pauli_string, time)

    def pauli_evolution(self, pauli_string, time):
        """Get the evolution block for the given pauli string."""
        # for some reason this is in reversed order
        pauli_string = pauli_string[::-1]

        # trim the pauli string if identities are included
        trimmed = []
        indices = []
        for i, pauli in enumerate(pauli_string):
            if pauli != 'I':
                trimmed += [pauli]
                indices += [i]

        evo = QuantumCircuit(len(pauli_string))

        if len(trimmed) == 0:
            return evo

        def basis_change(circuit, inverse=False):
            for i, pauli in enumerate(pauli_string):
                if pauli == 'X':
                    circuit.h(i)
                elif pauli == 'Y':
                    if self._degree>0:
                        circuit.ry(time._apply_operation(operator.pow, self._degree), i)

        def cx_chain(circuit, inverse=False):
            num_cx = len(indices) - 1
            for i in reversed(range(num_cx)) if inverse else range(num_cx):
                circuit.cx(indices[i], indices[i + 1])

        basis_change(evo)
        cx_chain(evo)
        if trimmed[0] != 'Y': 
            evo.p(time, indices[-1])
        cx_chain(evo, inverse=True)

        
        return evo            