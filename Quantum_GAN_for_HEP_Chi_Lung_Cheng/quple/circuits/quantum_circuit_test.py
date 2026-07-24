from abc import ABC, abstractmethod
import itertools
from typing import (Any, Callable, cast, Dict, FrozenSet, Iterable, Iterator,
                    List, Optional, overload, Sequence, Set, Tuple, Type,
                    TYPE_CHECKING, TypeVar, Union)
import re
import numpy as np 
from pdb import set_trace
import sympy

import cirq   
from cirq import  devices, GridQubit, ExpressionMap
from cirq.circuits import InsertStrategy
from cirq.protocols import qasm

import quple
from quple.components.gate_ops import RXX, RYY, RZZ
from quple.circuits.qubit_register import QubitRegister
from quple.utils.utils import parallel_run


class QuantumCircuit(ABC):
    def __init__(self, n_qubit:Union[int, Sequence[GridQubit]]=0, name:str='QuantumCircuit',
                insert_strategy:InsertStrategy=None,
                backend=None) -> None:
        """Creates a quantum circuit
        
        Args:
            n_qubit: int, iterable of cirq.GridQubit
                If int, it specifies the number of qubits in the circuit.
                If iterable of cirq.GridQubit object, it specifies the exact
                qubit layout of the circuit. 
            name: str
                Name of the circuit
            insert_strategy: cirq.InsertStrategy, default None
                The insertion strategy of gate operations in the circuit.
                If None, defaults to INLINE. 
            backend: default None
                The backend for the quantum circuit. 
                If None, defaults to quantum simulator.
        """
        super().__init__()
        self._name = name
        self._qr = QubitRegister(n_qubit)
        self._insert_strategy = insert_strategy or InsertStrategy.INLINE
        self._expr_map = None
        self._backend = backend
    
    @property
    def backend(self):
        return self._backend
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def n_qubit(self) -> int:
        return self._qr.size
    

    @n_qubit.setter
    def n_qubit(self, value:int):
        self._qr = QubitRegister(value)
    
    @property
    def circuit(self):
        return self
    
    @property
    def qr(self):
        return self._qr

    @property
    def qubits(self):
        return self._qr.qubits
    
    @property
    def insert_strategy(self):
        return self._insert_strategy

    @property
    def diagram(self):
        print(self)
    
    @property
    def expr_map(self):
        return self._expr_map
    
    @staticmethod
    def _parse_gate_operation(gate:Union[str, cirq.Gate]):
        if isinstance(gate, cirq.Gate):
            return gate
        elif isinstance(gate, str) and (gate in kGateMapping):
            return kGateMapping[gate]
        else:
            raise KeyError("Unknown gate operation {}".format(gate))

    @property
    def symbols(self):
        return quple.get_circuit_symbols(self, to_str=True)  
    
    @property
    def raw_symbols(self):
        return quple.get_circuit_unflattened_symbols(self, to_str=True)  

    def apply_gate_operation(self, operation:[str, cirq.Gate], qubit_expr, params=None):
        """Apply a gate operation on designated qubits
        
        Args:
            operation: the gate operation to apply
            qubit_expr: the qubits on which the gate operation is applied
            params: a dict containing the parameters for the gate operation
        """
        operation = QuantumCircuit._parse_gate_operation(operation)
        if params is not None:
            operation = operation(**params)
        qubit_sequence = self.qr.get(qubit_expr)
        if not isinstance(qubit_sequence, (list, np.ndarray)):
            qubit_sequence = [qubit_sequence]
        strategy = self.insert_strategy
        if all(isinstance(qubits, GridQubit) for qubits in qubit_sequence):
            self.append([operation(qubit) for qubit in qubit_sequence], strategy=strategy)
        elif all(isinstance(qubits, tuple) for qubits in qubit_sequence):
            self.append([operation(*qubits) for qubits in qubit_sequence], strategy=strategy)
        else:
            raise ValueError('Inconsistent qubit representation: {}'.format(qubit_sequence))
            
    def get_param_resolver(self, param_values: Dict):
        """
        """
        return cirq.ParamResolver(param_values)
    
    def X(self, qubit_expr):
        """The Pauli X gate
        
        X = {{0, 1}, {1, 0}}
        
        Performs the Pauli X gate operation 
        """
        self.apply_gate_operation('X', qubit_expr)
        
    def Y(self, qubit_expr):
        """The Pauli Y gate
        
        Y = {{0, -i}, {i, 0}}
        
        Performs the PauliY gate operation 
        """        
        self.apply_gate_operation('Y', qubit_expr)
        
    def Z(self, qubit_expr):
        """The Pauli Z gate
        
        Z = {{1, 0}, {0, -1}}
        
        Performs the Pauli Z gate operation 
        """        
        self.apply_gate_operation('Z', qubit_expr)        

    def S(self, qubit_expr):
        """The Clifford-S (phase) gate
        
        S = {{1, 0}, {0, i}}
        
        Performs the phase gate opreation
        """
        self.apply_gate_operation(cirq.ops.S, qubit_expr)

    def T(self, qubit_expr):
        """The non-Clifford-T gate
        
        T = {{1, 0}, {0, exp(i\pi/4)}}
        
        Performs the T gate operation
        """
        self.apply_gate_operation(cirq.ops.T, qubit_expr)

    def H(self, qubit_expr):
        """The Hadamard gate
        
        H = 1/sqrt{2} {{1,1}, {1, -1}}
        
        Performs the Hadamard gate operation
        """
        self.apply_gate_operation(cirq.ops.H, qubit_expr)
    
    def RX(self, theta:Union[int, float], qubit_expr):
        """Rotation about the X axis
        
        RX(θ) = exp(-iθX/2) = {{cos(θ/2), -i*sin(θ/2)}, {-i*sin(θ/2), cos(θ/2)}}
        
        Performs single qubit rotation about the X axis
        """
        self.apply_gate_operation(cirq.ops.rx(theta), qubit_expr)

    def RY(self, theta:Union[int, float], qubit_expr):
        """Rotation about the Y axis
        
        RY(θ) = exp(-iθY/2) = {{cos(θ/2), -*sin(θ/2)}, {sin(θ/2), cos(θ/2)}}
        
        Performs single qubit rotation about the Y axis
        """        
        self.apply_gate_operation(cirq.ops.ry(theta), qubit_expr)

    def RZ(self, theta:Union[int, float], qubit_expr):
        """Rotation about the Z axis
        
        RZ(θ) = exp(-iθZ/2) = {{exp(-iθ/2), 0}, {0, exp(iθ/2)}}
        
        Performs single qubit rotation about the Z axis
        """        
        self.apply_gate_operation(cirq.ops.rz(theta), qubit_expr)                
    
    def SWAP(self, qubit_expr):
        """The SWAP gate
        
        SWAP = {{1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}}
        
        Performs the swap gate operation
        """
        self.apply_gate_operation(cirq.ops.SWAP, qubit_expr)
        
    def ISWAP(self, qubit_expr):
        """The iSWAP gate
        
        ISWAP = R_{XX+YY}(-pi/2) = 
        {{1, 0, 0, 0}, {0, 0, i, 0}, {0, i, 0, 0}, {0, 0, 0, 1}}
        
        Performs the iswap gate operation
        """
        self.apply_gate_operation(cirq.ops.ISWAP, qubit_expr)
        
    def RISWAP(self, theta:Union[int, float], qubit_expr):
        """The iSWAP power gate
        
        RISWAP = R_{XX+YY}(θ) = exp(iθ(X⊗X + Y⊗Y)/2) = 
        {{1, 0, 0, 0}, {0, c, i*s, 0}, {0, i*s, c, 0}, {0, 0, 0, 1}}
        
        where c = cos(θ) and s = sin(θ)
        
        Performs iswap gate operation to the power 2*theta/pi
        """
        self.apply_gate_operation(cirq.ops.riswap(rads=theta), qubit_expr)      
        
    def FSim(self, qubit_expr):
        """The FSim gate
        
        FSimGate(θ, φ) = ISWAP**(-2θ/π) CZPowGate(exponent=-φ/π) =
        {{1, 0, 0, 0}, {0, a, b, 0}, {0, b, a, 0}, {0, 0, 0, c}}
        
        where a = cos(0), b = -i*sin(0), c = exp(-iφ)
        
        Performs the fsim gate operation
        """
        self.apply_gate_operation(cirq.ops.FSimGate, qubit_expr)        
                 

    def CNOT(self, qubit_expr):
        """The controlled NOT gate
        
        CNOT = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}}
        
        Performs the CNOT gate operation
        """
        self.apply_gate_operation(cirq.ops.CNOT, qubit_expr)
    
    def Toffoli(self, qubit_expr):
        """The Toffoli gate (Controlled-Controlled NOT gate)
        
        TOFF = {{I, 0, 0, 0}, {0, I, 0, 0}, {0, 0, I, 0}, {0, 0, 0, J}}
        
        where I = {{1, 0}, {0, 1}} and J = {{0, 1}, {1, 0}}
        
        Performs the Toffoli gate operation
        """
        self.apply_gate_operation(cirq.ops.TOFFOLI, qubit_expr)

    def PhaseShift(self, phi, qubit_expr):
        """The phase shift gate
        
        PhaseShift = {{1, 0}, {0, exp(iφ)}}
        
        Performs the phase shift gate operation
        """
        self.apply_gate_operation(cirq.ZPowGate(exponent=phi / np.pi), qubit_expr)
            
    def CX(self, qubit_expr):
        """Equivalent to CNOT gate
        
        CX = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}}
        
        Performs the controlled-X gate operation
        """
        self.apply_gate_operation(cirq.ops.CX, qubit_expr)
  
    def CZ(self, qubit_expr):
        """The Controlled-Z gate
        
        CZ = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, -1}}
        
        Performs the controlled-Z gate operation
        """
        self.apply_gate_operation(cirq.ops.CZ, qubit_expr)
        
    def XX(self, qubit_expr):
        """The Ising coupling (XX) gate
        
        The XX gate is a 2 qubit gate that is a tensor product of two Pauli X gates
        
        XX = {{0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}}
        
        Performs the XX gate operation
        """
        self.apply_gate_operation(cirq.ops.XX, qubit_expr)
        
    def YY(self, qubit_expr):
        """The Ising coupling (XX) gate
        
        The YY gate is a 2 qubit gate that is a tensor product of two Pauli Y gates
        
        YY = {{0, 0, 0, -1}, {0, 0, 1, 0}, {0, -1, 0, 0}, {1, 0, 0, 0}}
        
        Performs the YY gate operation
        """
        self.apply_gate_operation(cirq.ops.YY, qubit_expr)
        

    def ZZ(self, qubit_expr):
        """The Ising coupling (ZZ) gate
        
        The ZZ gate is a 2 qubit gate that is a tensor product of two Pauli Z gates
        
        ZZ = {{1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, 1}}
        
        Performs the ZZ gate operation
        """
        self.apply_gate_operation(cirq.ops.ZZ, qubit_expr)        
        
    def XPowGate(self, theta:Union[int, float], qubit_expr):
        """The X Power Gate
        
        XPowGate = {{1, 0}, {0, exp(i*pi*theta)}}
        
        Performs the rotation along X axis with some global phase shift 
        """
        self.apply_gate_operation(cirq.ops.XPowGate(exponent=theta), qubit_expr)            
        
    def YPowGate(self, theta:Union[int, float], qubit_expr):
        """The Y Power Gate
        
        YPowGate = {{1, 0}, {0, exp(i*pi*theta)}}
        
        Performs the rotation along Y axis with some global phase shift 
        """
        self.apply_gate_operation(cirq.ops.YPowGate(exponent=theta), qubit_expr)     
        
    def ZPowGate(self, theta:Union[int, float], qubit_expr):
        """The Z Power Gate
        
        ZPowGate = {{1, 0}, {0, exp(i*pi*theta)}}
        
        Performs the rotation along Z axis with some global phase shift 
        """
        self.apply_gate_operation(cirq.ops.ZPowGate(exponent=theta), qubit_expr) 
        
    def clear(self) -> None:
        """Clears all gate operations in the circuit
        """
        self._moments = []
        
    def _entangled_qubit_pairing(self, qubits: Sequence[int], 
        *args, **kwargs) -> List[Tuple[int]]:
        """Determines how qubits are paired in the entanglement operation
        Args:
            qubits: qubits to be entangled
        Return:
            list of pair of qubit indices for entanglement
        Example:
        >>> cq = EntanglementCircuit(n_qubit=5)
        >>> cq._entangled_qubit_pairing((1,3,4)) #entangle qubits 1, 3 and 4
        [(1,3), (3,4)]
        """
        pairing_indices = [(qubits[i], qubits[i+1]) for i in range(len(qubits)-1)]
        
        return pairing_indices
    
    def entangle(self, qubits:Sequence[int],
                 inverse:bool=False,
                 gate:cirq.Gate=cirq.ops.CNOT):
        """entangle qubits in a quantum cirquit
        Args:
            qubits: qubits to be entangled
            inverse: reverse the order of operation
            gate: gate operation that entangles the qubits
        Example:
        >>> cq = QuantumCircuit(n_qubit=5)
        >>> cq.entangle((1,2))
        >>> cq.entangle((0,3,4))
                       ┌──┐
            (0, 0): ─────@────────
                         │
            (1, 0): ────@┼────────
                        ││
            (2, 0): ────X┼────────
                         │
            (3, 0): ─────X────@───
                              │
            (4, 0): ──────────X───
                       └──┘
        """
        # cannot entangle itself
        if len(qubits) == 1:
            return
        pairing_indices = self._entangled_qubit_pairing(qubits)
        qubit_pairs = self.qr.get(pairing_indices)
        if inverse:
            qubit_pairs = qubit_pairs[::-1]
        self.append([gate(*qpair) for qpair in qubit_pairs])  
        
    def measure(self, qubit_expr, key=None):
        """Performs a measurement on specific qubits
        
        Args:
            qubit_expr: The qubits or the qubit indices corresponding to the qubits on which
                        the measurement is performed
            key: The string key of the measurement.
        """
        qubit = self.qr.get(qubit_idx)
        self.append(cirq.measure(qubit, key=key))

    def gate_operation(self, operation, qubit_expr, params=None):
        """Returns gate operation on specific qubit
        
        Args:
            operation: The gate operation to use
            qubit_expr: The qubits or the qubit indices corresponding to the qubits on which
                        the measurement is performed
            key: The string key of the measurement.
        """
        operation = QuantumCircuit._parse_gate_operation(operation)
        if params is not None:
            operation = operation(**params)    
        qubit = self.qr.get(qubit_expr)     
        return operation(qubit)     
    
    def assign(self, circuit:'cirq.Circuit'):
        """Replaces the gate operations of the current circuit by that of another circuit
        
        Args:
            circuit: The circuit from which the gate operations are replaced
        """
        self._moments = circuit._moments
    
    @classmethod
    def from_cirq(cls, circuit:cirq.Circuit):
        """Creates a quple.QuantumCircuit instance from a cirq.Circuit instance
        
        Args:
            circuit: A cirq quantum circuit
        
        Returns:
            A quple quantum circuit 
        """
        qubits = quple.get_circuit_qubits(circuit)
        symbols = quple.get_circuit_symbols(circuit)
        cq = cls(qubits)
        cq._parameter_table.append(symbols)
        cq.append(circuit)
        return cq

    
    @staticmethod
    def _get_unflattened_circuit(circuit:'quple.QuantumCircuit'):
        """Returns the quantum circuit with flattened gate operations recovered
        
        Args:
            circuit: The quantum circuit to unflatten its gate operations
            
        Returns:
            A quantum circuit with unflattened gate operations
        """
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError('Circuit to unflatten must be a quple.QuantumCircuit instance')
        # skip if circuit is not flattened in the first place
        if not circuit.expr_map:
            return circuit
        reverse_expr_map = { val: key for key, val in circuit.expr_map.items()}
        return cirq.protocols.resolve_parameters(circuit, reverse_expr_map)
    
    def get_unflattened_circuit(self):
        """Returns the current circuit with flattened gate operations recovered
        
        Returns:
            An unflattened version of the current circuit
        """        
        return self._get_unflattened_circuit(self)
        
    def resolve_parameters(self, vals:np.ndarray)-> Union['cirq.Circuit', List['cirq.Circuit']]:
        """Resolves symbol parameters in the circuit 
        
        Args:
            vals: dict, list of dict, numpy array
                The values feeded to the parameter resolver
                If dict, represents the map from sympy.Symbol objects to symbol values
                If list of dict, represents a list of parameter resolvers each with a
                map from sympy.Symbol objects to symbol values.
                If numpy array, represents a mapping of symbol values according to the
                symbol indices. For example for a circuit with parameter symbols 
                ['x_0', 'x_1', 'x_2'], feeding parameter values [1, 2, 3] is equivalent
                to feeding the map {'x_0': 1, 'x_1':2, 'x_2': 3}. The parameter symbols
                are ordered according to natural sorting, i.e. 'x_1' < 'x_2' < 'x_10'.
        Returns:
            A circuit or list of circuits with parameter expressions resolved
        """
        param_resolver = self.get_parameter_resolver(vals)
        
        if isinstance(param_resolver, (list, cirq.ListSweep)):
            return parallel_run(self.get_resolved_circuit, param_resolver)
        else:
            return self.get_resolved_circuit(param_resolver)
                                
    def get_resolved_circuit(self, param_resolver):
        return cirq.protocols.resolve_parameters(self, param_resolver)        
        
    def get_parameter_resolver(self, vals:np.ndarray) -> Union['cirq.ParamResolver',List['cirq.ParamResolver']]:
        """Returns a parameter resolver for the parameter expressions in the circuit
        
        Args:
            vals: dict, list of dict, numpy array
                The values feeded to the parameter resolver
                If dict, represents the map from sympy.Symbol objects to symbol values
                If list of dict, represents a list of parameter resolvers each with a
                map from sympy.Symbol objects to symbol values.
                If numpy array, represents a mapping of symbol values according to the
                symbol indices. For example for a circuit with parameter symbols 
                ['x_0', 'x_1', 'x_2'], feeding parameter values [1, 2, 3] is equivalent
                to feeding the map {'x_0': 1, 'x_1':2, 'x_2': 3}. The parameter symbols
                are ordered according to natural sorting, i.e. 'x_1' < 'x_2' < 'x_10'.
        Returns:
            A parameter resolver for the parameter expressions in the circuit
        """
        resolver = None
        if isinstance(vals, dict):
            resolver = cirq.ListSweep([cirq.ParamResolver(vals)])
        elif isinstance(vals, list):
            if all(isinstance(val, dict) for val in vals):
                resolver = cirq.ListSweep([cirq.ParamResolver(val) for val in vals])
            else:
                raise ValueError('Parameter values of type list must have elements '
                                 'of dict type representing the map from sympy.Symbol '
                                 'objects to symbol values')
        elif isinstance(vals, np.ndarray):
            ndim = vals.ndim
            if ndim == 1:
                vals = vals.reshape(-1,vals.size)
            elif ndim != 2:
                raise ValueError('Array type parameter values must be of dimension 1 or 2')
            import pandas as pd 
            df = pd.DataFrame(vals, columns=self.raw_symbols)
            resolver = cirq.ListSweep(np.array([i for i in df.T.to_dict().values()]))
        else:
            raise ValueError('Invalid value format for parameter resolver. '
                             'Allowed formats are dict, sequence of dicts '
                             'or numpy arrays')
        if self.expr_map and (resolver is not None):
            resolver = cirq.ListSweep(parallel_run(self.expr_map.transform_params, resolver))
        
        if isinstance(resolver, cirq.ListSweep) and (len(resolver)==1):
            resolver = resolver[0]
            
        return resolver
    
    def _get_resolved_symbols_and_values(self, vals:np.ndarray):
        resolvers = self.get_parameter_resolver(vals)
        values = np.array([list(resolver.param_dict.values()) for resolver in resolvers])
        symbols = list(resolvers[0].param_dict.keys())
        return symbols, values
    
    def get_state_vectors(self, vals:Optional[np.ndarray]=None):
        import tensorflow_quantum as tfq
        state_layer = tfq.layers.State(backend=self.backend)
        num_parameters = len(self.raw_symbols) 
        if num_parameters > 0:
            if vals is None:
                raise ValueError('Parameter values are required for a parameterized circuit')    
            ndim = vals.ndim
            if ndim != 2:
                raise ValueError('Parameter values must be a 2D array')
            if num_parameters != vals.shape[1]:
                raise ValueError('Parameter dimension does not match the number of symbols in parameterized circuit')
            symbols, resolved_values = self._get_resolved_symbols_and_values(vals)
            states = state_layer(self, symbol_names=symbols, symbol_values=resolved_values)
        else:
            if vals is not None:
                raise ValueError('Parameter values not allowed for non-parameterized circuit')
            states = state_layer(self)
        state_vectors = np.array([state.numpy() for state in states])
        return state_vectors
    
    def to_qasm_str(self):
        """Returns the qasm string of the circuit
        """
        return qasm(self)
    
    def flatten(self):
        """Flattens all gate operations with symbolic expression as parameters 
        """
        # get flattened circuit and corresponding expr_map
        cq_flat, expr_map = cirq.flatten(self)
        self.assign(cq_flat)
        if self.expr_map is not None:
            self._expr_map = quple.resolve_expression_map_conflicts(self.expr_map, expr_map)
        else:
            self._expr_map = expr_map
        
    def unflatten(self):
        """Unflattens all gate operations with symbolic expression as parameters 
        """        
        self.assign(self.get_unflattened_circuit())
        self._expr_map = None
            