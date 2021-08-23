from abc import ABC, abstractmethod
import itertools
from typing import (Any, Callable, cast, Dict, FrozenSet, Iterable, Iterator,
                    List, Optional, overload, Sequence, Set, Tuple, Type,
                    TYPE_CHECKING, TypeVar, Union)
import re
import math
import numpy as np 
from pdb import set_trace
import sympy

import cirq   
from cirq import  devices, GridQubit, ExpressionMap
from cirq.circuits import InsertStrategy
from cirq.protocols import qasm

import quple
from quple.components.gate_ops import RXX, RYY, RZZ, CompositeGate, PauliRotation
from quple.circuits.qubit_register import QubitRegister
from quple.utils.utils import parallel_run

kGateMapping = {
        "H": cirq.H, # Hadamard gate
        "I": cirq.I,  # one-qubit Identity gate
        "S": cirq.S, # Clifford S gate
        "T": cirq.T, # non-Clifford T gate
        'X': cirq.X, # Pauli-X gate
        "Y": cirq.Y, # Pauli-Y gate
        "Z": cirq.Z, # Pauli-Z gate
        "PauliX": cirq.X, # Pauli-X gate
        "PauliY": cirq.Y, # Pauli-Y gate
        "PauliZ": cirq.Z, # Pauli-Z gate
        "CX": cirq.CX, # Controlled-NOT gate
        "CNOT": cirq.CNOT, # Controlled-NOT gate
        "CZ": cirq.CZ, # Controlled-Z ef
        "XX": cirq.XX, # tensor product of two X gates (X parity gate)
        "YY": cirq.YY, # tensor product of two Y gates (Y parity gate)
        "ZZ": cirq.ZZ, # tensor product of two Z gates (Z parity gate)
        "XPowGate": cirq.XPowGate, # rotation along X axis with extra phase factor
        "YPowGate": cirq.YPowGate, # rotation along Y axis with extra phase factor 
        "ZPowGate": cirq.ZPowGate, # rotation along Z axis with extra phase factor
        "XXPowGate": cirq.XXPowGate, # X parity gate raised to some power
        "YYPowGate": cirq.YYPowGate, # Y parity gate raised to some power
        "ZZPowGate": cirq.ZZPowGate, # Z parity gate raised to some power
        "MS": cirq.ms, # Mølmer–Sørensen gate == RXX (A rotation around the XX axis in the two-qubit bloch sphere)
        "RXX": RXX, # XX Ising coupling gate (A rotation around the XX axis in the two-qubit bloch sphere)
        "RYY": RYY, # YY Ising coupling gate (A rotation around the YY axis in the two-qubit bloch sphere)
        "RZZ": RZZ, # ZZ Ising coupling gate (A rotation around the ZZ axis in the two-qubit bloch sphere)                        
        "RX": cirq.rx, # rotation along X axis
        "RY": cirq.ry, # rotation along Y axis
        "RZ": cirq.rz, # rotation along Z axis
        "CCNOT": cirq.CCNOT, # Toffoli gate
        "CCX": cirq.CCX, # Toffoli gate
        "Toffoli": cirq.TOFFOLI, # Toffoli gate
        "SWAP": cirq.SWAP, # SWAP gate
        "CSWAP": cirq.CSWAP, # Controlled SWAP gate
        "ISWAP": cirq.ISWAP, # ISWAP gate
        "RISWAP": cirq.riswap, #Rotation ISWAP gate (X⊗X + Y⊗Y)
        "FSim": cirq.FSimGate, # Fermionic simulation gate
        "Fredkin": cirq.FREDKIN, # Controlled SWAP gate
        "CXPowGate": cirq.CXPowGate, # Controlled Power of an X gate
        "CZPowGate": cirq.CZPowGate, # Controlled Power of an Z gate
        "CNOTPowGate": cirq.CXPowGate, # Controlled Power of an X gate
        "PauliRotation": PauliRotation # General Pauli Rotation 
    }


class QuantumCircuit(cirq.Circuit):
    """A wrapper for the construction of quantum circuits based on the Google cirq library   
    
    A quantum circuit consists of a system of qubits together with a sequence of unitary operations
    (quantum gates) applied to the qubits which transform the quantum state of the system. Designing
    a quantum circuit is the basis for construction of quantum algorithms for solving problems which
    may be classically unreachable. 
    
    In the scenario of a parameterised quantum circuit, the gate operations are parameterised by a 
    symbolic expression via the sympy library. The expressions in a paramterised circuit can 
    subsequently be resolved by providing a symbol to value map. Parameterised circuits are important
    for applications in machine learning alogorithms. They can serve as data encoding circuits where
    a feature vector may be embeded into the gate parameters, or model circuits where the gate parameters
    represent the weights of a model.
        
    Examples:
    # Construction of the circuit for a Bell state
    >>> # Creates a circuit with 2 qubits
    >>> cq = QuantumCircuit(2, name='BellCircuit')
    >>> # Apply the Hadamard gate to the qubit with index 0 
    >>> cq.H(0)
    >>> # Apply the CNOT gate to qubits with indices 0 and 1
    >>> # with the first one being the control qubit
    >>> cq.CNOT((0,1))
    >>> # Print out the circuit diagram
    >>> cq
    (0, 0): ───H───@───
                   │
    (0, 1): ───────X───
    # Construction of a parameterised circuit
    >>> # Creates a circuit with 3 qubits
    >>> cq = QuantumCircuit(3, name='PQC')    
    >>> # Apply the Hadamard gate to all qubits
    >>> cq.H(cq.qubits)
    >>> # Create an array of symbols of size 3 with 'θ' as prefix
    >>> theta = sympy.symarray('θ', 3)
    >>> # Apply the RZ gate to all qubits parameterised by θ
    >>> for i, qubit in enumerate(cq.qubits):
    >>>    cq.RZ(theta[i], qubit)
    >>> # Apply CNOT to qubits with indices 0 and 1 and
    >>> # qubits with indices 1 and 2 with the first one
    >>> # being the control qubit
    >>> cq.CNOT([(0,1), (1, 2)])
    >>> # Print out the circuit diagram
    >>> cq
    (0, 0): ───H───Rz(θ_0)───@───────
                             │
    (0, 1): ───H───Rz(θ_1)───X───@───
                                 │
    (0, 2): ───H───Rz(θ_2)───────X───
    >>> # Resolve the parameters of the circuit
    >>> parameter_values = np.array([np.pi, 2*np.pi, 3*np.pi])
    >>> resolved_cq = cq.resolve_parameters(parameter_values)
    >>> # Print out the resolved circuit
    >>> resolved_cq
    (0, 0): ───H───Rz(π)────@───────
                            │
    (0, 1): ───H───Rz(2π)───X───@───
                                │
    (0, 2): ───H───Rz(-π)───────X───

    Args:
        n_qubit: int, iterable of cirq.GridQubit
            If int, it specifies the number of qubits in the circuit.
            If iterable of cirq.GridQubit object, it specifies the exact
            qubit layout of the circuit. 
        name: str
            Name of the circuit
        insert_strategy: cirq.InsertStrategy, default=None
            The insertion strategy of gate operations in the circuit.
            If None, defaults to INLINE. 
        backend: default None
            The backend for the quantum circuit. 
            If None, defaults to quantum simulator.
    """
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
                The backend for circuit simulation. 
                If None, defaults to the native TensorFlow simulator
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
    def n_qubit(self, value:Union[int, Sequence[GridQubit]]):
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
        if isinstance(gate, (cirq.Gate, CompositeGate)):
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
    
    def set_insert_strategy(self, insert_strategy: InsertStrategy):
        if not isinstance(insert_strategy, InsertStrategy):
            raise ValueError('insert strategy must be derived from the cirq.circuits.InsertStrategy class')
        self._insert_strategy = insert_strategy
        
    def get_gate_operation(self, operation:[str, cirq.Gate], qubit_expr, params:Optional[Dict]=None):
        """Obtain a list of gate operations on designated qubits
        
        Args:
            operation: the gate operation to apply
            qubit_expr: the qubits on which the gate operation is applied
            params: a dict containing the parameters for the gate operation
        
        Returns:
            List of gate operations on designated qubits
        """
        operation = QuantumCircuit._parse_gate_operation(operation)
        if params is not None:
            operation = operation(**params)
        qubit_sequence = self.qr.get(qubit_expr)
        if not isinstance(qubit_sequence, (list, np.ndarray)):
            qubit_sequence = [qubit_sequence]
        
        gate_ops = None
        if all(isinstance(qubits, GridQubit) for qubits in qubit_sequence):
            gate_ops = [operation(qubit) for qubit in qubit_sequence]
        elif all(isinstance(qubits, tuple) for qubits in qubit_sequence):
            gate_ops = [operation(*qubits) for qubits in qubit_sequence]
        else:
            raise ValueError('Inconsistent qubit representation: {}'.format(qubit_sequence))

        return gate_ops

    def apply_gate_operation(self, operation:[str, cirq.Gate], qubit_expr, params:Optional[Dict]=None):
        """Apply a gate operation on designated qubits
        
        Args:
            operation: the gate operation to apply
            qubit_expr: the qubits on which the gate operation is applied
            params: a dict containing the parameters for the gate operation
        """
        gate_ops = self.get_gate_operation(operation, qubit_expr, params)
        strategy = self.insert_strategy
        self.append(gate_ops, strategy=strategy)
            
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

    def RXX(self, theta:Union[int, float], qubit_expr):
        """Rotation about the X \otimes X axis
        
        RXX
        
        Performs two qubit rotation about the X \otimes X axis
        """        
        self.apply_gate_operation(RXX(theta), qubit_expr)  

    def RXX(self, theta:Union[int, float], qubit_expr):
        """Rotation about the Y \otimes Y axis
        
        RXX
        
        Performs two qubit rotation about the Y \otimes Y axis
        """        
        self.apply_gate_operation(RYY(theta), qubit_expr)     

    def RZZ(self, theta:Union[int, float], qubit_expr):
        """Rotation about the Z \otimes Z axis
        
        RXX
        
        Performs two qubit rotation about the Z \otimes Z axis
        """        
        self.apply_gate_operation(RZZ(theta), qubit_expr)                           

    def RZ(self, theta:Union[int, float], qubit_expr):
        """Rotation about the Z axis
        
        RZ(θ) = exp(-iθZ/2) = {{exp(-iθ/2), 0}, {0, exp(iθ/2)}}
        
        Performs single qubit rotation about the Z axis
        """        
        self.apply_gate_operation(cirq.ops.rz(theta), qubit_expr) 

    def PauliRotation(self, pauli_string, theta:Union[int, float], qubit_expr, global_shift=False):
        self.apply_gate_operation(PauliRotation(pauli_string, theta, global_shift=global_shift), qubit_expr)               
    
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
        qubit = self.qr.get(qubit_expr)
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
        num_parameters = len(self.raw_symbols) 
        if num_parameters == 0:
            raise ValueError('circuit is not parameterised')
        param_resolver = self.get_parameter_resolver(vals)
        
        if isinstance(param_resolver, (list, cirq.ListSweep)):
            return parallel_run(self.get_resolved_circuit, param_resolver)
        else:
            return self.get_resolved_circuit(param_resolver)
    
    def resolve_parameters_with_tfq(self, vals:np.ndarray)-> Union['cirq.Circuit', List['cirq.Circuit']]:
        import tensorflow_quantum as tfq
        symbols, resolved_values = self._get_resolved_symbols_and_values(vals)
        tensor_circuits =  tfq.convert_to_tensor([self for _ in range(len(resolved_values))])
        resolved_circuits = tfq.from_tensor(tfq.resolve_parameters(tensor_circuits, symbols, resolved_values))
        return resolved_circuits
                                
    def get_resolved_circuit(self, param_resolver):
        return cirq.protocols.resolve_parameters(self, param_resolver)  
    
    def get_kernel_matrix_with_tfq(self, x_left:np.ndarray, x_right:np.ndarray=None, stride:int=50):
        import tensorflow_quantum as tfq
        import tensorflow as tf
        resolved_circuits_left = tfq.convert_to_tensor(self.resolve_parameters(x_left))
        if x_right == None:
            resolved_circuits_right = resolved_circuits_left
        else:
            resolved_circuits_right = tfq.convert_to_tensor(self.resolve_parameters(x_right))
        dimension = tf.gather(tf.shape(resolved_circuits_left), 0)
        empty_symbols = tf.convert_to_tensor([], dtype=tf.dtypes.string)
        empty_values = tf.tile(tf.convert_to_tensor([[]]), [stride, 1])
        resolved_circuits_right = tf.reshape(tf.tile(resolved_circuits_right, [stride]), [stride, dimension])
        
        kernel_matrix = np.empty((dimension, dimension))
        for i in range(0, dimension, stride):
            interim = tfq.math.inner_product(resolved_circuits_left[i:i+stride], empty_symbols, empty_values, resolved_circuits_right)
            kernel_matrix[i:i+stride] = (tf.math.abs(interim) ** 2).numpy()

        return kernel_matrix
        
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
        elif isinstance(vals, np.ndarray) or hasattr(vals, 'ndim'):
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
    
    def _get_tfq_layer_result(self, layer, vals:Optional[np.ndarray]=None, **kwargs):
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
            result = layer(self, symbol_names=symbols, symbol_values=resolved_values, **kwargs)
        else:
            if vals is not None:
                raise ValueError('Parameter values not allowed for non-parameterized circuit')
            result = layer(self, **kwargs)
        return result
    
    def get_state_vectors(self, vals:Optional[np.ndarray]=None):
        import tensorflow_quantum as tfq
        state_layer = tfq.layers.State(backend=self.backend)
        states = self._get_tfq_layer_result(state_layer, vals)
        state_vectors = np.array([state.numpy() for state in states])
        return state_vectors
    
    def get_expectation(self, vals:Optional[np.ndarray]=None, operators=None, repetitions=5000):
        '''Sample expectation values given measurement operators and repetitions
        '''
        if operators is None:
            operators = [cirq.Z(self.qubits[0])]
        import tensorflow_quantum as tfq
        expactation_layer = tfq.layers.SampledExpectation(backend=self.backend)
        results = self._get_tfq_layer_result(expactation_layer, vals,
                                             operators=operators,
                                             repetitions=repetitions)
        expectation = np.array([result.numpy() for result in results])
        return expectation
    
    def get_state_vectors_deprecated(self, vals:Optional[np.ndarray]=None):
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
    
    def get_state_vectors_test(self, x:Optional[np.ndarray]=None, max_dim = 2000):
        import tensorflow_quantum as tfq
        state_layer = tfq.layers.State(backend=self.backend)
        num_parameters = len(self.raw_symbols) 
        if num_parameters > 0:
            if x is None:
                raise ValueError('Parameter values are required for a parameterized circuit')    
            ndim = x.ndim
            if ndim != 2:
                raise ValueError('Parameter values must be a 2D array')
            if num_parameters != x.shape[1]:
                raise ValueError('Parameter dimension does not match the number of symbols in parameterized circuit')
            symbols, resolved_values = self._get_resolved_symbols_and_values(x)
            dim = len(resolved_values)
            state_vectors = np.zeros((dim, 2**self.n_qubit), dtype=np.complex64)
            for i, symbol_values in enumerate(np.array_split(resolved_values, math.ceil(dim/max_dim))):
                split_dim = len(symbol_values)
                states = state_layer(self, symbol_names=symbols, symbol_values=symbol_values)
                for j, state in enumerate(states):
                    state_vectors[i*max_dim+j] = state.numpy().copy()
                    state = None
        else:
            if x is not None:
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
            