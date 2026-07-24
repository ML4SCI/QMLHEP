from typing import Callable, Optional
import numpy as np
import sympy as sp

from quple.data_encoding.general_pauli_encoding import GeneralPauliEncoding

class GeneralPauliZEncoding(GeneralPauliEncoding):
    """The general Pauli Z encoding circuit
    
    An encoding circuit consisting of layers of unitary operators of the
    form exp(iψ(x)Z^{⊗k})H^{⊗n} where ψ is a data encoding function, Z^{⊗k}
    is a k fold tensor product of Pauli Z operator for some k <= n on n qubits,
    and x = (x_1, . . . , x_n ) are the input features to be encoded.
    
    To encode data of feature dimension n, a set of general Pauli operators 
    are chosen to encode the data into an n qubit circuit. Each Pauli operator
    will contribute to a unitary operation \exp(i\sum_{s\in S}ψ_s(x_s)Σ_s)H^{⊗n}
    where s is the indices of a subset of all qubit indices S. For a general Pauli 
    operator of order k, s is a tuple of k elements. 
    
    For example, for k = 1 case S is the set {1, 2, ..., n} and 
    the unitary operation is \exp(i\sum_{j=1}^n ψ_j(x_j)Z_j)H^{⊗n}, 
    where Z_j is the Pauli Z operator acting on the j-th qubit. 
    
    If instead k = 2, then S is a set of 2-tuple of
    qubit indices determined by the interaction graph. For a fully connected
    graph, S is the set of combinations of 2-qubit pairs. Then the unitary operation
    is \exp(i\sum{s=(j, k) in S}ψ_{j,k}(x_j, x_k)Z_j⊗Z_k)H^{⊗n}
    
    The Pauli encoding uses the circuit-centric design with alternating
    rotation and entanglement layer to achieve a strongly entangling circuit. 
    Ref: https://arxiv.org/pdf/1804.00633.pdf    

    Examples:
    >> cq = GeneralPauliZEncoding(feature_dimension=3, z_order=2, copies=1)
    >> cq
    (0, 0): ───H───Rz(pi*x_0)───@──────────────────────@───@──────────────────────@──────────────────────────────
                                │                      │   │                      │
    (0, 1): ───H───Rz(pi*x_1)───X───Rz(pi*<x_0*x_1>)───X───┼──────────────────────┼───@──────────────────────@───
                                                           │                      │   │                      │
    (0, 2): ───H───Rz(pi*x_2)──────────────────────────────X───Rz(pi*<x_0*x_2>)───X───X───Rz(pi*<x_1*x_2>)───X───
    
    Args:
        feature_dimension: int
            Dimension of data to be encoded (=number of qubits in the circuit)
        copies: int
            Number of repetition of the encoding circuit
        encoding_map: function that maps a numpy array to a number
            Data mapping function from R^(feature_dimension) to R
        z_order: int
            Order of pauli z operations to be performed on each circuit block
        name: str
            Name of the encoding circuit
    """
    def __init__(self, feature_dimension: int,
                 copies:int=2, z_order:int=2,
                 encoding_map:Optional[Callable[[np.ndarray], float]] = None,
                 global_shift:bool=False,
                 parameter_symbol:str='x',
                 parameter_scale=sp.pi,
                 name:str='GeneralPauliZEncoding', *args, **kwargs):
        '''Creates the general Pauli Z encoding circuit

        Args:
            feature_dimension: int
                Dimension of data to be encoded (=number of qubits in the circuit)
            copies: int
                Number of repetition of the encoding circuit
            z_order: int
                Order of pauli z operations to be performed on each circuit block
            encoding_map: function that maps a numpy array to a number
                Data mapping function from R^(feature_dimension) to R
            global_shift: bool
                Include global shift to RZ gate
            name: str
                Name of the encoding circuit
        '''      
        paulis = []
        for i in range(1, z_order + 1):
            paulis.append('Z' * i)
        super().__init__(feature_dimension, copies=copies, paulis=paulis,
                         encoding_map=encoding_map, global_shift=global_shift,
                         parameter_symbol=parameter_symbol,
                         parameter_scale=parameter_scale,
                         name=name, *args, **kwargs)