from typing import Callable, Optional
import numpy as np
import sympy as sp

from quple.data_encoding.general_pauli_z_encoding import GeneralPauliZEncoding

class SecondOrderPauliZEncoding(GeneralPauliZEncoding):
    """The second order Pauli Z encoding circuit
    
    An encoding circuit consisting of layers of unitary operators of the
    form exp(iψ(x)Z⊗Z)H^{⊗n} where ψ is a data encoding function, Z⊗Z is the 
    tensor product of two Pauli Z operators and x = (x_1, . . . , x_n ) are 
    the input features to be encoded.
    
    To encode data of feature dimension n, a set of general Pauli operators 
    are chosen to encode the data into an n qubit circuit. Each Pauli operator
    will contribute to a unitary operation \exp(i\sum_{s\in S}ψ_s(x_s)Z_s) where
    S is a set of 2-tuple of qubit indices determined by the interaction graph. 
    For a fully connected graph, S is the set of combinations of 2-qubit pairs.
    Then the unitary operation is 
    \exp(i\sum{s=(j, k) in S}ψ_{j,k}(x_j, x_k)Z_j⊗Z_k)H^{⊗n}
    
    The Pauli encoding uses the circuit-centric design with alternating
    rotation and entanglement layer to achieve a strongly entangling circuit. 
    Ref: https://arxiv.org/pdf/1804.00633.pdf   

    Examples:
    >> cq = SecondOrderPauliZEncoding(feature_dimension=3, copies=1)
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
        name: str
            Name of the encoding circuit
    """
    def __init__(self, feature_dimension: int,
                 copies:int=2, 
                 encoding_map:Optional[Callable[[np.ndarray], float]] = None,
                 global_shift:bool=False, 
                 parameter_symbol:str='x',
                 parameter_scale=sp.pi,
                 name:str='SecondOrderPauliZEncoding', *args, **kwargs):
        '''Creates the second order Pauli Z encoding circuit
        
        Args:
            feature_dimension: int
                Dimension of data to be encoded (=number of qubits in the circuit)
            copies: int
                Number of repetition of the encoding circuit
            encoding_map: function that maps a numpy array to a number
                Data mapping function from R^(feature_dimension) to R
            global_shift: bool
                Include global shift to RZ gate                
            name: str
                Name of the encoding circuit
        '''         
        super().__init__(feature_dimension, copies=copies, z_order=2,
                         encoding_map=encoding_map, global_shift=global_shift,
                         parameter_symbol=parameter_symbol,
                         parameter_scale=parameter_scale,
                         name=name, *args, **kwargs)