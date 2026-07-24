from abc import ABC, abstractmethod
from typing import Optional, List, Union, Callable, Tuple
import numpy as np
import sympy as sp

import cirq

from quple import ParameterisedCircuit, TemplateCircuitBlock
from quple.data_encoding.encoding_maps import encoding_map_registry

class EncodingCircuit(ParameterisedCircuit, ABC):
    '''The data encoding circuit (or state preparation circuit)
    
    A quantum circuit for encoding classical data into a quantum state by embedding the
    data in the parameters of unitary operations on the circuit qubits using a suitable encoding function.
    
    If the qubit encoding method is used, the number of qubits in the encoding circuit should match the 
    feature dimesion of the input data.
    
    '''
    def __init__(self, feature_dimension:int, copies:int=1,
                rotation_blocks:Optional[Union[str, cirq.Gate, Callable, TemplateCircuitBlock,
                                                               List[str],List[cirq.Gate],List[Callable],
                                                               List[TemplateCircuitBlock]]]='H',
                entanglement_blocks:Optional[Union[str, cirq.Gate, Callable, TemplateCircuitBlock,
                                               List[str],List[cirq.Gate],List[Callable],
                                               List[TemplateCircuitBlock]]]=None,
                entangle_strategy:Optional[Union[str,List[str], Callable[[int,int],List[Tuple[int]]],
                                                 List[Callable[[int,int],List[Tuple[int]]]]]]=None,
                encoding_map: Optional[Union[str, Callable]]=None,
                parameter_symbol:str='x',
                parameter_scale=sp.pi,
                name:str='EncodingCircuit',
                flatten_circuit=True,
                *args, **kwargs):
        '''Create a new Pauli expansion circuit.
        Args:
            feature_dimension: dimension of data to be encoded (=number of qubits in the circuit)
            copies: the number of repetition of the encoding circuit
            encoding_map: data mapping function from R^(feature_dimension) to R
            parameter_symbol: the symbol prefix for the encoded data
            name: name of circuit
            flatten_circuit: whether or not to flatten the circuit
        '''
        if isinstance(feature_dimension, int):
            self._feature_dimension = feature_dimension
        else:
            self._feature_dimension = len(feature_dimension)
        self.encoding_map = encoding_map
    
        super().__init__(n_qubit=feature_dimension, copies=copies,
                         rotation_blocks=rotation_blocks,
                         entanglement_blocks=entanglement_blocks,
                         entangle_strategy=entangle_strategy,
                         parameter_symbol=parameter_symbol,
                         final_rotation_layer=False, 
                         flatten_circuit=flatten_circuit,
                         parameter_scale=parameter_scale,
                         reuse_param_per_layer=True,
                         name=name, *args, **kwargs)
        
        
    def get_parameters(self, indices:List[int]):
        '''Obtain parameter symbols corresponding to the array indices
        Args:
            indices: array indices of the parameter
        Return:
            a numpy array of parameter symbols
        Example:    
        >>>  cq.get_parameters([1, 2, 3])
        array([x_1, x_2, x_3], dtype=object)
        '''
        if self.num_param == 0:
            self.new_param(size=self.feature_dimension)
        if isinstance(indices, tuple):
            indices = list(indices)
        return self.parameters[indices]

    def encode_parameters(self, indices:List[int], encoding_map:Callable=None):
        '''Obtain the encoded value for the given parameters using the encoding map
        Args:
            indices: indices of parameter array which corresponds to the indices of qubits in the circuit
        Returns:
            A symbolic expression of the encoded data
        Example:
        >>> from quple.data_encoding import PauliExpansion
        >>> cq = PauliExpansion(feature_dimension=5, copies=1, encoding_map='self_product')
        >>> cq.encode_parameters(indices=[0,2,4])
        >>> 𝑥0𝑥2𝑥4
        '''
        encoding_map = encoding_map or self.encoding_map
        parameters = self.get_parameters(indices)
        encoded_value = self.parameter_scale*encoding_map(parameters)
        return encoded_value
    
    def get_total_parameter_count(self) -> int:
        return self._feature_dimension
    
    @property
    def feature_dimension(self):
        return self._feature_dimension
    
    @property
    def encoding_map(self):
        return self._encoding_map
    
    @encoding_map.setter
    def encoding_map(self, val):
        if val is None:
            val = encoding_map_registry['self_product']
        elif isinstance(val, str):
            val = encoding_map_registry[val]
        self._encoding_map = val