from typing import List, Union, Optional, Callable, Sequence, Tuple

from quple import ParameterisedCircuit

class RealAmplitudes(ParameterisedCircuit):
    '''The real amplitudes parameterised circuit
    
    Implementation based on the Qiskit library.
    
    The RealAmplitudes circuit consists of a layer
    of Pauli Y rotations acting on each qubit followed by
    a layer of CNOT entanglement on pairs of qubits under
    a given interaction graph. The resultant quantum state
    from the RealAmplitudes circuit will only have real
    amplitudes with 0 complex part. 
    
    The RealAmplitudes circuit may be used as a variational
    circuit or a model circuit in the PQC layer of a machine 
    learning model. It represents a unitary U(θ) parameterised 
    by a set of free  parameters θ. The circuit may be repeated
    several times to  increase the number of parameters and 
    hence the complexity of the model. 
    
    It is an example of a strongly entangling circuit which has 
    the advantage of capturing correlations between the data
    features at all ranges for a short range circuit. 
    Reference: https://arxiv.org/pdf/1804.00633.pdf

    Args:
       n_qubit: int
            Number of qubits in the circuit
       copies: int
           Number of times the layers are repeated.
       entangle_strategy: Determines how the qubits are connected in an entanglement block.
           If str, it specifies the name of the strategy.
           If callable, it specifies the function to map to an interaction graph.
           If list of str, it specifies the names of a list of strategies. The 
           strategy to use is decided by the current block index. For example, if
           the circuit is building the n-th entanglement block in the entanglement layer,
           then the n-th strategy in the list will be used.
           If list of callable, it specifies the list of functions to map to an interaction
           graph. The function to use is decided by the current block index.
       parameter_symbol: str
           Symbol prefix for circuit parameters.
       name: str
           Name of the circuit.
    '''    
    def __init__(self, n_qubit: int, copies: int=2, 
                 entangle_strategy:Optional[Union[str,List[str], Callable[[int,int],List[Tuple[int]]],
                                                 List[Callable[[int,int],List[Tuple[int]]]]]]=None,
                 parameter_symbol:str='θ', name:str='RealAmplitudes', *args, **kwargs):
        super().__init__(n_qubit=n_qubit, copies=copies,
                         rotation_blocks='RY',
                         entanglement_blocks='CX',
                         entangle_strategy=entangle_strategy,
                         parameter_symbol=parameter_symbol,
                         name=name,
                         final_rotation_layer=True,
                         flatten_circuit=False,
                         *args, **kwargs)