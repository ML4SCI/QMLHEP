from typing import List, Union, Optional, Callable, Sequence, Tuple

from quple import ParameterisedCircuit

class ExcitationPreserving(ParameterisedCircuit):
    '''The excitation preserving parameterised circuit
    
    Implementation based on the Qiskit library.
    
    The ExcitationPreserving circuit consists of layers 
    of the two qubit Fermionic simulation, or fSim, gate set. 
    Under this gate set, the σ_Xσ_X and σ_Yσ_Y couplings 
    between the qubits have equal coefficients which
    conserves the number of excitations of the qubits. 
    Algorithms performed with just Pauli Z rotations 
    and fSim gates enable error mitigation techiques
    including post selection and zero noise extrapolation.
    Referemce: https://arxiv.org/pdf/1805.04492.pdf
    
    Further reference: https://arxiv.org/pdf/2001.08343.pdf
    
    The ExcitationPreserving circuit may be used as a variational
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
      entanglement_gate: 'RISWAP' or 'FSim', default='RISWAP'
         Excitation preserving gate operation to use for entangling qubits
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
                 entanglement_gate:str='RISWAP',
                 entangle_strategy:Optional[Union[str,List[str], Callable[[int,int],List[Tuple[int]]],
                                                 List[Callable[[int,int],List[Tuple[int]]]]]]=None,
                 parameter_symbol:str='θ', name:str='ExcitationPreserving', *args, **kwargs):
        
        allowed_blocks = ['RISWAP', 'FSim']
        if entanglement_gate not in allowed_blocks:
            raise ValueError('Unsupported gate operation {}, choose one of '
                             '{}'.format(entanglement_gate, allowed_blocks))
 
        super().__init__(n_qubit=n_qubit, copies=copies,
                         rotation_blocks='RZ',
                         entanglement_blocks=entanglement_gate,
                         entangle_strategy=entangle_strategy,
                         parameter_symbol=parameter_symbol,
                         name=name,
                         final_rotation_layer=True,
                         flatten_circuit=False,
                         *args, **kwargs)