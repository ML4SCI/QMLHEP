from typing import List, Union, Optional, Callable, Sequence, Tuple

from quple import ParameterisedCircuit


class IsingCoupling(ParameterisedCircuit):
    '''The efficient SU2 parameterised circuit
    
    Implementation based on the Qiskit library.
    
    The IsingCoupling circuit consists of a layer of single-qubit
    rotation gates and a layer of two-qubit ising coupling gates.
    Such gate set is naturally implemented by trapped ions quantum
    computers.
    
    The IsingCoupling circuit may be used as a variational circuit
    or a model circuit in the PQC layer of a machine learning model.
    It represents a unitary U(θ) parameterised by a set of free 
    parameters θ. The circuit may be repeated several times to 
    increase the number of parameters and hence the complexity 
    of the model. 
    
    It is an example of a strongly entangling circuit which has 
    the advantage of capturing correlations between the data
    features at all ranges for a short range circuit. 
    Reference: https://arxiv.org/pdf/1804.00633.pdf
    
    Args:
      n_qubit: int
          Number of qubits in the circuit
      copies: int
         Number of times the layers are repeated.
      rotation_gates: default=['RY','RZ']
          The rotation gates to be used in the circuit.
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
    		     rotation_gates: Optional[Union[str, 'cirq.Gate', Callable, 
                                          List[str],List['cirq.Gate'],List[Callable]]]=None,
             ising_gates:Optional[Union[str, 'cirq.Gate', Callable, 
                                      List[str],List['cirq.Gate'],List[Callable]]]=None,
             entangle_strategy:Optional[Union[str,List[str], Callable[[int,int],List[Tuple[int]]],
                                             List[Callable[[int,int],List[Tuple[int]]]]]]=None,
             parameter_symbol:str='θ', name:str='IsingCoupling', *args, **kwargs):
        rotation_gates = rotation_gates or ['RY', 'RZ']
        ising_gates = ising_gates or ['XX']
        super().__init__(n_qubit=n_qubit, copies=copies,
                         rotation_blocks=rotation_gates,
                         entanglement_blocks=ising_gates,
                         entangle_strategy=entangle_strategy,
                         parameter_symbol=parameter_symbol,
                         name=name,
                         flatten_circuit=False,
                         final_rotation_layer=True,
                         *args, **kwargs)
