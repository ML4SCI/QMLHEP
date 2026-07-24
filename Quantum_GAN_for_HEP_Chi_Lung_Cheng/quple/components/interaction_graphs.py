from typing import List, Tuple
from math import sqrt
import itertools


"""
Interaction graphs determine how qubits are connected in terms of multi-qubit operations.
An interaction graph accepts two inputs, n and m. Here n refers to the number of qubits in
the system and m is the number of qubits involved in the multi-qubit interaction
"""
    
def nearest_neighbor(n:int, m:int=2) -> List[Tuple[int]]:
    """The nearest neighbor(linear) interaction
    
    In the nearest neighbor interaction, each qubit is connected to its
    next nearest neighbor in a linear manner. 
    
    Examples:
    # Nearest neighbor interaction graph for 4 qubit system with 2 qubit interaction
    >> nearest_neighbor(n=4, m=2)  
    [(0, 1), (1, 2), (2, 3)]
    # Building a circuit of 4 qubits with a layer of CNOT gates using
    # the nearest neighbor interaction graph
    >> cq = ParameterisedCircuit(n_qubit=4)
    >> cq.add_entanglement_layer(['CNOT'], entangle_strategy='linear')
    >> cq
    (0, 0): ───@───────────
               │
    (0, 1): ───X───@───────
                   │
    (0, 2): ───────X───@───
                       │
    (0, 3): ───────────X───
    
    
    Args:
        n: int
        Number of qubits in the system
        m: int
        Number of qubits involved in the multi-qubit interaction
        
    Returns:
        A list of tuples of qubit indices. Each tuple specifies the indices of the
        qubits that are involved in the multi-qubit interaction. 
    """
    return [tuple(range(i, i+m)) for i in range(n - m + 1)]    

def cyclic(n:int, m:int=2) -> List[Tuple[int]]:
    """The cyclic(circular) interaction
    
    In the cyclic interaction, each qubit is connected to its
    next nearest neighbor in a circular manner.

    Examples:
    # Cyclic interaction graph for 4 qubit system with 2 qubit interaction
    >> cyclic(n=4, m=2) 
    [(0, 1), (1, 2), (2, 3), (3, 0)]
    # Building a circuit of 4 qubits with a layer of CNOT gates using
    # the cyclic interaction graph
    >> cq = ParameterisedCircuit(n_qubit=4)
    >> cq.add_entanglement_layer(['CNOT'], entangle_strategy='cyclic')
    >> cq
    (0, 0): ───@───────────X───
               │           │
    (0, 1): ───X───@───────┼───
                   │       │
    (0, 2): ───────X───@───┼───
                       │   │
    (0, 3): ───────────X───@───
    
    
    Args:
        n: int
        Number of qubits in the system
        m: int
        Number of qubits involved in the multi-qubit interaction
        
    Returns:
        A list of tuples of qubit indices. Each tuple specifies the indices of the
        qubits that are involved in the multi-qubit interaction. 
    """    
    return nearest_neighbor(n,m) + [tuple(range(n - m + 1, n)) + (0,)]

def fully_connected(n:int, m:int=2) -> List[Tuple[int]]:
    """The fully-connected(full) interaction
    
    In the fully-connected interaction, every distinct unordered tuple of m qubits
    are connected exactly once. 
    
    Examples:
    # Fully connected interaction graph for 4 qubit system with 2 qubit interaction
    >> fully_connected(n=4, m=2)  
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    # Building a circuit of 4 qubits with a layer of CNOT gates using
    # the fully connected interaction graph
    >> cq = ParameterisedCircuit(n_qubit=4)
    >> cq.add_entanglement_layer(['CNOT'], entangle_strategy='full')
    >> cq
                       ┌──┐
    (0, 0): ───@───@────@─────────────
               │   │    │
    (0, 1): ───X───┼────┼@────@───────
                   │    ││    │
    (0, 2): ───────X────┼X────┼───@───
                        │     │   │
    (0, 3): ────────────X─────X───X───
                       └──┘
    
    Args:
        n: int
        Number of qubits in the system
        m: int
        Number of qubits involved in the multi-qubit interaction
        
    Returns:
        A list of tuples of qubit indices. Each tuple specifies the indices of the
        qubits that are involved in the multi-qubit interaction. 
    """        
    return list(itertools.combinations(list(range(n)), m))

def all_to_all(n:int, m:int=2) -> List[Tuple[int]]:
    """The all-to-all interaction
    
    In the fully-connected interaction, every distinct ordered tuple of m qubits
    are connected.
    
    Examples:
    # All to all interaction graph for 4 qubit system with 2 qubit interaction
    >> all_to_all(n=4, m=2)  
    [(0, 1),
     (0, 2),
     (0, 3),
     (1, 0),
     (1, 2),
     (1, 3),
     (2, 0),
     (2, 1),
     (2, 3),
     (3, 0),
     (3, 1),
     (3, 2)]
    # Building a circuit of 4 qubits with a layer of CNOT gates using
    # the all to all interaction graph
    >> cq = ParameterisedCircuit(n_qubit=4)
    >> cq.add_entanglement_layer(['CNOT'], entangle_strategy='all_to_all')
    >> cq
                                   ┌──┐
    (0, 0): ───@───@───@───X─────────X────────────X───────────
               │   │   │   │         │            │
    (0, 1): ───X───┼───┼───@───@────@┼────X───────┼───X───────
                   │   │       │    ││    │       │   │
    (0, 2): ───────X───┼───────X────┼@────@───@───┼───┼───X───
                       │            │         │   │   │   │
    (0, 3): ───────────X────────────X─────────X───@───@───@───
                                   └──┘

    Args:
        n: int
        Number of qubits in the system
        m: int
        Number of qubits involved in the multi-qubit interaction
        
    Returns:
        A list of tuples of qubit indices. Each tuple specifies the indices of the
        qubits that are involved in the multi-qubit interaction. 
    """        
    return list(itertools.permutations(list(range(n)), m))

def star(n:int, m:int=2) -> List[Tuple[int]]:
    """The star interaction
    
    In the star interaction, the first qubit is connected to every other qubit.
    
    Examples:
    # Star interaction graph for 4 qubit system with 2 qubit interaction
    >> star(n=4, m=2)  
    [(0, 1), (0, 2), (0, 3)]
    # Building a circuit of 4 qubits with a layer of CNOT gates using
    # the fully connected interaction graph
    >> cq = ParameterisedCircuit(n_qubit=4)
    >> cq.add_entanglement_layer(['CNOT'], entangle_strategy='star')
    >> cq
    (0, 0): ───@───@───@───
               │   │   │
    (0, 1): ───X───┼───┼───
                   │   │
    (0, 2): ───────X───┼───
                       │
    (0, 3): ───────────X───
    
    Args:
        n: int
        Number of qubits in the system
        m: int
        Number of qubits involved in the multi-qubit interaction
        
    Returns:
        A list of tuples of qubit indices. Each tuple specifies the indices of the
        qubits that are involved in the multi-qubit interaction. 
    """        
    if m > 2:
        raise ValueError('the connectiviy graphs of the "star" method'
                         ' requires 2 qubits in an interaction unit'
                        ' but {} is given'.format(m))
    if m == 2:
        return [(0, i+1) for i in range(n - m + 1)]
    else:
        return [(i,) for i in range(n)]

def alternate_linear(n:int, m:int=2) -> List[Tuple[int]]:
    """The alternate linear interaction
    
    In the alternate, all neiboring qubits are connected but in an alternating manner
    
    Examples:
    # Alternate linear graph for 5 qubit system with 2 qubit interaction
    >> alternate_linear(n=5, m=2)  
    [(0, 1), (2, 3), (1, 2), (3, 4)]
    # Building a circuit of 5 qubits with a layer of CNOT gates using
    # the alternate linear interaction graph
    >> cq = ParameterisedCircuit(n_qubit=4)
    >> cq.add_entanglement_layer(['CNOT'], entangle_strategy='alternate_linear')
    >> cq
    (0, 0): ───@───────
               │
    (0, 1): ───X───@───
                   │
    (0, 2): ───@───X───
               │
    (0, 3): ───X───@───
                   │
    (0, 4): ───────X───
    
    Args:
        n: int
        Number of qubits in the system
        m: int
        Number of qubits involved in the multi-qubit interaction
        
    Returns:
        A list of tuples of qubit indices. Each tuple specifies the indices of the
        qubits that are involved in the multi-qubit interaction. 
    """       
    if m > 2:
        raise ValueError('The "alternate linear" connectivity graphs requires <= 2 '
                         'qubits in the interaction unit'
                        ' but {} is given'.format(m))
    if n < 2:
        raise ValueError('The "alternate linear" connectivity graphs requires at least '
                         '2 qubits in the circuit')
    if m == 2:
        return [(i, i+1) for i in range(0, n - 1, m)] + [(i+1, i+2) for i in range(0, n - 2, m)]  
    else:
        return [(i,) for i in range(n)]
    
def alternate_cyclic(n:int, m:int=2) -> List[Tuple[int]]:
    """The alternate linear interaction
    
    In the alternate, all neiboring qubits as well as the first and last qubit 
    are connected in an alternating manner
    
    Examples:
    # Alternate cyclic graph for 5 qubit system with 2 qubit interaction
    >> alternate_cyclic(n=5, m=2)  
    [(0, 1), (2, 3), (1, 2), (3, 4), (4, 0)]
    # Building a circuit of 5 qubits with a layer of CNOT gates using
    # the alternate linear interaction graph
    >> cq = ParameterisedCircuit(n_qubit=4)
    >> cq.add_entanglement_layer(['CNOT'], entangle_strategy='alternate_linear')
    >> cq
    (0, 0): ───@───────X───
               │       │
    (0, 1): ───X───@───┼───
                   │   │
    (0, 2): ───@───X───┼───
               │       │
    (0, 3): ───X───@───┼───
                   │   │
    (0, 4): ───────X───@───
    
    Args:
        n: int
        Number of qubits in the system
        m: int
        Number of qubits involved in the multi-qubit interaction
        
    Returns:
        A list of tuples of qubit indices. Each tuple specifies the indices of the
        qubits that are involved in the multi-qubit interaction. 
    """   
    if m > 2:
        raise ValueError('The "alternate linear" connectivity graphs requires <= 2 '
                         'qubits in the interaction unit'
                        ' but {} is given'.format(m))
    if n < 2:
        raise ValueError('The "alternate linear" connectivity graphs requires at least '
                         '2 qubits in the circuit')
    if m == 2:
        return [(i, i+1) for i in range(0, n - 1, m)] + [(i+1, i+2) for i in range(0, n - 2, m)] + [tuple(range(n - m + 1, n)) + (0,)]
    else:
        return [(i,) for i in range(n)]   
    
def filter_mesh(n:int, m:int=2) -> List[Tuple[int]]:   
    #check if number of qubit is a square number
    dimension = sqrt(n)
    if not dimension.is_integer():
        raise ValueError('The "filter_mesh" connectivity graphs requires number of qubits being a square number')
    dimension = int(dimension)
    
    horizontal_filter = [(dimension*i+j, dimension*i+j+1) for i in range(dimension) for j in range(dimension-1)]
    vertical_filter = [(dimension*j+i, dimension*(j+1)+i) for i in range(dimension) for j in range(dimension-1)]
    return horizontal_filter+vertical_filter

def pool_mesh(n:int, m:int=2) -> List[Tuple[int]]:   
    #check if number of qubit is a square number
    dimension = sqrt(n)
    if not dimension.is_integer():
        raise ValueError('The "pool_mesh" connectivity graphs requires number of qubits being a square number')
    dimension = int(dimension)
    if dimension % 2 != 0:
        raise ValueError('The "pool_mesh" connectivity graphs requires number of qubits being a square of an even number')
    
    horizontal_filter = [(dimension*i+j, dimension*i+j+1) for i in range(dimension) for j in range(0, dimension-1, 2)]
    vertical_filter = [(dimension*j+i, dimension*(j+1)+i) for i in range(dimension) for j in range(0, dimension-1, 2)]
    return horizontal_filter+vertical_filter

# dictionary for mapping a str to the corresponding interaction graph
interaction_graph = {
    'nearest_neighbor': nearest_neighbor,
    'linear': nearest_neighbor,
    'cyclic': cyclic,
    'circular': cyclic,
    'full': fully_connected,
    'fully_connected': fully_connected,
    'all_to_all': all_to_all,
    'star': star,
    'alternate_linear': alternate_linear,
    'alternate_cyclic': alternate_cyclic,
    'filter_mesh': filter_mesh,
    'pool_mesh': pool_mesh
}