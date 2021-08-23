from typing import List
import re
import copy
import cirq
import quple
import numpy as np
import sympy as sp
import numba
from concurrent.futures  import ProcessPoolExecutor


def replace_symbol_in_op(op, old_symbol:sp.Symbol, new_symbol:sp.Symbol) -> None:
    """Replace symbols in a parameterised gate operation with new symbols
    
    Args:
        op: gate operation to which the associated symbols are replaced
        old_symbol: the original symbol associated with the gate operation
        new_symbol: the new symbol which the original symbol is to be replaced by
    """
    if isinstance(op, cirq.EigenGate):
        if old_symbol in op.exponent.free_symbols:
            op._exponent = op.exponent.subs(old_symbol, new_symbol)

    if isinstance(op, cirq.FSimGate):
        if isinstance(op.theta, sympy.Basic):
            if old_symbol in op.theta.free_symbols:
                op._theta = op.theta.subs(old_symbol, new_symbol)
        if isinstance(op.phi, sympy.Basic):
            if old_symbol in op.phi.free_symbols:
                op._phi = op.phi.subs(old_symbol, new_symbol)

    if isinstance(op, cirq.PhasedXPowGate):
        if isinstance(op.exponent, sympy.Basic):
            if old_symbol in op.exponent.free_symbols:
                op._exponent = op.exponent.subs(old_symbol, new_symbol)
        if isinstance(op.phase_exponent, sympy.Basic):
            if old_symbol in op.phase_exponent.free_symbols:
                op._phase_exponent = op.phase_exponent.subs(old_symbol, new_symbol)
                
def resolve_expression_map_conflicts(old_expr_map, new_expr_map):
    """Recovers reflattened expressions from a new expression map given the old expression map
    
    Args:
        old_expr_map: cirq.ExpressionMap
            The original expression map which maps flattened expressions to unflattened expressions
        new_expr_map: cirq.ExpressionMap
            The new expression map which is possibly reflattened from a flattened expressions
    Returns:
        An expression map with reflattened expressions recovered
    """    
    inverted_map = {v:k for k,v in old_expr_map.items()}
    reflattened = set(inverted_map).intersection(set(new_expr_map.values()))
    expr_map = cirq.ExpressionMap({(k if v not in reflattened else inverted_map[v]):v 
                                   for k,v in new_expr_map.items()})
    return expr_map
                
def pqc_symbol_map(circuit:cirq.Circuit, symbols_map) -> cirq.Circuit:
    """Maps the old symbols in a circuit with the new ones
    
    Args:
        circuit: cirq.Circuit, quple.QuantumCircuit
            The circuit to map the parameter symbols
        symbols_map:
            A dictionary that maps old parameter symbols to new ones
            
    Returns:
        A new circuit with the old parameter symbols replaced by the new ones
    """
    new_circuit = copy.deepcopy(circuit)
    for moment in new_circuit:
        for op in moment:
            if cirq.is_parameterized(op):
                symbols_in_op = quple.symbols_in_op(op.gate)
                for sym in symbols_in_op:
                    replace_symbol_in_op(op.gate, sym, symbols_map[sym])
    return new_circuit



def merge_pqc(circuits:List[cirq.Circuit], symbol:str='θ') -> cirq.Circuit:
    """Merges a list of parameterized circuit and updates the parameter symbols
    
    Circuits are merged in the order they are listed. The set of symbols in one circuit 
    will be treated as distinct from the set of symbols in another circuit. The merged
    circuit will have all parameter symbols of the form {symbol}_{index}. 
    
    Args:
        circuits: list of cirq.Circuit or quple.QuantumCircuit
            The circuits to merge
        symbol: str
            The parameter symbol prefix for the merged circuit

    Returns:
        The merged circuit
    """        
    symbol_size = 0
    if not all(isinstance(circuit, cirq.Circuit) for circuit in circuits):
        raise ValueError('Circuits to be merged must be intances of cirq.Circuit object')
    circuits = [circuit.get_unflattened_circuit() if isinstance(circuit, quple.QuantumCircuit) \
                else circuit for circuit in circuits]
    for circuit in circuits:
        symbol_size += len(quple.get_circuit_symbols_in_order(circuit))
    all_symbols = sp.symarray(symbol, symbol_size)
    qubits = set()
    for circuit in circuits:
        qubits |= set(quple.get_circuit_qubits(circuit))
    qubits = sorted(list(qubits))
    merged_circuit = quple.QuantumCircuit(qubits)
    for circuit in circuits:
        old_symbols = quple.get_circuit_symbols_in_order(circuit)
        new_symbols = all_symbols[:len(old_symbols)]
        all_symbols = all_symbols[len(old_symbols):]
        symbols_map = {old:new for old, new in zip(old_symbols, new_symbols)}
        merged_circuit.append(pqc_symbol_map(circuit, symbols_map))
    return merged_circuit

def plot_stacked_roc_curve(fpr_list:List[np.ndarray], 
                   tpr_list:List[np.ndarray], labels:List[str]=None, title:str='ROC Curve', with_auc:bool=True):
    """Plots multiple roc curves stacked together
    
    Args:
        fpr: list/array of numpy array
            A collection of arrays containing the false positive rates for 
            different experiments
        tpr: list/array of numpy array
            A collection of arrays containing the false positive rates for 
            different experiments
        labels: list of str, default = None
            List of labels for different experiments.
        title: str, default = 'ROC Curve'
            Title of the plot
        with_auc: True or False
            Whether to include the auc in the labels

    Returns:
        A matplotlib plot of the stacked roc curve
    """     
    assert len(fpr_list) == len(tpr_list)
    n = len(fpr_list)
    if labels is None:
        labels = ['']*n
    assert len(labels) == n
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    plt.clf()
    plt.rcParams["font.weight"] = "bold"
    plt.xlabel("Signal Efficiency", fontsize=18,fontweight='bold')
    plt.ylabel("Background Rejection", fontsize=18,fontweight='bold')
    plt.title(title, fontsize=16,fontweight='bold')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(color='gray', linestyle='--', linewidth=1)  
    for fpr,tpr,label in zip(fpr_list, tpr_list, labels):
        if with_auc:
            roc_auc = auc(fpr, tpr)
            label += ', AUC={:.4f}'.format(roc_auc)
        plt.plot(tpr,1-fpr, linestyle='-',label=label,linewidth=2)
    plt.plot([0, 1], [1, 0], linestyle='--', color='black', label='Luck, AUC= 0.5')
    plt.legend(loc='best',prop={'size': 8})    
    return plt       


def plot_roc_curve(fpr:np.ndarray, tpr:np.ndarray, label:str='', title:str='ROC Curve', with_auc:bool=True):
    """Plots a roc curve
    
    Args:
        fpr: numpy array
            An array containing the false positive rates
        tpr: numpy array
            An array containing the true positive rates
        label: str
            Label of the curve
        title: str
            Title of the plot
        with_auc: True or False
            Whether to include the auc in the labels
    
    Returns:
        A matplotlib plot of the roc curve
    """        
    return plot_stacked_roc_curve([fpr], [tpr], [label], title=title, with_auc=with_auc)    

def atoi(symbol):
    return int(symbol) if symbol.isdigit() else symbol

def natural_key(symbol):
    '''Keys for human sorting
    Reference:
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [ atoi(s) for s in re.split(r'(\d+)', symbol.name) ]

def get_unique_symbols(symbols, sort_key=natural_key):
    unique_symbols = set(symbols)
    return sorted(list(unique_symbols), key=sort_key)

def parallel_run(func, *iterables, max_workers=None):
    max_workers = max_workers or quple.MAX_WORKERS 

    with ProcessPoolExecutor(max_workers) as executor:
        result = executor.map(func, *iterables)

    return [i for i in result]

def execute_multi_tasks(func, *iterables, parallel):
    if parallel == 0:
        result = []
        for args in zip(*iterables):
            result.append(func(*args))
        return result
    else:
        if parallel == -1:
            max_workers = get_cpu_count()
        else:
            max_workers = parallel
        return parallel_run(func, *iterables, max_workers=max_workers)


def batching(l:List, n:int):
    for i in range(0, len(l), n):
        yield l[i:i + n]
    
def flatten_list(l):
    return [item for sublist in l for item in sublist]