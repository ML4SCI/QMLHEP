import cirq
import numpy as np
import sympy as sp
from itertools import combinations
from sympy import default_sort_key

import quple
from quple.utils.utils import natural_key

#reference: https://github.com/tensorflow/quantum/blob/v0.3.0/tensorflow_quantum/python/util.py
def symbols_in_op(op):
    """Returns the set of symbols associated with a parameterized gate operation.
    
    Arguments:
        op: cirq.Gate
            The parameterised gate operation to find the set of symbols associated with
    
    Returns:
        Set of symbols associated with the parameterized gate operation
    """
    if isinstance(op, cirq.EigenGate):
        return op.exponent.free_symbols

    if isinstance(op, cirq.FSimGate):
        ret = set()
        if isinstance(op.theta, sp.Basic):
            ret |= op.theta.free_symbols
        if isinstance(op.phi, sp.Basic):
            ret |= op.phi.free_symbols
        return ret

    if isinstance(op, cirq.PhasedXPowGate):
        ret = set()
        if isinstance(op.exponent, sp.Basic):
            ret |= op.exponent.free_symbols
        if isinstance(op.phase_exponent, sp.Basic):
            ret |= op.phase_exponent.free_symbols
        return ret

    raise ValueError("Attempted to scan for symbols in circuit with unsupported"
                     " ops inside. Expected op found in tfq.get_supported_gates"
                     " but found: ".format(str(op)))
    
def symbols_in_expr_map(expr_map, to_str=False, sort_key=natural_key):
    """Returns the set of symbols in an expression map
    
    Arguments:
        expr_map: cirq.ExpressionMap
            The expression map to find the set of symbols in
        to_str: boolean, default=False
            Whether to convert symbol to strings
        sort_key: 
            Sort key for the list of symbols
    Returns:
        Set of symbols in the experssion map
    """    
    all_symbols = set()
    for expr in expr_map:
        if isinstance(expr, sp.Basic):
            all_symbols |= expr.free_symbols
    sorted_symbols = sorted(list(all_symbols), key=sort_key)
    if to_str:
        return [str(x) for x in sorted_symbols]
    return sorted_symbols

def get_circuit_unflattened_symbols(circuit:'quple.QuantumCircuit',
                                    to_str=True,
                                    sort_key=natural_key):
    """Returns a list of unflattened parameter symbols in a circuit
    
    Arguments:
        circuit: quple.QuantumCircuit
            The circuit to find the unflattened parameter symbols in
        to_str: boolean, default=True
            Whether to convert symbol to strings
        sort_key:
            Sort key for the list of symbols
    Returns:
        List of unflattened parameter symbols in a circuit
    """      
    if isinstance(circuit, quple.QuantumCircuit):
        expr_map = circuit.expr_map
        if expr_map is not None:
            symbols = quple.symbols_in_expr_map(expr_map, 
                                                to_str=to_str,
                                                sort_key=sort_key)
        else:
            symbols = quple.get_circuit_symbols(circuit,
                                                to_str=to_str,
                                                sort_key=sort_key)
    else:
        symbols = quple.get_circuit_symbols(circuit,
                                            to_str=to_str,
                                            sort_key=sort_key)
    return symbols

#reference: https://github.com/tensorflow/quantum/blob/v0.3.0/tensorflow_quantum/python/util.py
def get_circuit_symbols(circuit, to_str=True, sort_key=natural_key):
    """Returns a list of parameter symbols in a circuit
    
    Arguments:
        circuit: cirq.Circuit, quple.QuantumCircuit
            The circuit to find the associated parameter symbols
        to_str: boolean, default=True
            Whether to convert symbol to strings
        sort_key:
            Sort key for the list of symbols
    Returns:
        A list of symbols in the circuit
    """      
    all_symbols = set()
    for moment in circuit:
        for op in moment:
            if cirq.is_parameterized(op):
                all_symbols |= symbols_in_op(op.gate)
    sorted_symbols = sorted(list(all_symbols), key=sort_key)
    if to_str:
        return [str(x) for x in sorted_symbols]
    return sorted_symbols

def get_circuit_qubits(circuit):
    """Returns a list of qubits in a circuit
    
    Arguments:
        circuit: cirq.Circuit, quple.QuantumCircuit
            The circuit to find the associated qubits
    Returns:
        A list of qubits in the circuit
    """      
    all_qubits = set()
    for moment in circuit:
        for op in moment:
            all_qubits |= set(op._qubits)
    return sorted(list(all_qubits))

def get_circuit_symbols_in_order(circuit, to_str=False):
    """Returns a list of parameter symbols in a circuit in order of creation
    
    Arguments:
        circuit: cirq.Circuit, quple.QuantumCircuit
            The circuit to find the associated parameter symbols
        to_str: boolean, default=False
            Whether to convert the sympy symbols to strings
    Returns:
        A list of symbols in the circuit in order of creation
    """        
    all_symbols = set()
    symbols_in_order = []
    for moment in circuit:
        for op in moment:
            if cirq.is_parameterized(op):
                new_symbols = symbols_in_op(op.gate)
                symbols_in_order += (new_symbols - all_symbols)
                all_symbols |= symbols_in_op(op.gate)
    if to_str:
        symbols_in_order = [str(symbol) for symbol in symbols_in_order]
    return symbols_in_order

def sample_final_states(circuit, samples=1, data=None, backend=None):
    """Samples the final states of a circuit
    
    If the circuit has parameterised gate operation, random values of
    the symbol values in the range (0, 2π) will be assigned to the gate operation.
    If `data` is given, the symbol values will be assigned according to the given data.
    
    Arguments:
        circuit: cirq.Circuit, quple.QuantumCircuit
            The circuit to sample the final states
        samples: int
            Number of samples, default=1
        data: real numbers with shape
            [batch_size, n_params] specifying parameter values to resolve
            into the circuits, following the natural ordering of symbols, e.g. x_1 < x_2 < x_10
    
    Returns:
        A list of sampled final states of the circuit
    """      
    if isinstance(circuit, cirq.Circuit) and not isinstance(circuit, quple.QuantumCircuit):
        circuit = QuantumCircuit.from_cirq(circuit)
    n_symbols = len(circuit.raw_symbols)
    if (n_symbols > 0) and (data is None):
        data = np.random.rand(samples, n_symbols)*2*np.pi
    return circuit.get_state_vectors(data)

def sample_density_matrices(circuit, samples=1, data=None, backend=None):
    """Samples the density matrices of a circuit
    
    If the circuit has parameterised gate operation, random values of
    the symbol values in the range (0, 2π) will be assigned to the gate operation.
    If `data` is given, the symbol values will be assigned according to the given data.
    
    Arguments:
        circuit: cirq.Circuit, quple.QuantumCircuit
            The circuit to sample the density matrices
        samples: int, default=1
            Number of samples
        data: real numbers with shape
            [batch_size, n_params] specifying parameter values to resolve
            into the circuits, following the natural ordering of symbols, e.g. x_1 < x_2 < x_10
    
    Returns:
        A list of sampled density matrices of the circuit
    """       
    final_states = sample_final_states(circuit, samples, data=data, backend=backend)
    density_matrices = [cirq.density_matrix_from_state_vector(fs) for fs in final_states] 
    return density_matrices


def sample_fidelities(circuit, samples=1, data=None, backend=None):
    """Samples the fidelities between two sampled final states of a circuit
    
    Two independent set of samples of circuit final states are first generated.
    The two set of samples are then paired according to the indices of the set.
    The fidelity between the final states in each pair of the samples is then calculated
    to obtain the sampled fidelties of the circuit. 
    
    If the circuit has parameterised gate operation, random values of
    the symbol values in the range (0, 2π) will be assigned to the gate operation.
    If `data` is given, the symbol values will be assigned according to the given data.
    
    Arguments:
        circuit: cirq.Circuit, quple.QuantumCircuit
            The circuit to sample the fidelities
        samples: int, default=1
            Number of samples
        data: real numbers with shape
            [batch_size, n_params] specifying parameter values to resolve
            into the circuits, following the ordering dictated by `symbol_names`
    
    Returns:
        A list of sampled fidelities between two sampled final states of a circuit
    """           
    sample_states_1 = sample_final_states(circuit, samples, data=data, backend=backend)
    sample_states_2 = sample_final_states(circuit, samples, data=data, backend=backend)
    fidelities = []
    for s1, s2 in zip(sample_states_1, sample_states_2):
        fidelities.append(cirq.fidelity(s1, s2))
    return fidelities

def circuit_fidelity_pdf(circuit, samples=3000, bins=100, data=None, backend=None):
    """Returns the binned probability density function from the sampled fidelties of a circuit
    
    If the circuit has parameterised gate operation, random values of
    the symbol values in the range (0, 2π) will be assigned to the gate operation.
    If `data` is given, the symbol values will be assigned according to the given data.    
    
    Arguments: 
        circuit: cirq.Circuit, quple.QuantumCircuit
            The circuit to sample the fidelities
        samples: int, default=3000
            Number of samples
        bins: int, default=100
            Number of bins for the fidelity pdf
        data: real numbers with shape
            [batch_size, n_params] specifying parameter values to resolve
            into the circuits, following the natural ordering of symbols, e.g. x_1 < x_2 < x_10
    Returns:
        A numpy array of the probability density function from the sampled fidelties of a circuit
    """      
    data = np.array(sample_fidelities(circuit, samples, data=data, backend=backend))
    pdf = np.histogram(data, bins=bins, range=(0,1), density=True)[0]
    pdf /= pdf
    return pdf


def get_data_Haar(n_qubit, samples=3000, bins=100):
    """Returns the fidelity values sampled from the Haar distribution
    
    Arguments:
        n_qubit: int
            Number of qubitts
        samples: int, default=3000
            Number of samples
        bins: int, default=100
            Number of bins of the Haar fidelity distribution
    Returns:
        A list of fidelity values sampled from the Haar distribution
    """      
    x = np.linspace(0., 1., bins)
    pdf = get_pdf_Haar(n_qubit, x)
    data = [np.random.choice(x, p=pdf) for _ in range(samples)]
    return data

def get_pdf_Haar(n_qubit, f_values):
    """Returns the Haar probability density function

    Arguments: 
        n_qubit: integer
            Number of qubits
        f_values: list/array of float/int
            A collection of fidelity values to be sampled 
    Returns:
        A numpy array of the Haar probability density function
    """          
    N = 2**n_qubit
    pdf = (N-1)*(1-f_values)**(N-2)
    pdf /= pdf.sum()
    return pdf

def circuit_fidelity_plot(circuit, samples=3000, bins=100, data=None, KL=True, epsilon=1e-10, backend=None):  
    """Returns a plot of the fidelity distribution for the final states sampled from a parameterised circuit overlayed with the Haar fidelity distrubiotn
    
    
    If the circuit has parameterised gate operation, random values of
    the symbol values in the range (0, 2π) will be assigned to the gate operation.
    If `data` is given, the symbol values will be assigned according to the given data.
    
    Arguments:
        circuit: cirq.Circuit, quple.QuantumCircuit
            The circuit to sample the fidelities
        samples: int
            Number of samples
        bins: int, default=100
            Number of bins for the fidelity pdf
        data: real numbers with shape
            [batch_size, n_params] specifying parameter values to resolve
            into the circuits, following the ordering dictated by `symbol_names`
        KL: boolean, default=True
            If True, include the KL divergence between the fidelity distribution from the circuit and the Haar distribution
        epsilon: float, default=1e-10
            Replace a zero bin with a bin of size epsilon to prevent infinity in the KL-divergence
    Returns:
        A plot of the fidelity distribution for the final states sampled from a parameterised circuit overlayed with the Haar fidelity distrubiotn
    """      
    import matplotlib.pyplot as plt
    data_pqc = np.array(sample_fidelities(circuit, samples, data=data, backend=backend))
    #data_Haar = np.linspace(0., 1., samples)
    n_qubit = len(get_circuit_qubits(circuit))
    data_Haar = get_data_Haar(n_qubit, samples, bins)
    plt.clf()
    plt.hist(data_Haar, bins=bins, range=(0.,1.), alpha=0.5, density=True, label='Haar')
    plt.hist(data_pqc, bins=bins, range=(0.,1.), alpha=0.5, density=True, label='PQC')
    plt.legend()
    plt.xlabel('Fidelity')
    plt.ylabel('Probability')
    
    if KL:
        import scipy as sp
        pdf_pqc = np.histogram(data_pqc, bins=bins, range=(0.,1.), density=True)[0]
        pdf_pqc /= pdf_pqc.sum()
        pdf_Haar = get_pdf_Haar(n_qubit, np.linspace(0., 1., bins))
        pdf_Haar = np.array([epsilon if v == 0 else v for v in pdf_Haar])
        Kullback_Leibler_divergence = sp.stats.entropy(pdf_pqc, pdf_Haar)
        plt.title('$D_{KL}=$'+str(Kullback_Leibler_divergence))
    
    return plt

def circuit_expressibility_measure(circuit, samples = 3000, bins=100, data=None, relative=False, epsilon=1e-10, backend=None):
    """Returns the expressibility measure of a parameterised circuit.
    
    The expressibility measure is the KL-divergence between the sampled fidelity distribution of the parameterised circuit and the Haar distribution. 

    Reference: https://arxiv.org/pdf/1905.10876.pdf
    
    If the circuit has parameterised gate operation, random values of
    the symbol values in the range (0, 2π) will be assigned to the gate operation.
    If `data` is given, the symbol values will be assigned according to the given data.
    
    Arguments:
        circuit: cirq.Circuit, quple.QuantumCircuit
            The circuit to find the expressibiliy measure
        samples: int, default=3000
            Number of samples
        bins: int, default=100
            Number of bins for the fidelity pdf
        data: real numbers with shape
            [batch_size, n_params] specifying parameter values to resolve
            into the circuits, following the ordering dictated by `symbol_names`
        relative: boolean, default=False
            If True, calculate the relative KL-divergence
        epsilon: float, default=1e-10
            Replace a zero bin with a bin of size epsilon to prevent infinity in the KL-divergence
    Returns:
        The expressibility measure of a parameterised circuit.
    """       
    import scipy as sp
    data_pqc = np.array(sample_fidelities(circuit, samples, data=data, backend=backend))
    pdf_pqc = np.histogram(data_pqc, bins=bins, range=(0.,1.), density=True)[0]
    pdf_pqc /= pdf_pqc.sum()
    n_qubit = len(get_circuit_qubits(circuit))
    pdf_Haar = get_pdf_Haar(n_qubit, np.linspace(0., 1., bins))
    pdf_Haar = np.array([epsilon if v == 0 else v for v in pdf_Haar])
    Kullback_Leibler_divergence = sp.stats.entropy(pdf_pqc, pdf_Haar)
    expressibility = Kullback_Leibler_divergence
    if relative:
        expressibility_idle_circuit = (2**n_qubit-1)*(np.log(bins))
        expressibility = -np.log(expressibility/expressibility_idle_circuit)
    return expressibility

def Meyer_Wallach_measure(state):
    """Returns the Meyer Wallach measure of a quantum state
    
    Reference: https://arxiv.org/pdf/quant-ph/0305094.pdf
    
    Arguments:
        state: array like
            The quantum state to calculate the Meyer Wallach measure
            
    Returns:
        The Meyer Wallach measure of a quantum state
    """      
    state = np.array(state)
    size = state.shape[0]
    n = int(np.log2(size))
    Q = 0.
    
    def linear_mapping(b:int, j:int) -> np.ndarray:
        keep_indices = [i for i in range(size) if b == ((i & (1 << j))!=0)]
        return state[keep_indices]
    
    def distance(u:np.ndarray, v:np.ndarray) -> float:
        return np.sum(np.abs(np.outer(u,v) - np.outer(v,u))**2)/2
    
    for k in range(n):
        iota_0k = linear_mapping(0, k)
        iota_1k = linear_mapping(1, k)
        Q += distance(iota_0k, iota_1k)
    Q = (4/n)*Q
    return Q
    
        
def circuit_entangling_measure(circuit, samples=200, data=None, backend=None):
    """Returns the entangling measure of a parameterised circuit
    
    The entangling measure of a parameterised circuit is the average Meyer Wallach 
    measure of the sampled final states of the circuit.
    Reference: https://arxiv.org/pdf/1905.10876.pdf
    
    If the circuit has parameterised gate operation, random values of
    the symbol values in the range (0, 2π) will be assigned to the gate operation.
    If `data` is given, the symbol values will be assigned according to the given data.
    
    Arguments:
        circuit: cirq.Circuit, quple.QuantumCircuit
            The circuit to find the entangling measure
        samples: int, default=200
            Number of samples
        data: real numbers with shape
            [batch_size, n_params] specifying parameter values to resolve
            into the circuits, following the ordering dictated by `symbol_names`            
    Returns:
        The entangling measure of a parameterised circuit
    """      
    final_states = sample_final_states(circuit, samples, data=data, backend=backend)
    mw_measures = [Meyer_Wallach_measure(fs) for fs in final_states]
    return np.mean(mw_measures)


def circuit_von_neumann_entropy(circuit, samples=200, data=None, backend=None):
    """Returns the average von Neumann entropy of the sampled density matrices of a parameterised circuit
    
    If the circuit has parameterised gate operation, random values of
    the symbol values in the range (0, 2π) will be assigned to the gate operation.
    If `data` is given, the symbol values will be assigned according to the given data.
    
    Arguments:
        circuit: cirq.Circuit, quple.QuantumCircuit
            The circuit to find the von Neumann entropy
        samples: int, default=200
            Number of samples
        data: real numbers with shape
            [batch_size, n_params] specifying parameter values to resolve
            into the circuits, following the ordering dictated by `symbol_names`            
    Returns:
        The average von neumann entropy of the sampled density matrices of a parameterised circuit
    """        
    density_matrices = sample_density_matrices(circuit, samples, data=data, backend=backend)
    von_neumann_entropy = [cirq.von_neumann_entropy(dm) for dm in density_matrices]
    return np.mean(von_neumann_entropy)
    
    
def gradient_variance_test(circuits, op, symbol=None):
    """Performs the gradient variance test for a parameter symbol in a parameterised circuit
    
    Reference: https://www.nature.com/articles/s41467-018-07090-4
    
    Arguments:
        circuits: list of cirq.Circuit, quple.QuantumCircuit
            The circuits to perform the gradient variance test
        op: cirq.Gate
            The gate operation to sample the expectation value from
        symbol: str, default=None
            The parameter symbol which the values are varied in the gradient variance test.
            If None, the first symbol that appears in the circuit will be varied whereas others are fixed
            by some random value.
    Returns:
        Gradient variance of a parameter symbol among the given circuits
    """      
    import tensorflow_quantum as tfq
    import tensorflow as tf
    """Compute the variance of a batch of expectations w.r.t. op on each circuit that 
    contains `symbol`."""
    resolved_circuits = []
    # Resolve irrelevant symbols:
    for circuit in circuits: 
        symbols = get_circuit_symbols(circuit)
        if not len(symbols) > 0:
            raise ValueError('No symbols found in circuit')
        if not symbol:
            symbol = symbols[0]
            symbols = symbols[1:]
        else:
            symbols.remove(symbol)
        resolver = cirq.ParamResolver({s:np.random.uniform() * 2.0 * np.pi for s in symbols})
        resolved_circuits.append(cirq.protocols.resolve_parameters(circuit, resolver))

    # Setup a simple layer to batch compute the expectation gradients.
    expectation = tfq.layers.Expectation()

    # Prep the inputs as tensors
    circuit_tensor = tfq.convert_to_tensor(resolved_circuits)
    n_circuits = len(resolved_circuits)
    values_tensor = tf.convert_to_tensor(
        np.random.uniform(0, 2 * np.pi, (n_circuits, 1)).astype(np.float32))

    # Use TensorFlow GradientTape to track gradients.
    with tf.GradientTape() as g:
        g.watch(values_tensor)
        forward = expectation(circuit_tensor,
                              operators=op,
                              symbol_names=[symbol],
                              symbol_values=values_tensor)

    # Return variance of gradients across all circuits.
    grads = g.gradient(forward, values_tensor)
    grad_var = tf.math.reduce_std(grads, axis=0)
    return grad_var.numpy()[0]


def has_composite_symbols(circuit):
    if not isinstance(circuit, cirq.Circuit):
        raise ValueError("circuit must be a cirq.Circuit object.")
    if (isinstance(circuit, quple.QuantumCircuit)) and (circuit.expr_map is not None):
        circuit = circuit.get_unflattened_circuit()     
    for moment in circuit:
        for op in moment:
            if cirq.is_parameterized(op):
                if len(symbols_in_op(op.gate)) > 1:
                    return True
    return False
'''
    if not isinstance(circuit, quple.QuantumCircuit):
        circuit = quple.QuantumCircuit.from_cirq(circuit)
    # circuit is already flattened
    if circuit.expr_map is not None:
        return set(circuit.symbols) != set(circuit.raw_symbols)
    circuit_symbols = circuit.symbols
    flattened_circuit, _ = cirq.flatten(circuit)
    flattened_circuit_symbols = get_circuit_symbols(flattened_circuit)
    return set(circuit_symbols) != set(flattened_circuit_symbols)
'''