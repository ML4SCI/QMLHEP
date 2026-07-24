import itertools
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.aqua.algorithms.classifiers import QSVM 
from qiskit.aqua.components.feature_maps import FeatureMap
import quple
from quple.utils.utils import parallel_run, batching, flatten_list

def construct_circuit(x, feature_map):

    q = QuantumRegister(feature_map.num_qubits, 'q')
    c = ClassicalRegister(feature_map.num_qubits, 'c')
    qc = QuantumCircuit(q, c)

    # write input state from sample distribution
    if isinstance(feature_map, FeatureMap):
        qc += feature_map.construct_circuit(x, q)
    else:
        raise ValueError('Only FeatureMap istance is allowed')
    return qc

def execute(circuits, quantum_instance):
    return quantum_instance.execute(circuits, had_transpiled=True)

def assign_parameters(param_values, param_names, feature_map):
    return feature_map.assign_parameters({param_names: param_values})

def batch_assign_parameters(param_values, param_names, feature_map):
    return [feature_map.assign_parameters({param_names: val}) for val in param_values]

def get_qiskit_state_vectors_deprecated(quantum_instance, feature_map, x, batchsize=100):
    if not quantum_instance.is_statevector:
        raise ValueError('Quantum instance must be a statevector simulator')
        
    q = QuantumRegister(feature_map.num_qubits, 'q')
    c = ClassicalRegister(feature_map.num_qubits, 'c')
    qc = QuantumCircuit(q, c)
    
    if isinstance(feature_map, QuantumCircuit):
        use_parameterized_circuits = True
    else:
        use_parameterized_circuits = feature_map.support_parameterized_circuit    
        
    if isinstance(feature_map, FeatureMap):
        circuits = parallel_run(construct_circuit, x, itertools.repeat(feature_map))
    else:
        feature_map_params = ParameterVector('x', feature_map.feature_dimension)
        parameterized_circuit = QSVM._construct_circuit(
          (feature_map_params, feature_map_params), feature_map, False,
          is_statevector_sim=True)
        parameterized_circuit = quantum_instance.transpile(parameterized_circuit)[0]
        if batchsize is not None:
            circuits = flatten_list(parallel_run(batch_assign_parameters, batching(x, batchsize), 
                                                 itertools.repeat(feature_map_params), itertools.repeat(parameterized_circuit)))
        else:
            circuits = parallel_run(assign_parameters, x, itertools.repeat(feature_map_params), itertools.repeat(parameterized_circuit))    
    results = quantum_instance.execute(circuits, had_transpiled=use_parameterized_circuits) 
    statevectors = np.array([result.data.statevector for result in results.results])
    return statevectors


def get_qiskit_state_vectors_test(quantum_instance, feature_map, x, batchsize=100):
    if not quantum_instance.is_statevector:
        raise ValueError('Quantum instance must be a statevector simulator')
    
    q = QuantumRegister(feature_map.num_qubits, 'q')
    c = ClassicalRegister(feature_map.num_qubits, 'c')
    qc = QuantumCircuit(q, c)
    
    feature_map_params = ParameterVector('x', feature_map.feature_dimension)
    parameterized_circuit = QSVM._construct_circuit(
      (feature_map_params, feature_map_params), feature_map, False,
      is_statevector_sim=True)
    parameterized_circuit = quantum_instance.transpile(parameterized_circuit)[0]
    if batchsize is not None:
        circuits = flatten_list(parallel_run(batch_assign_parameters, batching(x, batchsize), 
                                             itertools.repeat(feature_map_params), itertools.repeat(parameterized_circuit)))
    else:
        circuits = parallel_run(assign_parameters, x, itertools.repeat(feature_map_params), itertools.repeat(parameterized_circuit)) 
        
    results = quantum_instance.execute(circuits, had_transpiled=True) 
    statevectors = np.array([result.data.statevector for result in results.results])
    return statevectors


def get_qiskit_state_vectors(quantum_instance, feature_map, x, batchsize=None):
    if not quantum_instance.is_statevector:
        raise ValueError('Quantum instance must be a statevector simulator')
    n_qubit = feature_map.num_qubits
    data_size = x.shape[0]
    q = QuantumRegister(n_qubit, 'q')
    c = ClassicalRegister(n_qubit, 'c')
    qc = QuantumCircuit(q, c)
    
    feature_map_params = ParameterVector('x', feature_map.feature_dimension)
    parameterized_circuit = QSVM._construct_circuit(
      (feature_map_params, feature_map_params), feature_map, False,
      is_statevector_sim=True)
    parameterized_circuit = quantum_instance.transpile(parameterized_circuit)[0]
    if batchsize is not None:
        circuits = flatten_list(parallel_run(batch_assign_parameters, batching(x, batchsize), 
                                             itertools.repeat(feature_map_params), itertools.repeat(parameterized_circuit)))
    else:
        circuits = parallel_run(assign_parameters, x, itertools.repeat(feature_map_params), itertools.repeat(parameterized_circuit)) 

    results = quantum_instance.execute(circuits, had_transpiled=True) 
    statevectors = np.zeros((data_size, 2**n_qubit), dtype=results.results[0].data.statevector.dtype)
    for i, result in enumerate(results.results):
        statevectors[i] = result.data.statevector.copy()
        del result.data.statevector
        
    return statevectors
    