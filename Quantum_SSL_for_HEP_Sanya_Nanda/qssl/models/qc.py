import pennylane as qml
import numpy as np

n_qubits = 6
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev, interface='torch')
def quantum_circuit_angle_entangle_weights(inputs, weights):
    # Explicit AngleEmbedding gates (RY rotations)
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    
    # Explicit BasicEntanglerLayer gates
    for layer in range(len(weights)):
        for i in range(n_qubits):
            qml.RX(weights[layer][i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])  # Chain entanglement
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


@qml.qnode(dev, interface='torch')
def quantum_circuit_angle_entangle_inputs(inputs):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i) 
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])  
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


@qml.qnode(dev, interface='torch')
def quantum_circuit_amplitude(data):
    # Amplitude embedding (data should be of size 2^n_qubits)
    qml.AmplitudeEmbedding(features=data, wires=range(n_qubits), normalize=True)
    
    # Apply rotations using weights for parameterization
    for i in range(n_qubits):
        qml.RY(data[i], wires=i)  # Rotation using the provided weights
    
    # Apply CNOT gates for strong entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    
    # Add final CNOT between the last and first qubit for full entanglement
    qml.CNOT(wires=[n_qubits - 1, 0])
    
    # Return the expectation values of Pauli-Z on each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]