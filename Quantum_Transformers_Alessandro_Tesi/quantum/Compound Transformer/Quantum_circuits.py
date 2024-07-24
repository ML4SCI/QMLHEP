#Circuits are clearly explained in QViT paper circuits.ipynb

import cirq
from cirq import ops
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import sympy

class RBSGate(cirq.Gate):
    def __init__(self, theta):
        super(RBSGate, self).__init__()
        self.theta = theta

    def _num_qubits_(self):
        return 2

    def _decompose_(self, qubits):
        q0, q1 = qubits
        yield cirq.H(q0), cirq.H(q1)
        yield cirq.CZ(q0, q1)
        yield cirq.ry(self.theta / 2)(q0), cirq.ry(-self.theta / 2)(q1)
        yield cirq.CZ(q0, q1)
        yield cirq.H(q0), cirq.H(q1)

    def _unitary_(self):
        cos = np.cos(self.theta)
        sin = np.sin(self.theta)
        return np.array([
            [1, 0, 0,0],
            [0, cos, sin, 0],
            [0, -sin, cos, 0],
            [0, 0, 0, 1]
        ])
    
    def _circuit_diagram_info_(self, args):
        return "[RBS({:.2f})]".format(self.theta), "[RBS({:.2f})]".format(self.theta)

def convert_array(X):
    X = tf.cast(X, dtype=tf.float32)  
    alphas = tf.zeros(X.shape[:-1] + (X.shape[-1]-1,), dtype=X.dtype)
    X_normd = X / (tf.sqrt(tf.reduce_sum(X**2, axis=-1, keepdims=True))+ 1e-10) # 1e-10 is added to prevent divisions by 0
    for i in range(X.shape[-1]-1):
        if i == 0:
            alphas = tf.tensor_scatter_nd_update(alphas, [[i]], [tf.acos(X_normd[..., i])])
        elif i < (X.shape[-1]-2):
            prod_sin_alphas = tf.reduce_prod(tf.sin(alphas[..., :i]), axis=-1)
            updated_value = tf.acos(X_normd[..., i] / (prod_sin_alphas + 1e-10)) # 1e-10 is added to prevent divisions by 0
            alphas = tf.tensor_scatter_nd_update(alphas, [[i]], [updated_value])
        else:
            updated_value = tf.atan2(X_normd[..., -1], X_normd[..., -2])
            alphas = tf.tensor_scatter_nd_update(alphas, [[i]], [updated_value])
    # Replace NaN values with 1.57 (π/2)
    alphas = tf.where(tf.math.is_nan(alphas), tf.fill(tf.shape(alphas), 1.57), alphas)

    return alphas

def convert_matrix(X):
    mag_alphas = convert_array(tf.sqrt(tf.reduce_sum(X**2, axis=1)))
    alphas = tf.TensorArray(dtype=X.dtype, size=X.shape[0])
    for i in range(X.shape[0]):
        alphas = alphas.write(i, convert_array(X[i]))
    alphas = alphas.stack()
    return mag_alphas, alphas

def vector_loader(circuit, alphas, wires=None, is_x=True, is_conjugate=False):
    if wires is None:
        wires = [i for i in range(len(alphas) + 1)]
    if is_x and not is_conjugate:
        circuit.append(cirq.X(cirq.LineQubit(wires[0])))
    if is_conjugate:
        for i in range(len(wires) - 1):
            # Ensure alphas[i] is a float before using it
            alpha_value = alphas[i].numpy() if hasattr(alphas[i], 'numpy') else alphas[i]
            rbs_gate = RBSGate(-alpha_value)
            circuit.append(rbs_gate.on(cirq.LineQubit(wires[i]), cirq.LineQubit(wires[i+1])))
    else:
        for i in range(len(wires) - 1):
            # Ensure alphas[i] is a float before using it
            alpha_value = alphas[i].numpy() if hasattr(alphas[i], 'numpy') else alphas[i]
            rbs_gate = RBSGate(alpha_value)
            circuit.append(rbs_gate.on(cirq.LineQubit(wires[i]), cirq.LineQubit(wires[i+1])))
    if is_x and is_conjugate:
        circuit.append(cirq.X(cirq.LineQubit(wires[0])))

def matrix_loader(circuit, mag_alphas, alphas, mag_wires, wires, is_conjugate=False):
    if not is_conjugate:
        vector_loader(circuit, mag_alphas, wires=mag_wires, is_x=False)
        for i in range(len(mag_wires)):
            circuit.append(cirq.CNOT(cirq.LineQubit(mag_wires[i]), cirq.LineQubit(wires[0])))
            vector_loader(circuit, alphas[i], wires=wires, is_x=False)
            if i != len(mag_alphas):
                vector_loader(circuit, alphas[i+1], wires=wires, is_x=False, is_conjugate=True)
    else:
        for i in reversed(range(len(mag_wires))):
            if i != len(mag_alphas):
                vector_loader(circuit, alphas[i+1], wires=wires, is_x=False, is_conjugate=False)
            vector_loader(circuit, alphas[i], wires=wires, is_x=False, is_conjugate=True)
            circuit.append(cirq.CNOT(cirq.LineQubit(mag_wires[i]), cirq.LineQubit(wires[0])))
        vector_loader(circuit, mag_alphas, wires=mag_wires, is_x=False, is_conjugate=True)

def pyramid_circuit(circuit, parameters, wires=None):
    # If wires is None, use all qubits in the circuit
    if wires is None:
        wires = list(circuit.all_qubits())
        length = len(wires)
    else:
        # If wires is not None, ensure it's a list of qubits
        length = len(wires)

    k = 0  

    for i in range(2 * length - 2):
        j = length - abs(length - 1 - i)

        if i % 2:
            for _ in range(j):
                if _ % 2 == 0 and k < len(parameters):
                    circuit.append(RBSGate(parameters[k]).on(wires[_], wires[_ + 1]))
                    k += 1
        else:
            for _ in range(j):
                if _ % 2 and k < len(parameters):
                    circuit.append(RBSGate(parameters[k]).on(wires[_], wires[_ + 1]))
                    k += 1

def x_circuit(circuit, parameters, wires=None):
    # If wires is None, use all qubits in the circuit
    if wires is None:
        wires = list(circuit.all_qubits())
        length = len(wires)
    else:
        # If wires is not None, ensure it's a list of qubits
        length = len(wires)

    k = 0  

    for i in range(len(wires) - 1):
        j = len(wires) - 2 - i
        
        if i == j:
            circuit.append(RBSGate(parameters[k]).on(wires[j], wires[j + 1]))
            k += 1
        else:
            circuit.append(RBSGate(parameters[k]).on(wires[i], wires[i + 1]))
            k += 1
            circuit.append(RBSGate(parameters[k]).on(wires[j], wires[j + 1]))
            k += 1

def butterfly_circuit(circuit, parameters, wires=None):
    # If wires is None, use all qubits in the circuit
    if wires is None:
        wires = list(circuit.all_qubits())
        length = len(wires)
    else:
        # If wires is not None, ensure it's a list of qubits
        length = len(wires)
    if length > 1:
        n=length//2
        x = 0
        for i in range(n):
            circuit.append(RBSGate(parameters[x])(wires[i], wires[i+n]))
            x += 1
        butterfly_circuit(circuit, parameters[x: (len(parameters)//2 + x//2) ], wires = wires[:n])
        butterfly_circuit(circuit, parameters[(len(parameters)//2 + x//2):], wires = wires[n:])

def orthogonal_patch_wise_NN_circuit(circuit, patch, parameters, wires=None):
    if wires==None:
        wires = list(circuit.all_qubits())[:len(patch)]
    # Load the vector (patch) onto the circuit
    alphas = convert_array(patch)
    vector_loader(circuit, alphas, wires=[qubit.x for qubit in wires])

    # Apply the butterfly circuit to create the orthogonal layer
    butterfly_circuit(circuit, parameters, wires=wires)

