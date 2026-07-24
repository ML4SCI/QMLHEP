# src/model.py

import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

def create_qnode(n_qubits=5):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.RY(inputs[0], wires=1)
        qml.RY(inputs[1], wires=2)
        qml.RY(inputs[2], wires=3)
        qml.RY(inputs[3], wires=4)

        qml.RX(weights['w000'], wires=1)
        qml.RX(weights['w001'], wires=2)

        qml.RY(weights['w008'], wires=1)
        qml.RY(weights['w009'], wires=2)

        qml.RZ(weights['w016'], wires=1)
        qml.RZ(weights['w017'], wires=2)
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 1])

        qml.RX(weights['w200'], wires=1)
        qml.RX(weights['w201'], wires=2)

        qml.RY(weights['w208'], wires=1)
        qml.RY(weights['w209'], wires=2)

        qml.RZ(weights['w216'], wires=1)
        qml.RZ(weights['w217'], wires=2)
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 1])

        qml.Hadamard(0)
        qml.CSWAP(wires=[0, 1, 3])
        qml.CSWAP(wires=[0, 2, 4])
        qml.Hadamard(0)

        return qml.probs(wires=[0])

    return qnode

class QuantumGAN(nn.Module):
    def __init__(self, qnode, weight_shapes):
        super(QuantumGAN, self).__init__()
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        return self.qlayer(x)
