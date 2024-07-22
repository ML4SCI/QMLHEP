import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QuantumGenerator(nn.Module):
    def __init__(self, n_qubits=4, depth=3, output_dim=16*16):
        super(QuantumGenerator, self).__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.output_dim = output_dim
        self.params = nn.Parameter(torch.randn((depth * 2 * n_qubits,), requires_grad=True))
        self.fc1 = nn.Linear(n_qubits, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, self.output_dim)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        q_out = []
        for i in range(x.shape[0]):
            data = x[i].detach().cpu().numpy()
            result = np.array(qnode(self.params.detach().cpu().numpy(), data))
            q_out.append(result)
        q_out = torch.tensor(q_out, dtype=torch.float32).to(device)
        q_out = F.relu(self.bn1(self.fc1(q_out)))
        q_out = F.relu(self.bn2(self.fc2(q_out)))
        q_out = torch.tanh(self.fc3(q_out))
        return q_out.view(-1, 1, 16, 16)

n_qubits = 4
dev = qml.device('default.qubit', wires=n_qubits)

def quantum_circuit(params, data, n_qubits=4):
    depth = len(params) // (2 * n_qubits)
    for d in range(depth):
        for i in range(n_qubits):
            qml.RY(data[i], wires=i)
            qml.RY(params[d * 2 * n_qubits + i], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
        for i in range(n_qubits):
            qml.RZ(params[d * 2 * n_qubits + n_qubits + i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

qnode = qml.QNode(quantum_circuit, dev)
