# quantum_gan_with_Perceptual_quantum_loss/src/model.py



def create_generator_qnode(dev, n_qubits):
    @qml.qnode(dev, interface='torch')
    def generator_qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
    return generator_qnode

def create_discriminator_qnode(dev, n_qubits):
    @qml.qnode(dev, interface='torch')
    def discriminator_qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))
    return discriminator_qnode

class QuantumGAN(nn.Module):
    def __init__(self, qnode, weight_shapes):
        super(QuantumGAN, self).__init__()
        self.qnn = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        return self.qnn(x)
