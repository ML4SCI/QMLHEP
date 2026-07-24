import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import pennylane as qml
from qssl.config import Config

n_qubits = Config.N_QUBITS

# Define Quantum Circuit Function
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# Quantum Circuit Class
class QuantumCircuit:
    def __init__(self, n_qubits=Config.N_QUBITS, n_layers=Config.N_LAYERS):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qnode = qml.QNode(quantum_circuit, self.dev)

    def get_quantum_layer(self):
        return qml.qnn.KerasLayer(self.qnode, self.weight_shapes, output_dim=self.n_qubits)

# CNN Model with Quantum Layer
class QuantumCNN:
    def __init__(self, input_shape, quantum_layer, n_qubits=Config.N_QUBITS):
        self.input_shape = input_shape
        self.quantum_layer = quantum_layer
        self.n_qubits = n_qubits
    
    def create_model(self, return_embeddings=False):
        model = models.Sequential()
        model.add(layers.Input(shape=self.input_shape))
        model.add(layers.Conv2D(32, (3, 3), activation='relu')) # Conv layer 1
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Conv layer 2
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
    
        # ------ Quantum layer
        # Reducing dimensions to match n_qubits
        model.add(layers.Dense(self.n_qubits)) 
        # Quantum layer
        model.add(self.quantum_layer)
        # Dense layer after quantum layer
        model.add(layers.Dense(self.n_qubits, activation='relu'))
        if return_embeddings:
            return model
        # --------------------------------------------------------------
        
        # model.add(layers.Dense(1, activation='sigmoid'))  
        return model


# Siamese Network
class SiameseNetwork:
    def __init__(self, input_shape, quantum_cnn):
        self.input_shape = input_shape
        self.quantum_cnn = quantum_cnn

    def create_network(self):
        base_model = self.quantum_cnn.create_model()

        input_0 = layers.Input(shape=self.input_shape)
        input_1 = layers.Input(shape=self.input_shape)

        processed_0 = base_model(input_0)
        processed_1 = base_model(input_1)

        distance = layers.Lambda(
            lambda embeddings: tf.sqrt(tf.reduce_sum(tf.square(embeddings[0] - embeddings[1]), axis=-1)),
            output_shape=(1,)
        )([processed_0, processed_1])

        siamese_model = models.Model([input_0, input_1], distance)
        return siamese_model