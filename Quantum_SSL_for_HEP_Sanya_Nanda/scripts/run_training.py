import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qssl.data.data_loader import DataLoader
from qssl.models.qcl import QuantumCircuit, QuantumCNN, SiameseNetwork
from qssl.training.train import Trainer
from qssl.config import Config

if __name__ == "__main__":    
    print("Data Loading ...")
    data_loader = DataLoader('../data/quark_gluon_dataset/qg_20000_pairs_c4.npz')
    pairs_train, labels_train = data_loader.get_train_data()
    pairs_test, labels_test = data_loader.get_test_data()
    print("Data Loaded.")
    
    print("Quantum Circuit initiated")
    quantum_circuit = QuantumCircuit()
    quantum_layer = quantum_circuit.get_quantum_layer()
    
    print("Model compiled")
    input_shape = pairs_train.shape[2:]
    quantum_cnn = QuantumCNN(input_shape, quantum_layer)
    siamese_network = SiameseNetwork(input_shape, quantum_cnn).create_network()
    
    print("Training ...")
    trainer = Trainer(siamese_network, pairs_train, labels_train, pairs_test, labels_test)
    trainer.train(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE)