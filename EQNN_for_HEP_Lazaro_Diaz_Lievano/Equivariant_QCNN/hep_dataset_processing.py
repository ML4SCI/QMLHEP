
import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 


# Cargar el archivo .h5
def quark_gluon():  
    file_path = "Equivariant_QCNN/hep_data/QG_16x16x1_dataset_50k"
    with h5py.File(file_path, "r") as file:
        X = np.array(file["X"])
        y = np.array(file["y"])

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convertir los datos a tensores de TensorFlow
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    Y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    Y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_train: {Y_train.shape}")
    print(f"Shape of y_test: {Y_test.shape}")

    return X_train, X_test, Y_train, Y_test 


def electron_photon():
    file_path = "Equivariant_QCNN/hep_data/electron.h5"
    with h5py.File(file_path, "r") as file:
        X_e = np.array(file["X"])
        y_e = np.array(file["y"])

    file_path = "Equivariant_QCNN/hep_data/photon.h5"
    with h5py.File(file_path, "r") as file:
        X_p = np.array(file["X"])
        y_p = np.array(file["y"])

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convertir los datos a tensores de TensorFlow
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    Y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    Y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_train: {Y_train.shape}")
    print(f"Shape of y_test: {Y_test.shape}")

    return X_train, X_test, Y_train, Y_test 

