

# Loads and Processes the data that will be used in QCNN and Hierarchical Classifier Training

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split
import h5py

def data_load_and_process(dataset, classes=[0, 1], feature_reduction= "img16x16x1", binary=True):
    
    if dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0  # normalize the data

    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0  # normalize the data

    elif dataset == "quark_gluon":
        QG_path = "/home/lazaror/quantum/pruebas/EQCNN_local_testing/EQNN_for_HEP/Equivariant_QCNN/data/Q-G_resize/QG-bilinear-ECAL-(16, 16, 1).h5py" ## quark_gluon-16x16-MMS.h5 solo tiene una clase
        with h5py.File(QG_path, "r") as file:
            X = np.array(file["X"])
            y = np.array(file["y"])
    
        X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.2,random_state=42, shuffle = True)
        
        X_train = X_train[:20000]
        Y_train = Y_train[:20000]
        X_test = X_test[:5000]
        Y_test = Y_test[:5000]

    elif dataset == "electron_photon":
        path_ep = "/home/lazaror/quantum/pruebas/EQCNN_local_testing/EQNN_for_HEP/Equivariant_QCNN/data/E-P_rescaled"
        with h5py.File(path_ep, "r") as file:
            X_ep = np.array(file["X"])
            y_ep = np.array(file["y"])
        
        x_train, x_test, y_train, y_test = train_test_split(X_ep, y_ep, test_size=0.2, random_state=42, stratify=y_ep)
        X_train = x_train[:20000]
        Y_train = y_train[:20000]
        X_test = x_test[:5000]
        Y_test = y_test[:5000]

    else: 
        print("No valid dataset. Please choose one of the following: mnist, fashion_mnist, quark_gluon or electron_photon.")   

## classes 
    if classes == 'odd_even':
        odd = [1, 3, 5, 7, 9]
        X_train = x_train
        X_test = x_test
        if binary == False:
            Y_train = [1 if y in odd else 0 for y in y_train]
            Y_test = [1 if y in odd else 0 for y in y_test]
        elif binary == True:
            Y_train = [1 if y in odd else -1 for y in y_train]
            Y_test = [1 if y in odd else -1 for y in y_test]

    elif classes == '>4':
        greater = [5, 6, 7, 8, 9]
        X_train = x_train
        X_test = x_test
        if binary == False:
            Y_train = [1 if y in greater else 0 for y in y_train]
            Y_test = [1 if y in greater else 0 for y in y_test]
        elif binary == True:
            Y_train = [1 if y in greater else -1 for y in y_train]
            Y_test = [1 if y in greater else -1 for y in y_test]

## processing
    else:
        if dataset == "quark_gluon" or "electron_photon": 
            pass          
        else: 
            x_train_filter_01 = np.where((y_train == classes[0]) | (y_train == classes[1]))
            x_test_filter_01 = np.where((y_test == classes[0]) | (y_test == classes[1]))

            X_train, X_test = x_train[x_train_filter_01], x_test[x_test_filter_01]
            Y_train, Y_test = y_train[x_train_filter_01], y_test[x_test_filter_01]
       
    if binary == False:
        Y_train = [1 if y == classes[0] else 0 for y in Y_train]
        Y_test = [1 if y == classes[0] else 0 for y in Y_test]
    elif binary == True:
        Y_train = [1 if y == classes[0] else -1 for y in Y_train]
        Y_test = [1 if y == classes[0] else -1 for y in Y_test]

    if feature_reduction == 'img16x16x1':
        X_train = tf.image.resize(X_train[:], (16,16)).numpy()
        X_test = tf.image.resize(X_test[:], (16,16)).numpy()
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        X_train = np.squeeze(X_train, axis=-1)
        X_test = np.squeeze(X_test, axis=-1)
        return X_train, X_test, Y_train, Y_test

    elif feature_reduction == 'resize256':
        X_train = tf.image.resize(X_train[:], (256, 1)).numpy()
        X_test = tf.image.resize(X_test[:], (256, 1)).numpy()
        X_train, X_test = tf.squeeze(X_train).numpy(), tf.squeeze(X_test).numpy()
        return X_train, X_test, Y_train, Y_test

    elif feature_reduction == 'pca8':
        X_train = tf.image.resize(X_train[:], (X_train.shape[1]*X_train.shape[1], 1)).numpy()
        X_test = tf.image.resize(X_test[:], (X_test.shape[1]*X_test.shape[1], 1)).numpy()
        X_train, X_test = tf.squeeze(X_train), tf.squeeze(X_test)

        pca = PCA(8)

        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        # Rescale for angle embedding
        X_train, X_test = (X_train - X_train.min()) * (np.pi / (X_train.max() - X_train.min())),\
                              (X_test - X_test.min()) * (np.pi / (X_test.max() - X_test.min()))
            
            
        return X_train, X_test, Y_train, Y_test


