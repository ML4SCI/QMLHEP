"""
Utility functions for data preprocessing
"""

import cirq
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, LocallyLinearEmbedding


def PCA_reduce(data, pca_components, val_data=None, test_data=None):
    """Reduce data dimension using pca

    Args:
        data (array): input data
        pca_components (int): number of principle components to redce to
        val_data (array): if given will be reduced just like data
        test_data (array): if given will be reduced just like data

    Returns:
        Data with reduced dimension, if val_data / test_data given, tuple of data sets
    """
    data_flat = np.array(data).reshape(data.shape[0],-1)
    pca_dims = PCA(n_components=pca_components)
    data_reduced = pca_dims.fit_transform(data_flat)

    minimum = abs(np.min(data_reduced))
    maximum = abs(np.max(data_reduced))
    
    data_reduced = (data_reduced + minimum) / (maximum + minimum)
    
    ret = data_reduced

    if not val_data is None:
        val_data_flat = np.array(val_data).reshape(val_data.shape[0],-1)
        val_data_reduced = pca_dims.transform(val_data_flat)
        
        val_data_reduced = (val_data_reduced + minimum) / (maximum + minimum)
        
        ret = (data_reduced, val_data_reduced)

    if not test_data is None:
        test_data_flat = np.array(test_data).reshape(test_data.shape[0],-1)
        test_data_reduced = pca_dims.transform(test_data_flat)
        
        test_data_reduced = (test_data_reduced + minimum) / (maximum + minimum)
        
        ret = (data_reduced, val_data_reduced, test_data_reduced)

    return ret


def TruncatedPCA_reduce(data, pca_components, val_data=None, test_data=None):
    """like Pca_reduce with truncated pca

    Args:
        data (array): input data
        pca_components (int): number of principle components to redce to
        val_data (array): if given will be reduced just like data
        test_data (array): if given will be reduced just like data

    Returns:
        Data with reduced dimension, if val_data / test_data given, tuple of data sets
    """
    data_flat = np.array(data).reshape(data.shape[0],-1)
    pca_dims = TruncatedSVD(n_components=pca_components)
    data_reduced = pca_dims.fit_transform(data_flat)

    minimum = abs(np.min(data_reduced))
    maximum = abs(np.max(data_reduced))

    data_reduced = (data_reduced + minimum) / (maximum + minimum)

    ret = data_reduced

    if not val_data is None:
        val_data_flat = np.array(val_data).reshape(val_data.shape[0],-1)
        val_data_reduced = pca_dims.transform(val_data_flat)

        val_data_reduced = (val_data_reduced + minimum) / (maximum + minimum)

        ret = (data_reduced, val_data_reduced)

    if not test_data is None:
        test_data_flat = np.array(test_data).reshape(test_data.shape[0],-1)
        test_data_reduced = pca_dims.transform(test_data_flat)

        test_data_reduced = (test_data_reduced + minimum) / (maximum + minimum)

        ret = (data_reduced, val_data_reduced, test_data_reduced)

    return ret


def tsne_reduce(data, pca_components, val_data=None, test_data=None):
    """like Pca_reduce with tsne

    Args:
        data (array): input data
        pca_components (int): number of principle components to redce to
        val_data (array): if given will be reduced just like data
        test_data (array): if given will be reduced just like data

    Returns:
        Data with reduced dimension, if val_data / test_data given, tuple of data sets
    """
    data_flat = np.array(data).reshape(data.shape[0],-1)
    tsne_dims = TSNE(n_components=pca_components, learning_rate='auto', method="exact")
    data_reduced = tsne_dims.fit_transform(data_flat)

    minimum = abs(np.min(data_reduced))
    maximum = abs(np.max(data_reduced))

    data_reduced = (data_reduced + minimum) / (maximum + minimum)

    ret = data_reduced

    if not val_data is None:
        val_data_flat = np.array(val_data).reshape(val_data.shape[0],-1)
        val_data_reduced = pca_dims.transform(val_data_flat)

        val_data_reduced = (val_data_reduced + minimum) / (maximum + minimum)

        ret = (data_reduced, val_data_reduced)

    if not test_data is None:
        test_data_flat = np.array(test_data).reshape(test_data.shape[0],-1)
        test_data_reduced = pca_dims.transform(test_data_flat)

        test_data_reduced = (test_data_reduced + minimum) / (maximum + minimum)

        ret = (data_reduced, val_data_reduced, test_data_reduced)

    return ret

def lle_reduce(data, pca_components, val_data=None, test_data=None):
    """like Pca_reduce with lle

    Args:
        data (array): input data
        pca_components (int): number of principle components to redce to
        val_data (array): if given will be reduced just like data
        test_data (array): if given will be reduced just like data

    Returns:
        Data with reduced dimension, if val_data / test_data given, tuple of data sets
    """
    data_flat = np.array(data).reshape(data.shape[0],-1)
    lle_dims = LocallyLinearEmbedding(n_components=pca_components, n_neighbors=8)
    data_reduced = lle_dims.fit_transform(data_flat)

    minimum = abs(np.min(data_reduced))
    maximum = abs(np.max(data_reduced))

    data_reduced = (data_reduced + minimum) / (maximum + minimum)

    ret = data_reduced

    if not val_data is None:
        val_data_flat = np.array(val_data).reshape(val_data.shape[0],-1)
        val_data_reduced = pca_dims.transform(val_data_flat)

        val_data_reduced = (val_data_reduced + minimum) / (maximum + minimum)

        ret = (data_reduced, val_data_reduced)

    if not test_data is None:
        test_data_flat = np.array(test_data).reshape(test_data.shape[0],-1)
        test_data_reduced = pca_dims.transform(test_data_flat)

        test_data_reduced = (test_data_reduced + minimum) / (maximum + minimum)

        ret = (data_reduced, val_data_reduced, test_data_reduced)

    return ret


def input_states(data, data_qbits, latent_qbits):
    """Prepare input states for tfq training of the large QAE architecture

    Args:
        data (array): input data as 2d array
        data_qubits (int): number of data qubits
        latent_qubits (int): number of qubits for the latent space

    Returns:
        cirq circuits with prepared input states
    """
    network_qbits = data_qbits + (data_qbits - latent_qbits)
    qubits = cirq.GridQubit.rect(1, network_qbits + data_qbits + 1)

    values = data
    
    data_circuit = cirq.Circuit([cirq.X(q) ** v for v,q in zip(values, qubits[:data_qbits])])
    reference_circuit = cirq.Circuit([cirq.X(q) ** v for v,q in zip(values, qubits[network_qbits:-1])])
    
    train_circuit = data_circuit + reference_circuit
    return train_circuit

def input_states_SQAE(data, data_qbits, latent_qbits):
    """Prepare input states for tfq training of the SQAE

    Args:
        data (array): input data as 2d array
        data_qubits (int): number of data qubits
        latent_qubits (int): number of qubits for the latent space

    Returns:
        cirq circuits with prepared input states
    """
    non_latent = data_qbits - latent_qbits
    qubits = cirq.GridQubit.rect(1, data_qbits + non_latent + 1)

    data_circuit = cirq.Circuit([cirq.X(q) ** v for v,q in zip(data, qubits[:data_qbits])])

    return data_circuit
