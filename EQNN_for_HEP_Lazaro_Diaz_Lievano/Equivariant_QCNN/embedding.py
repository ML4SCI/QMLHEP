# This is an implementation of data_embedding function used for 8 qubits Quantum Convolutional Neural Network (QCNN)
# and Hierarchical Quantum Classifier circuit.
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
import numpy as np

def equivariant_amplitude_encoding(img: np.ndarray) -> None:
    # n = 8
    wires=range(8)
    n = len(wires) // 2
    # If the image is single-channel, reshape it to 2D
    if img.shape[2] == 1:
        img = img.reshape(img.shape[0], img.shape[1])

    # Initialize the feature vector with zeros
    features = np.zeros(2 ** (2*n)) # 2^(2*4) = 2^8 = 256 = 16x16

    # for each pixel in the image, we asign a value using the sine function with the
    # value of the pixel as an argument.

    # Then, Fill the feature vector with sine-transformed pixel values
    for i in range(2**n): # iterate in the width size (16 pixels)
        for j in range(2**n): # iterate in the height size (16 pixels)
            #features = features.at[2**n * i + j].set(
            #    np.sin(np.pi / 2 * (2 * img[i, j] - 1))
            features[2 ** n * i + j] = np.sin(np.pi / 2 * (2 * img[i, j] - 1))
    # Normalize the feature vector
    features = features / np.sqrt(np.sum(features**2))

    # Use amplitude embedding to encode the feature vector into quantum state
    qml.AmplitudeEmbedding(features, wires=wires)


def data_embedding(X, embedding_type='Amplitude'):
    """
    Embeds the input data X using various embedding types.

    Parameters:
    X (numpy array): An array of dimension (16,16,1).
    embedding_type (str): The type of embedding to use. Options are:
        - "Amplitude"
        - "Equivariant-Amplitude"
        - "Angle"
        - "Angle-compact"
        
    Notes:
    - "Amplitude" reshapes X and uses AmplitudeEmbedding with normalization.
    - "Equivariant-Amplitude" uses equivariant amplitude encoding.
    - "Angle" uses AngleEmbedding with rotation 'Y'. It works only if len(X) = 8.
    - "Angle-compact" uses AngleEmbedding with rotation 'X' and 'Y'. It works only if len(X) = 16.
    """

    if embedding_type == 'Amplitude': 
        X = X.reshape(-1)
        AmplitudeEmbedding(X, wires=range(8), normalize=True)

    # it works if X is an array of dimension (16,16,1)
    elif embedding_type == "Equivariant-Amplitude":
        equivariant_amplitude_encoding(X)
    
    # it works only if len(X) = 8 
    elif embedding_type == 'Angle':
        AngleEmbedding(X, wires=range(8), rotation='Y')
    
    # it works only if len(X) = 16. It's similar to Data-Reuploading
    elif embedding_type == 'Angle-compact':
        AngleEmbedding(X[:8], wires=range(8), rotation='X')
        AngleEmbedding(X[8:16], wires=range(8), rotation='Y')
        