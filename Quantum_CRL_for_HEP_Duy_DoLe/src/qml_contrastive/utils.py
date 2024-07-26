import torch, kornia
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt


def contrastive_loss_with_margins(embeddings, labels, pos_margin=0.25, neg_margin=1.5):
    """
    Custom contrastive loss function with positive and negative margins.
    
    Args:
        embeddings (Tensor): Embedding vectors.
        labels (Tensor): Corresponding labels.
        pos_margin (float): Margin for positive pairs.
        neg_margin (float): Margin for negative pairs.

    Returns:
        Tensor: Calculated loss.
    """
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)
    positive_loss = (1 - labels) * F.relu(distance_matrix - pos_margin).pow(2)
    negative_loss = labels * F.relu(neg_margin - distance_matrix).pow(2)
    combined_loss = 0.5 * (positive_loss + negative_loss)
    mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    combined_loss = combined_loss.masked_fill_(mask, 0)
    loss = combined_loss.mean()
    return loss

def get_preprocessing(preprocess):
    """
    Get preprocessing transformation based on the specified type.

    Args:
        preprocess (str): Type of preprocessing.

    Returns:
        Callable: Preprocessing transformation.
    """
    if preprocess == "RandAffine":
        return kornia.augmentation.RandomAffine(degrees=(-40, 40), translate=0.25, scale=[0.5, 1.5], shear=45)
    elif preprocess == "RandAug":
        return T.RandAugment()
    return None

def pca_proj(embeddings, labels, seed=1):
    """
    Perform PCA projection and plot the results.

    Args:
        embeddings (np.ndarray): Embedding vectors.
        labels (np.ndarray): Corresponding labels.
        seed (int): Random seed for reproducibility.
    """
    proj = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=labels, palette=sns.color_palette("tab10")).set(title="PCA")
    plt.show()

def tsne_proj(embeddings, labels, seed=1):
    """
    Perform t-SNE projection and plot the results.

    Args:
        embeddings (np.ndarray): Embedding vectors.
        labels (np.ndarray): Corresponding labels.
        seed (int): Random seed for reproducibility.
    """
    proj = TSNE(n_components=2, random_state=seed).fit_transform(embeddings)
    sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=labels, palette=sns.color_palette("tab10")).set(title="T-SNE")
    plt.show()

def gaussian_kde_2d(embeddings, labels):
    """
    Plot Gaussian KDE for embeddings in R2.

    Args:
        embeddings (np.ndarray): Embedding vectors.
        labels (np.ndarray): Corresponding labels.
    """
    sns.kdeplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=labels, fill=True, palette=sns.color_palette("tab10"))
    plt.title("Gaussian KDE in R2")
    plt.show()

def vmf_kde_angles(embeddings, labels, bins=100):
    """
    Plot von Mises-Fisher KDE for angles.

    Args:
        embeddings (np.ndarray): Embedding vectors.
        labels (np.ndarray): Corresponding labels.
        bins (int): Number of bins for histogram.
    """
    angles = np.arctan2(embeddings[:, 1], embeddings[:, 0])
    unique_labels = np.unique(labels)

    for label in unique_labels:
        label_angles = angles[labels == label]
        sns.kdeplot(label_angles, fill=True, label=f"Label {label}", bw_adjust=0.5)

    plt.title("von Mises-Fisher KDE on Angles")
    plt.xlabel("Angle (radians)")
    plt.legend()
    plt.show()

def vmf_kde_on_circle(embeddings, labels):
    """
    Plot embeddings as a scatter plot on a circle.

    Args:
        embeddings (np.ndarray): Embedding vectors.
        labels (np.ndarray): Corresponding labels.
    """
    angles = np.arctan2(embeddings[:, 1], embeddings[:, 0])
    radii = np.ones_like(angles)  # Set radius to 1 for all points
    unique_labels = np.unique(labels)

    ax = plt.subplot(111, projection='polar')
    for label in unique_labels:
        label_angles = angles[labels == label]
        ax.scatter(label_angles, radii[labels == label], label=f"Label {label}", alpha=0.75)

    ax.set_title("Scatter VMF KDE Plot")
    ax.set_ylim(0, 1.5)  # Extend the radius slightly for better visualization
    ax.set_yticks([])  # Remove radial ticks
    ax.legend()
    plt.show()

def plot_training(logdir):
    # Plot training and validation loss
    metrics_df = pd.read_csv(f"{logdir}/metrics.csv")
    train_loss_epoch = metrics_df['train_loss'].dropna().reset_index(drop=True)
    val_loss_epoch = metrics_df['val_loss'].dropna().reset_index(drop=True)
    min_length = min(len(train_loss_epoch), len(val_loss_epoch))
    train_loss_epoch = train_loss_epoch[:min_length]
    val_loss_epoch = val_loss_epoch[:min_length]

    plt.plot(train_loss_epoch, label='Train Loss')
    plt.plot(val_loss_epoch, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_embeddings(model, data_loader):
    """
    Generate embeddings for the given data using the provided model.

    Args:
        model (nn.Module): Trained model.
        data_loader (DataLoader): Data loader for the dataset.

    Returns:
        tuple: Embeddings and labels as numpy arrays.
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x = model.encoder(x)
            emb = model.head(x)
            embeddings.append(emb)
            labels.append(y)
    
    embeddings = torch.cat(embeddings).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    
    return embeddings, labels

# def swap_test_circuit(embedding1, embedding2):
#     """
#     Defines a quantum circuit for the SWAP test to compare two embeddings.
    
#     Args:
#         embedding1 (np.ndarray): First embedding vector.
#         embedding2 (np.ndarray): Second embedding vector.

#     Returns:
#         Callable: Quantum circuit function.
#     """
#     num_qubits_per_embedding = int(np.ceil(np.log2(len(embedding1))))
#     total_num_qubits = 2 * num_qubits_per_embedding + 1  # +1 for the ancilla qubit

#     dev = qml.device('default.qubit', wires=total_num_qubits)
    
#     @qml.qnode(dev)
#     def circuit(embedding1, embedding2):
#         qml.Hadamard(wires=0)
#         qml.AmplitudeEmbedding(features=embedding1, wires=range(1, num_qubits_per_embedding + 1), normalize=True)
#         qml.AmplitudeEmbedding(features=embedding2, wires=range(num_qubits_per_embedding + 1, 2 * num_qubits_per_embedding + 1), normalize=True)
        
#         for i in range(1, num_qubits_per_embedding + 1):
#             qml.CSWAP(wires=[0, i, i + num_qubits_per_embedding])
#         qml.Hadamard(wires=0)
        
#         return qml.expval(qml.PauliZ(0))
    
#     return circuit(embedding1, embedding2)
