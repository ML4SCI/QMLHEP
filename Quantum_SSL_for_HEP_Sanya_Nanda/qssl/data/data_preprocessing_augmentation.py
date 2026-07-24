import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch_geometric.utils import subgraph



def add_fourth_channel(images):
    '''4th channel from overlay of the 3 channels'''
    images_with_four_channels = []
    for image in images:
        superimposed_channel = np.mean(image, axis=2, keepdims=True)
        image_with_four_channels = np.concatenate((image, superimposed_channel), axis=2)
        images_with_four_channels.append(image_with_four_channels)
    return np.array(images_with_four_channels)

# Soft Transformations
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def translate_image(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def scale_image(image, scale_factor):
    h, w = image.shape[:2]
    resized = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))
    return resized

def shear_image(image, shear_factor):
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    sheared = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return sheared

def adjust_brightness(image, brightness_factor):
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, brightness_factor)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    image_bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    image[:, :, :3] = image_bright
    return image

def add_noise(image, noise_factor):
    row, col, ch = image.shape
    mean = 0
    sigma = noise_factor ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy


# collinear transformations
def affine_transform(image, pts1, pts2):
    M = cv2.getAffineTransform(pts1, pts2)
    rows, cols, ch = image.shape
    affine_transformed = cv2.warpAffine(image, M, (cols, rows))
    return affine_transformed

def perspective_transform(image, pts1, pts2):
    M = cv2.getPerspectiveTransform(pts1, pts2)
    rows, cols, ch = image.shape
    perspective_transformed = cv2.warpPerspective(image, M, (cols, rows))
    return perspective_transformed


# Normalisation
def z_score():
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(image)
    std = np.std(image)
    z_score_transformed = (image - mean) / std
    normalized_transformed = cv2.normalize(z_score_transformed, None, 0, 255, cv2.NORM_MINMAX)
    normalized_transformed = normalized_transformed.astype(np.uint8)
    return normalized_transformed


# Heatmaps
def compute_relative_difference(images):
    avgs = np.mean(images, axis=0)
    relative_diffs = np.zeros_like(avgs)
    for img in images:
        relative_diffs += np.abs(img - avgs) / (avgs + 1e-10)  # Adding a small constant to avoid division by zero
    relative_diffs /= len(images)
    return relative_diffs

def plot_heatmaps(relative_diffs):
    h, w, c = relative_diffs.shape
    for i in range(c):
        plt.figure(figsize=(10, 8))
        sns.heatmap(relative_diffs[:, :, i], cmap="viridis")
        plt.title(f'Channel {i+1} Sensitivity Heatmap')
        plt.show()


# Log transformation of the 4th channel
def preprocess_4th_channel(image):    
    if image.shape[-1] < 3:
        raise ValueError("Image must have at least 4 channels.")
    fourth_channel = image[:, :, 3]
    transformed_channel = np.log(np.abs(fourth_channel) + 1e-6)
    normalized_channel = (transformed_channel - transformed_channel.min()) / (transformed_channel.max() - transformed_channel.min() + 1e-6)
    return normalized_channel


def preprocess_all_images(images):
    if images.shape[-1] < 4:
        raise ValueError("Each image must have at least 4 channels.")

    # Extract the 4th channels from all images
    fourth_channels = images[:, :, :, 3]

    # Apply logarithmic transformation and normalization to the 4th channels
    transformed_channels = np.log(np.abs(fourth_channels) + 1e-6)
    min_vals = transformed_channels.min(axis=(1, 2), keepdims=True)
    max_vals = transformed_channels.max(axis=(1, 2), keepdims=True)
    normalized_channels = (transformed_channels - min_vals) / (max_vals - min_vals + 1e-6)

    return normalized_channels


# graph augmentations
# Function to randomly drop nodes in a graph
def drop_nodes(data, drop_prob=0.2):
    node_mask = torch.rand(data.x.size(0)) > drop_prob
    data.x = data.x[node_mask]
    data.edge_index, _ = subgraph(node_mask, data.edge_index, relabel_nodes=True)
    return data

# Function to randomly drop edges in a graph
def drop_edges(data, drop_prob=0.2):
    edge_mask = torch.rand(data.edge_index.size(1)) > drop_prob
    data.edge_index = data.edge_index[:, edge_mask]
    return data

# Function to randomly mask node features in a graph
def mask_features(data, mask_prob=0.2):
    feature_mask = torch.rand(data.x.size()) > mask_prob
    data.x = data.x * feature_mask.float()
    return data

# Define the augmentation function
def graph_augment(data):
    data_aug = data.clone()
    data_aug = drop_nodes(data_aug, drop_prob=0.2)
    data_aug = drop_edges(data_aug, drop_prob=0.1)
    data_aug = mask_features(data_aug, mask_prob=0.2)
    return data_aug
