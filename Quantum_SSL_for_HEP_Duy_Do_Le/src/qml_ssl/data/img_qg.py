import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset

def load_qg_img(x, y, reduced_dim=None, dataset_size=-1):
    """
    Load and preprocess electron and photon data.

    Args:
        x (np.array): Image data with 3 channels.
        y (np.array): Labels.
        reduced_dim (int): Size to resize the images to (default is None).
        dataset_size (tuple): Custom dataset size.

    Returns:
        dict: A dictionary with the preprocessed training, validation, and test datasets.
    """

    x = x[:dataset_size]
    y = y[:dataset_size]

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=True)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, shuffle=False)

    if reduced_dim:
        resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((reduced_dim, reduced_dim), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),  # Convert to tensor and scales to [0, 1] (H, W, C -> C, H, W)
        ])
    else:
        resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

    # x = torch.stack([resize_transform(img) for img in x])
    x_train = torch.stack([resize_transform(img) for img in x_train])
    x_val = torch.stack([resize_transform(img) for img in x_val])
    x_test = torch.stack([resize_transform(img) for img in x_test])

    # return {
    #     "data": x,
    #     "labels": torch.tensor(y)
    # }

    return {
        "train_data": x_train,  # Already in (C, H, W) format
        "val_data": x_val,
        "test_data": x_test,
        "train_labels": torch.tensor(y_train),
        "val_labels": torch.tensor(y_val),
        "test_labels": torch.tensor(y_test)
    }


def reduce_resolution(image):
    image = np.array(image)
    
    assert image.shape[0] % 5 == 0 and image.shape[1] % 5 == 0, "Image dimensions should be divisible by 5"
    
    # Reshape the image into 25x25x5x5
    reduced_image = image.reshape(25, 5, 25, 5).mean(axis=(1, 3))
    
    return reduced_image

def reduce_resolution_batch(images):
    return np.array([reduce_resolution(img) for img in images])

def visualize_image(image, label):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    im = axs[0].imshow(image[:, :, 0], cmap='binary')
    axs[0].title.set_text(f'Class {label} - Tracks')

    im = axs[1].imshow(image[:, :, 1],  cmap='binary')
    axs[1].title.set_text(f'Class {label} - ECAL')

    im = axs[2].imshow(image[:, :, 2],  cmap='binary')
    axs[2].title.set_text(f'Class {label} - HCAL')

def visualize_average_images(x_data, y_data, num=-1, use_lognorm=False):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    norm = LogNorm() if use_lognorm else None
    # Calculate average images for each class and channel
    avg_images = {}
    for class_label in [0, 1]:
        avg_images[class_label] = []
        class_data = x_data[y_data == class_label]
        for channel in range(3):
            # print(len(class_data))
            avg_image = np.average(class_data[:num, :, :, channel], 0)
            avg_images[class_label].append(avg_image)
    
    # Plot for class 0
    im = axs[0, 0].imshow(avg_images[0][0], norm = norm, cmap='binary')
    axs[0, 0].title.set_text('Class 0 - Channel 0')
    fig.colorbar(im, ax=axs[0, 0])

    im = axs[0, 1].imshow(avg_images[0][1], norm = norm, cmap='binary')
    axs[0, 1].title.set_text('Class 0 - Channel 1')
    fig.colorbar(im, ax=axs[0, 1])

    # im = axs[0, 2].imshow(reduce_resolution(avg_images[0][2]), norm=LogNorm(), cmap='binary')
    im = axs[0, 2].imshow(avg_images[1][2], norm = norm, cmap='binary')
    axs[0, 2].title.set_text('Class 0 - Channel 2')
    fig.colorbar(im, ax=axs[0, 2])

    # Plot for class 1
    im = axs[1, 0].imshow(avg_images[1][0], norm = norm, cmap='binary')
    axs[1, 0].title.set_text('Class 1 - Channel 0')
    fig.colorbar(im, ax=axs[1, 0])

    im = axs[1, 1].imshow(avg_images[1][1], norm = norm, cmap='binary')
    axs[1, 1].title.set_text('Class 1 - Channel 1')
    fig.colorbar(im, ax=axs[1, 1])
    # print(avg_images[1][2].shape)
    im = axs[1, 2].imshow(reduce_resolution(avg_images[1][2]), norm = norm, cmap='binary')
    axs[1, 2].title.set_text('Class 1 - Channel 2')
    fig.colorbar(im, ax=axs[1, 2])

    fig.tight_layout()
    plt.show()


def visualize_diff_average_images(x_data, y_data, num=-1):
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    # Calculate average images for each class and channel
    avg_images = {}
    for class_label in [0, 1]:
        avg_images[class_label] = []
        class_data = x_data[y_data == class_label]
        for channel in range(3):
            # print(len(class_data))
            avg_image = np.average(class_data[:num, :, :, channel], 0)
            avg_images[class_label].append(avg_image)

    im = axs[0].imshow(avg_images[1][0]/avg_images[0][0], norm=LogNorm(), cmap='binary')
    axs[0].title.set_text('Class 1 / Class 0 - Channel 0')
    fig.colorbar(im, ax=axs[0])

    im = axs[1].imshow(avg_images[1][1]/avg_images[0][1], norm=LogNorm(), cmap='binary')
    axs[1].title.set_text('Class 1 / Class 0 - Channel 1')
    fig.colorbar(im, ax=axs[1])

    im = axs[2].imshow(avg_images[1][2]/avg_images[0][2], norm=LogNorm(), cmap='binary')
    axs[2].title.set_text('Class 1 / Class 0 - Channel 2')
    fig.colorbar(im, ax=axs[2])

    fig.tight_layout()
    plt.show()
    

import torch_geometric.data as pyg_data

class QG_Images(pyg_data.InMemoryDataset):
    def __init__(self, images, labels, channel=None, root='../data/QG_Images', transform=None, pre_transform=None, force_reload=True):
        self.images = images
        self.labels = labels
        self.channel = channel if channel else "x"
            
        super().__init__(root, transform, pre_transform, force_reload=force_reload)

        self.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'data_{self.channel}.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        for index, img in enumerate(self.images):
            data = self.image_to_graph(img, self.labels[index])
            data_list.append(data)

        # self.data, self.slices = self.collate(data_list)
        # torch.save((self.data, self.slices), self.processed_paths[0])

        self.save(data_list, self.processed_paths[0])
        return data_list

    def image_to_graph(self, image, label):
        # Find non-zero pixels (nodes)
        y_coords, x_coords = np.nonzero(image)
        intensities = image[y_coords, x_coords]
        assert len(intensities != 0)

        # Create node features (intensity, x-coord, y-coord)
        # node_features = 
        coords = np.stack((x_coords, y_coords), axis=1)

        # Convert to PyTorch tensors
        node_features = torch.tensor(intensities, dtype=torch.float).unsqueeze(1)
        coords = torch.tensor(coords, dtype=torch.float)

        # Create PyTorch Geometric Data object with node features
        data = pyg_data.Data(x=node_features, pos=coords, num_nodes=node_features.shape[0], 
                             y=torch.tensor([label], dtype=torch.float))

        return data
    