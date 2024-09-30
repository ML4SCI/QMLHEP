import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.model_selection import train_test_split

class PE_IMG_Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

def inspect_h5py_file(file_path):
    """
    Inspect the contents and dimensions of an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:
        print(f"Inspecting HDF5 file: {file_path}")
        
        def print_attrs(name, obj):
            print(f"Name: {name}")
            for key, val in obj.attrs.items():
                print(f"    Attribute: {key} = {val}")
            if isinstance(obj, h5py.Dataset):
                print(f"    Shape: {obj.shape}")
                print(f"    Dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"    Group contains: {list(obj.keys())}")

        f.visititems(print_attrs)

def load_pe_img(electron_file, photon_file, reduced_dim=None, dataset_size=-1, channel=None):
    """
    Load and preprocess electron and photon data.

    Args:
        electron_file (str): Path to the electron data file.
        photon_file (str): Path to the photon data file.
        reduced_dim (int): Size to resize the images to (default is None).
        dataset_size (int): Custom dataset size.

    Returns:
        dict: A dictionary with the preprocessed training, validation, and test datasets.
    """
    f_electron = h5py.File(electron_file, "r")
    f_photon = h5py.File(photon_file, "r")

    # print(f_electron['X'].shape, f_photon['X'].shape)
    # print(f_electron['y'].shape, f_photon['y'].shape)
    
    if channel == None:
        electrons = f_electron['X'][:, :, :, :][:dataset_size]
        photons = f_photon['X'][:, :, :, :][:dataset_size]
        electrons_y = f_electron['y'][:][:dataset_size]
        photons_y = f_photon['y'][:][:dataset_size]
    elif channel in (0, 1): 
        electrons = f_electron['X'][:, :, :, channel][:dataset_size]
        photons = f_photon['X'][:, :, :, channel][:dataset_size]
        electrons_y = f_electron['y'][:][:dataset_size]
        photons_y = f_photon['y'][:][:dataset_size]
    else:
        raise ValueError("Channel must be either 0 or 1.")

    x = np.vstack((electrons, photons))
    y = np.hstack((electrons_y, photons_y))

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=True)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, shuffle=False)

    if reduced_dim:
        resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((reduced_dim, reduced_dim), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor()
        ])
        x_train = np.stack([resize_transform(img) for img in x_train])
        x_val = np.stack([resize_transform(img) for img in x_val])
        x_test = np.stack([resize_transform(img) for img in x_test])

    train_dataset = PE_IMG_Dataset(x_train, y_train)
    val_dataset = PE_IMG_Dataset(x_val, y_val)
    test_dataset = PE_IMG_Dataset(x_test, y_test)

    return {
        "train_data": torch.tensor(x_train),
        "val_data": torch.tensor(x_val),
        "test_data": torch.tensor(x_test),
        "train_labels": torch.tensor(y_train),
        "val_labels": torch.tensor(y_val),
        "test_labels": torch.tensor(y_test)
    }

def visualize_data(data, labels, classes=("P", "E"), title=""):
    """
    Visualize the dataset.

    Args:
        data (torch.Tensor): Dataset images.
        labels (torch.Tensor): Corresponding labels.
        classes (tuple): Tuple of class labels to visualize.
        title (str): Title of the plot (default is "").
    """
    fig, axs = plt.subplots(1, len(classes), figsize=(8, 4))

    for i, cls in enumerate(classes):
        class_data = data[labels == i]
        if len(class_data) > 0:
            axs[i].imshow(class_data[0].squeeze(), cmap='binary')
            axs[i].set_title(f"Class {cls}")
        else:
            axs[i].set_title(f"Class {cls} (No data)")
    fig.suptitle(title)
    plt.show()

if __name__ == "__main__":
    # Define the paths and size
    electron_file = "../../data/PhotonElectron/electron.hdf5"
    photon_file = "../../data/PhotonElectron/photon.hdf5"
    
    inspect_h5py_file(electron_file)
    inspect_h5py_file(photon_file)
    
    reduced_dim = 16
    dataset_size = 1000

    # Load and preprocess the particle data
    particle_data = load_pe_img(electron_file, photon_file, reduced_dim=16, dataset_size=dataset_size, channel=1)
    
    # Visualize the preprocessed images
    train_data = particle_data["train_dataset"].data
    train_labels = particle_data["train_dataset"].labels
    visualize_data(train_data, train_labels, classes=(0, 1), title="Preprocessed Particle Data")
