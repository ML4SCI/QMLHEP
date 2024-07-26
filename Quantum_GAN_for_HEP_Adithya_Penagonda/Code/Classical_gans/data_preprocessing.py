import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
from h5py import File as HDF5File

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_data(filepath):
    jet_mass_data = HDF5File(filepath, 'r')
    X_jet = jet_mass_data['image']
    X_jet = np.array(X_jet)
    X_jet = torch.tensor(X_jet, dtype=torch.float32)

    # Normalize the data
    X_jet = (X_jet - X_jet.min()) / (X_jet.max() - X_jet.min())

    X_jet = X_jet.unsqueeze(1)
    X_jet_resized = nn.functional.interpolate(X_jet, size=(16, 16), mode='bilinear', align_corners=False)
    return X_jet_resized

def get_dataloader(data, batch_size=128):
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == "__main__":
    jet_images_path = 'path/to/jet-images_Mass60-100_pT250-300_R1.25_Pix25.hdf5'
    data = load_data(jet_images_path)
    dataloader = get_dataloader(data)
