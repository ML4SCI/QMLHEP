import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import h5py
import random

def load_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def load_jet_data(jet_images_path, batch_size=64):
    with h5py.File(jet_images_path, 'r') as f:
        X_jet = np.array(f['image'])

    X_jet = torch.tensor(X_jet, dtype=torch.float32)
    X_jet = (X_jet - X_jet.min()) / (X_jet.max() - X_jet.min())
    X_jet = X_jet.unsqueeze(1)
    X_jet_resized = torch.nn.functional.interpolate(X_jet, size=(16, 16), mode='bilinear', align_corners=False)

    indices = random.sample(range(X_jet_resized.shape[0]), 100000)
    X_jet_sampled = X_jet_resized[indices]
    jet_dataset = TensorDataset(X_jet_sampled)
    jet_loader = DataLoader(jet_dataset, batch_size=batch_size, shuffle=True)
    return jet_loader
