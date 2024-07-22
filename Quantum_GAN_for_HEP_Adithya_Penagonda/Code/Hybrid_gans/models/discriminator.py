import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        self.conv3 = spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        self.fc1 = spectral_norm(nn.Linear(256 * 4 * 4, 1))

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))
        return x
