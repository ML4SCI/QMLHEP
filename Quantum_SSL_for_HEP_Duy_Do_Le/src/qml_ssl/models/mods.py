
import torch
# import kornia
import torch.nn.functional as F
import torchvision.transforms as T


def get_mnist_augmentations():
    return T.Compose([
        # Apply rotation (small angles to avoid too much distortion)
        T.RandomRotation(degrees=10),  
        
        # Apply translation (shift up/down, left/right)
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
        
        # Apply shear (mimic slanted writing)
        T.RandomAffine(degrees=0, shear=10),
        
        # Apply scaling (zoom in/out)
        T.RandomAffine(degrees=0, scale=(0.8, 1.2)),
    ])

