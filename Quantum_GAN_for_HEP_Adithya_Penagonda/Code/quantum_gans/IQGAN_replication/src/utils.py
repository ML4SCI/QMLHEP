# src/utils.py

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def plot_and_save_graphs(loss_list, output_list, epoch, output_dir='outputs/plots'):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_list, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(output_list, label='Output')
    plt.xlabel('Epoch')
    plt.ylabel('Output')
    plt.title('Output over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/output_Jet_IQGAN_epoch_{epoch}.png')
    plt.close()

def descale_points(d_point, scales, tfrm):
    for col in range(d_point.shape[1]):
        d_point[:, col] *= scales[col][1]
        d_point[:, col] += scales[col][0]
    reconstruction = tfrm.inverse_transform(d_point)
    return reconstruction

class Logloss(nn.Module):
    def __init__(self):
        super(Logloss, self).__init__()

    def forward(self, output):
        return torch.mean(-torch.log(output[0]))

def save_sample_images(model, epoch, pca, pca_descaler, output_dir='outputs/plots', sample_interval=1, device='cpu'):
    if epoch % sample_interval == 0:
        num_samples = 10
        sample_indices = np.random.choice(len(pca.data_), num_samples, replace=False)
        samples = pca.data_[sample_indices]
        descaled_samples = descale_points(samples, pca_descaler, pca)

        fig, axes = plt.subplots(1, num_samples, figsize=(15, 2))
        for i, sample in enumerate(descaled_samples):
            axes[i].imshow(sample.reshape(16, 16), cmap='viridis')
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/IQGAN_jet_epoch_{epoch}.png')
        plt.close()
