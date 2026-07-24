import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot(channel=0):
    '''Plots specified channel of quark-gluon dataset'''
    fig2 = plt.figure(figsize=(10,10))
    r = 1
    c = 2
    index = [np.where(data['y_train'] == 0)[0], np.where(data['y_train'] == 1)[0]]
    for i in range(2):
        fig2.add_subplot(r,c,i+1)
        plt.imshow(np.log(np.mean(data['x_train'][index[i],:,:,channel], axis=0)))
        plt.title('Quark' if i == 0 else 'Gluon')
        
def plot_image_grid_superimposed(data, label, channel=0, rows=5, cols=5):
    '''Plots image grid of the 4th channel'''
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i in range(rows):
        for j in range(cols):
            # axes[i, j].imshow(data[i * cols + j, :, :, channel])
            axes[i, j].imshow(data[i * cols + j, :, :, 3])
            axes[i, j].axis('off')
    plt.suptitle(f'{label}')
    plt.show()

def plot_image_grid(data, label, channel=0, rows=5, cols=5):
    '''Plots image grid of specified channel'''
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(data[i * cols + j, :, :, channel])
            axes[i, j].axis('off')
    plt.suptitle(f'{label}')
    plt.show()

def plot_sample_pairs(pairs, labels, shape, num_samples=5):
    '''Plots sample pairs'''
    plt.figure(figsize=(15, num_samples * 2))
    for i in range(num_samples):
        ax = plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(pairs[i, 0].reshape(shape, shape))
        ax.axis('off')

        ax = plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(pairs[i, 1].reshape(shape, shape))
        ax.axis('off')

        label = labels[i]
        plt.title(f'Sample: {i}, Label: {label}')

    plt.tight_layout()
    plt.show()