import matplotlib.pyplot as plt
import torch

def plot_losses(gen_losses, disc_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Loss During Training')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_generated_samples(generator, latent_dim, num_samples=16, device='cpu'):
    z = torch.randn(num_samples, latent_dim).to(device)
    gen_samples = generator(z).detach().cpu()
    gen_samples = (gen_samples + 1) / 2.0

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    for i in range(num_samples):
        img = gen_samples[i, 0].numpy()
        axes[i].imshow(img, cmap='viridis')
        axes[i].axis('off')
    plt.show()
