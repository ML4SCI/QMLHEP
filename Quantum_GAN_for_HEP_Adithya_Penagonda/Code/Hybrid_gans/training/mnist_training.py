import torch
from torch.optim import Adam
import torch.nn.functional as F
from models.quantum_generator import QuantumGenerator
from models.discriminator import Discriminator
from utils.data_loader import load_mnist_data
from utils.plot_utils import plot_losses, plot_generated_samples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def total_variation_loss(img, weight=1e-5):
    batch_size, _, height, width = img.size()
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return weight * (tv_h + tv_w)

def train_mnist_gan(n_epochs=50, batch_size=64, n_qubits=4):
    trainloader, _ = load_mnist_data(batch_size)
    latent_dim = n_qubits

    gen = QuantumGenerator(n_qubits=n_qubits, depth=3, output_dim=16*16).to(device)
    discriminator = Discriminator().to(device)

    criterion = torch.nn.BCELoss()
    optimizer_gen = Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_disc = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    gen_loss_list = []
    disc_loss_list = []

    for epoch in range(n_epochs):
        for i, (images, _) in enumerate(trainloader):
            optimizer_disc.zero_grad()

            real_images = images.to(device)
            real_labels = torch.ones(real_images.size(0), 1).to(device)
            real_outputs = discriminator(real_images)
            disc_real_loss = criterion(real_outputs, real_labels)

            latent_vectors = torch.randn((batch_size, latent_dim)).to(device)
            fake_images = gen(latent_vectors).view(batch_size, 1, 16, 16)
            fake_labels = torch.zeros(fake_images.size(0), 1).to(device)
            fake_outputs = discriminator(fake_images.detach())
            disc_fake_loss = criterion(fake_outputs, fake_labels)

            disc_loss = disc_real_loss + disc_fake_loss
            disc_loss.backward()
            optimizer_disc.step()

            optimizer_gen.zero_grad()
            fake_outputs = discriminator(fake_images)
            gen_labels = torch.ones(fake_images.size(0), 1).to(device)
            gen_loss = criterion(fake_outputs, gen_labels) + total_variation_loss(fake_images)
            gen_loss.backward()
            optimizer_gen.step()

        gen_loss_list.append(gen_loss.item())
        disc_loss_list.append(disc_loss.item())
        print(f'Epoch [{epoch+1}/{n_epochs}], Discriminator Loss: {disc_loss.item():.4f}, Generator Loss: {gen_loss.item():.4f}')

        if (epoch+1) % 2 == 0:
            plot_generated_samples(gen, latent_dim, num_samples=16, device=device)

    plot_losses(gen_loss_list, disc_loss_list)

if __name__ == '__main__':
    train_mnist_gan()
