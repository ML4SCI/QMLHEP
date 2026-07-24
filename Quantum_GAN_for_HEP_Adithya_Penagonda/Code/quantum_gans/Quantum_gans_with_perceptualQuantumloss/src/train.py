# quantum_gan_with_Perceptual_quantum_loss/src/train.py

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data_preprocessing import load_data, normalize_and_resize, sample_data, apply_pca, JetDataset
from src.model import create_generator_qnode, create_discriminator_qnode, QuantumGAN
from src.utils import perceptual_loss, quantum_fidelity, plot_losses, save_models
from torchvision import models, transforms
import argparse
import pennylane as qml

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Preprocessing
    X_jet = load_data(args.data_path)
    X_jet_resized = normalize_and_resize(X_jet)
    X_jet_sampled = sample_data(X_jet_resized, num_samples=args.num_samples)
    pca, X_pca = apply_pca(X_jet_sampled, n_components=args.pca_components)

    dataset = JetDataset(X_pca)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Define quantum devices
    n_qubits = args.pca_components
    n_layers = args.n_layers
    dev_generator = qml.device('default.qubit', wires=n_qubits)
    dev_discriminator = qml.device('default.qubit', wires=n_qubits)

    # Create QNodes
    generator_qnode = create_generator_qnode(dev_generator, n_qubits)
    discriminator_qnode = create_discriminator_qnode(dev_discriminator, n_qubits)

    # Define weight shapes
    generator_weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    discriminator_weight_shapes = {"weights": (n_layers, n_qubits, 3)}

    # Initialize models
    generator = QuantumGAN(generator_qnode, generator_weight_shapes).to(device)
    discriminator = QuantumGAN(discriminator_qnode, discriminator_weight_shapes).to(device)

    # Define optimizers and loss function
    g_optimizer = optim.Adam(generator.parameters(), lr=args.g_lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.d_lr)
    criterion = nn.BCELoss()

    # Load pre-trained VGG16 model for perceptual loss
    vgg = models.vgg16(pretrained=True).features[:8].eval().to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    # Training Loop
    for epoch in range(args.epochs):
        for i, real_data in enumerate(dataloader):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            real_output = discriminator(real_data)
            real_output = (real_output + 1) / 2  # Map to [0,1]
            d_loss_real = criterion(real_output, real_labels)

            noise = torch.randn(batch_size, n_qubits, device=device)
            fake_data = generator(noise).detach()
            fake_output = discriminator(fake_data)
            fake_output = (fake_output + 1) / 2  # Map to [0,1]
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            noise = torch.randn(batch_size, n_qubits, device=device)
            fake_data = generator(noise)
            fake_output = discriminator(fake_data)
            fake_output = (fake_output + 1) / 2  # Map to [0,1]
            g_loss = criterion(fake_output, real_labels)

            # Compute Perceptual Loss
            ploss = perceptual_loss(real_data, fake_data, vgg, pca)

            # Compute Quantum Fidelity
            q_fidelity = quantum_fidelity(real_data, fake_data, pca)

            # Combine Losses
            alpha = args.alpha
            beta = args.beta
            total_g_loss = g_loss + alpha * ploss - beta * torch.tensor(q_fidelity, device=device)

            total_g_loss.backward()
            g_optimizer.step()

            # Logging
            if (i + 1) % args.log_interval == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(dataloader)}], '
                      f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, '
                      f'Perceptual Loss: {ploss.item():.4f}, Fidelity: {q_fidelity:.4f}')

        # Save models and plot losses after each epoch
        save_models(generator, discriminator, epoch, args)
        plot_losses(epoch, args)

    print("Training Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantum GAN with Perceptual Quantum Loss Training')
    parser.add_argument('--data_path', type=str, default='data/jet-images_Mass60-100_pT250-300_R1.25_Pix25.hdf5', help='Path to the HDF5 data file')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to use')
    parser.add_argument('--pca_components', type=int, default=8, help='Number of PCA components')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in quantum circuits')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--g_lr', type=float, default=0.01, help='Generator learning rate')
    parser.add_argument('--d_lr', type=float, default=0.01, help='Discriminator learning rate')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for perceptual loss')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight for fidelity')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    args = parser.parse_args()
    main(args)
