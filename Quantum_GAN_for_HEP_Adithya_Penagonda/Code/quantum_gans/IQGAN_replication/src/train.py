# src/train.py

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.data_preprocessing import load_data, normalize_and_resize, sample_data, apply_pca, normalize_pca
from src.model import create_qnode, QuantumGAN
from src.utils import Logloss, plot_and_save_graphs, save_sample_images
import argparse

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Preprocessing
    X_jet = load_data(args.data_path)
    X_jet_resized = normalize_and_resize(X_jet)
    X_jet_sampled = sample_data(X_jet_resized, num_samples=args.num_samples)
    pca, pca_data = apply_pca(X_jet_sampled, n_components=args.pca_components)
    pca_data_rot, pca_descaler = normalize_pca(pca_data, n_components=args.pca_components)

    # PCA Transformer
    pca.data_ = pca_data  # Attach data to PCA object for utility functions

    # Quantum Model Setup
    qnode = create_qnode(n_qubits=5)
    weight_shapes = {f"w{key}": 1 for key in ['000', '001', '008', '009', '016', '017', '200', '201', '208', '209', '216', '217']}
    model = QuantumGAN(qnode, weight_shapes).to(device)

    # Training Setup
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_list = []
    output_list = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        loss_sum = 0.0
        output_sum = 0.0
        for i in range(len(pca_data_rot)):
            noise = torch.empty(2, dtype=torch.float32).uniform_(0.3, 0.9).to(device)
            inputs = torch.Tensor(pca_data_rot[i]).to(device)
            inputs = torch.cat((noise, inputs), 0)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_function = Logloss()
            loss = loss_function(outputs)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            running_loss += loss.item()
            output_sum += outputs[0].item()
            if (i + 1) % 200 == 0:
                print(f'epoch: {epoch}, loss: {running_loss / 200:.3f}')
                running_loss = 0.0

        scheduler.step()
        this_epoch_loss = loss_sum / len(pca_data_rot)
        this_epoch_output = output_sum / len(pca_data_rot)
        loss_list.append(this_epoch_loss)
        output_list.append(this_epoch_output)

        # Logging and Saving
        plot_and_save_graphs(loss_list, output_list, epoch)
        save_sample_images(model, epoch, pca, pca_descaler, device=device)

        print(f"Epoch {epoch} completed. Loss: {this_epoch_loss:.4f}, Output: {this_epoch_output:.4f}")

    # Save final model
    torch.save(model.state_dict(), 'outputs/models/quantum_gan_final.pth')
    print("Training completed and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Quantum GAN')
    parser.add_argument('--data_path', type=str, default='data/jet-images_Mass60-100_pT250-300_R1.25_Pix25.hdf5', help='Path to the HDF5 data file')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to use')
    parser.add_argument('--pca_components', type=int, default=2, help='Number of PCA components')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    args = parser.parse_args()
    main(args)
