import argparse
from data_preprocessing import load_data, get_dataloader
from vanilla_gan import train_vanilla_gan
from wgan import train_wgan
from total_variation import train_total_variation_gan
from perceptual_loss import train_perceptual_gan

def main(args):
    data = load_data(args.dataset_path)
    dataloader = get_dataloader(data, batch_size=args.batch_size)

    if args.model == 'vanilla':
        train_vanilla_gan(dataloader, latent_dim=args.latent_dim, lr=args.lr, n_epochs=args.epochs)
    elif args.model == 'wgan':
        train_wgan(dataloader, latent_dim=args.latent_dim, lr=args.lr, n_epochs=args.epochs)
    elif args.model == 'tv':
        train_total_variation_gan(dataloader, latent_dim=args.latent_dim, lr=args.lr, n_epochs=args.epochs)
    elif args.model == 'perceptual':
        train_perceptual_gan(dataloader, latent_dim=args.latent_dim, lr=args.lr, n_epochs=args.epochs)
    else:
        print(f"Unknown model type: {args.model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--model', type=str, required=True, choices=['vanilla', 'wgan', 'tv', 'perceptual'], help='Model type to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension for the generator')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training')
    args = parser.parse_args()

    main(args)
