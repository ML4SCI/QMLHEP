import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Hybrid GAN Training")
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'jet'], help='Dataset to use for training (mnist or jet)')
    args = parser.parse_args()

    if args.dataset == 'mnist':
        from training.mnist_training import train_mnist_gan
        print("Training on MNIST dataset...")
        train_mnist_gan()
    elif args.dataset == 'jet':
        from training.jet_training import train_jet_gan
        print("Training on Jet Images dataset...")
        train_jet_gan()
    else:
        print("Unknown dataset!")

if __name__ == '__main__':
    main()
