import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data_preprocessing import load_data, get_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.init_size = 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 1 * 1, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)  # Flatten the output
        validity = self.fc(out)
        return validity

def train_vanilla_gan(dataloader, latent_dim=100, lr=0.0002, b1=0.5, b2=0.999, n_epochs=20):
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    adversarial_loss = nn.BCELoss()

    d_losses = []
    g_losses = []

    for epoch in range(n_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        for i, (imgs,) in enumerate(dataloader):
            real_imgs = imgs.to(device)
            optimizer_D.zero_grad()

            z = torch.randn(imgs.shape[0], latent_dim).to(device)
            fake_imgs = generator(z)
            valid = torch.ones(imgs.size(0), 1).to(device) * 0.9
            fake = torch.zeros(imgs.size(0), 1).to(device)

            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            fake_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(fake_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

        epoch_d_loss /= len(dataloader)
        epoch_g_loss /= len(dataloader)
        d_losses.append(epoch_d_loss)
        g_losses.append(epoch_g_loss)

        print(f"[Epoch {epoch+1}/{n_epochs}] [D loss: {epoch_d_loss:.4f}] [G loss: {epoch_g_loss:.4f}]")

    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('GAN Training Loss')
    plt.legend()
    plt.show()

    print("Warning: This model may experience mode collapse.")

if __name__ == "__main__":
    jet_images_path = 'path/to/jet-images_Mass60-100_pT250-300_R1.25_Pix25.hdf5'
    data = load_data(jet_images_path)
    dataloader = get_dataloader(data)
    train_vanilla_gan(dataloader)
