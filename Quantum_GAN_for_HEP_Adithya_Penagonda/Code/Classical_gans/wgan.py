import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from data_preprocessing import load_data, get_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class WGANGenerator(nn.Module):
    def __init__(self, latent_dim):
        super(WGANGenerator, self).__init__()
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(64, 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class WGANDiscriminator(nn.Module):
    def __init__(self):
        super(WGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 64, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Flatten(),
            spectral_norm(nn.Linear(128*4*4, 1, bias=False))
        )

    def forward(self, x):
        return self.model(x)

def gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates)
    fake = torch.ones(d_interpolates.size(), device=real_samples.device, requires_grad=False)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def calculate_fid(real_images, fake_images, batch_size=128):
    inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    def get_features(images):
        features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            batch = batch.repeat(1, 3, 1, 1)
            with torch.no_grad():
                pred = inception_model(batch)
            features.append(pred.cpu().numpy())
        return np.concatenate(features, axis=0)

    real_features = get_features(real_images)
    fake_features = get_features(fake_images)

    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    if sigma1.ndim == 0:
        sigma1 = np.array([[sigma1]])
    if sigma2.ndim == 0:
        sigma2 = np.array([[sigma2]])

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

latent_dim = 100
lr = 0.0002
n_epochs = 30
lambda_gp = 10

generator = WGANGenerator(latent_dim).to(device)
discriminator = WGANDiscriminator().to(device)

optimizer_G = optim.RMSprop(generator.parameters(), lr=lr)
optimizer_D = optim.RMSprop(discriminator.parameters(), lr=lr)

g_losses = []
d_losses = []
fid_scores = []

def plot_generated_samples(generator, latent_dim, num_samples=16):
    z = torch.randn(num_samples, latent_dim, 1, 1).to(device)
    gen_samples = generator(z).detach().cpu()
    gen_samples = (gen_samples + 1) / 2.0

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples, 1))
    for i in range(num_samples):
        axes[i].imshow(gen_samples[i, 0])
        axes[i].axis('off')
    plt.show()

def plot_metrics(g_losses, d_losses, fid_scores):
    epochs = range(1, len(g_losses) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, g_losses, label='Generator Loss')
    plt.plot(epochs, d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Generator and Discriminator Losses')

    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(fid_scores)), fid_scores, label='FID Score')
    plt.xlabel('Epoch')
    plt.ylabel('FID Score')
    plt.legend()
    plt.title('FID Scores')

    plt.show()


for epoch in range(n_epochs):
    for i, (real_samples,) in enumerate(dataloader):
        real_samples = real_samples.to(device)
        batch_size = real_samples.size(0)

        optimizer_D.zero_grad()

        z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_samples = generator(z).detach()
        real_validity = discriminator(real_samples)
        fake_validity = discriminator(fake_samples)

        gp = gradient_penalty(discriminator, real_samples, fake_samples)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

        d_loss.backward()
        optimizer_D.step()

        if i % 5 == 0:
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            gen_samples = generator(z)
            gen_validity = discriminator(gen_samples)
            g_loss = -torch.mean(gen_validity)

            g_loss.backward()
            optimizer_G.step()

    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())

    z = torch.randn(1000, latent_dim, 1, 1).to(device)
    fake_images = generator(z).detach().cpu()
    real_images = next(iter(dataloader))[0][:1000].cpu()

    fid = calculate_fid(real_images, fake_images, batch_size=batch_size)
    fid_scores.append(fid)
    print(f"Epoch [{epoch}/{n_epochs}] D loss: {d_loss.item()} G loss: {g_loss.item()} FID: {fid}")

    output_image = fake_images[:16]  # Save 16 generated samples
    output_image = (output_image + 1) / 2.0  # Scale to [0, 1]
    grid = torchvision.utils.make_grid(output_image, nrow=4, normalize=True)
    np_img = grid.permute(1, 2, 0).numpy()
    plt.imsave(f'classical_Jet_image_epoch_{epoch}.png', np_img)
    plot_generated_samples(generator, latent_dim)

plot_metrics(g_losses, d_losses, fid_scores)
