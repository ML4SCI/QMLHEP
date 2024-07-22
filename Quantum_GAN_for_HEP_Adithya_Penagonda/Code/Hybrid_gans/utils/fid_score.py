import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
import numpy as np
from scipy.linalg import sqrtm

def calculate_fid(real_images, fake_images, device):
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    real_images = F.interpolate(real_images, size=(299, 299), mode='bilinear')
    fake_images = F.interpolate(fake_images, size=(299, 299), mode='bilinear')

    real_images = real_images.repeat(1, 3, 1, 1)
    fake_images = fake_images.repeat(1, 3, 1, 1)

    def get_activations(images):
        activations = []
        with torch.no_grad():
            for img in images:
                act = inception_model(img.unsqueeze(0)).view(-1)
                activations.append(act.cpu().numpy())
        return np.array(activations)

    real_activations = get_activations(real_images)
    fake_activations = get_activations(fake_images)

    mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_activations, axis=0), np.cov(fake_activations, rowvar=False)

    fid_score = np.sum((mu_real - mu_fake) ** 2) + np.trace(sigma_real + sigma_fake - 2 * sqrtm(np.dot(sigma_real, sigma_fake)))
    return fid_score
