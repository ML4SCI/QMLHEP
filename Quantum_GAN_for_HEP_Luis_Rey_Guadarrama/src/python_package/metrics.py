import numpy as np
from scipy.stats import wasserstein_distance, entropy
from scipy.special import rel_entr
from scipy.linalg import sqrtm


def frechet_distance(real_data, generated_data):
    mu_r = np.mean(real_data)
    mu_g = np.mean(generated_data)
    var_r = np.var(real_data)
    var_g = np.var(generated_data)

    mean_diff = mu_r - mu_g
    cov_mean = np.sqrt(var_r * var_g)

    distance = mean_diff**2 + var_r + var_g - 2 * cov_mean
    return distance


def FID(real_data, generated_data):
    mu_r = np.mean(real_data, axis=0)
    mu_g = np.mean(generated_data, axis=0)
    C_r = np.cov(real_data, rowvar=False)
    C_g = np.cov(generated_data, rowvar=False)

    mean_diff = mu_r - mu_g
    cov_mean = sqrtm(C_r.dot(C_g))

    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    distance = mean_diff.dot(mean_diff) + np.trace(C_r + C_g - 2*cov_mean)
    return distance


def relative_entropy(real_data, generated_data):
    return np.sum(rel_entr(real_data, generated_data))

