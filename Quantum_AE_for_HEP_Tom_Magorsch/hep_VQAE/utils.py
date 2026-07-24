"""
Utility functions mainly for data preperation and evaluation
"""

import energyflow as ef
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm
from matplotlib.colors import LogNorm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve


def test_collapse(x_true, x_recon):
    """Test for mode collapse. Ratio of the difference of reconstructed and true images. Bad sign if close to zero

    Args:
        x_true (array): True images
        x_recon (array): reconstructed image

    Returns:
        Value to check the for mode collapse, small values are worse
    """
    p = np.random.permutation(x_true.shape[0])
    x_true_shuffle = x_true[p]
    x_recon_shuffle = x_recon[p]

    true_diff = np.sum(np.abs(x_true - x_true_shuffle))
    recon_diff = np.sum(np.abs(x_recon - x_recon_shuffle))
    return recon_diff / true_diff


def intensity_hist(x_true, x_recon):
    """Compute list of ratios of intensity reconstruction

    Args:
        x_true (array): True images
        x_recon (array): reconstructed image

    Returns:
        list of ratios of intensity reconstruction sorted by the intesity of the true image
    """
    x_true = x_true.flatten()
    x_recon = x_recon.flatten()

    idx = np.argsort(x_true)

    relatives = x_recon[idx] / x_true[idx]
    return relatives


def eval_recon(x_test, x_recon, lognorm=False):
    """Evaluate the reconstruction capabilities of an autoencoder

    Args:
        x_true (array): True images
        x_recon (array): reconstructed image
        lognorm (bool): use logarithmic norm for example images

    Returns:
        evaluation plots
    """
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
    x_recon = x_recon.reshape(x_recon.shape[0], x_recon.shape[1], x_recon.shape[2],1)

    collapse_metric = test_collapse(x_test, x_recon)
    emd = avg_emd(x_test, x_recon)
    ssim = tf.reduce_mean(tf.image.ssim(x_test.astype('float64'), x_recon.astype('float64'), max_val=1.0)).numpy()
    MAE = tf.reduce_mean(tf.abs(x_test - x_recon)).numpy()
    normalized_MAE = MAE / tf.reduce_mean(tf.reduce_sum(x_test,axis=(1,2)).numpy().reshape((x_test.shape[0],1,1,1))).numpy()

    print(f'Collapse_metric: {collapse_metric:.3}')
    print(f'Average EMD: {emd:.3}')
    print(f'ssim: {ssim:.3}')
    print(f'MAE: {MAE:.3}')
    print(f'normalized MAE: {normalized_MAE:.3}')

    fig, axs = plt.subplots(2,3, figsize=(10, 5))

    if lognorm:
        norm = LogNorm()
    else:
        norm = None

    for i in range(3):
        rint = np.random.randint(len(x_test))
        axs[0,i].imshow(x_test[rint], cmap='binary', norm=norm)
        axs[0,i].title.set_text('true')

        fig.colorbar(cm.ScalarMappable(norm=norm, cmap='binary'), ax=axs[0,i])

        axs[1,i].imshow(x_recon[rint], cmap='binary', norm=norm)
        axs[1,i].title.set_text('reconstructed')

        fig.colorbar(cm.ScalarMappable(norm=norm, cmap='binary'), ax=axs[1,i])

    fig.tight_layout()


def eval_tagging(x_true_background, x_recon_background, x_true_signal, x_recon_signal):
    """Evaluate the anomaly tagging capabilities of an autoencoder

    Args:
        x_true_background (array): True images of background events
        x_recon_background (array): reconstructed images of background events
        x_true_signal (array): True images of signal events
        x_recon_signal (array): reconstructed images of signal events

    Returns:
        Plots for tagging evaluation
    """
    x_true_background = x_true_background.reshape(x_true_background.shape[0], x_true_background.shape[1], x_true_background.shape[2],1)
    x_recon_background = x_recon_background.reshape(x_recon_background.shape[0], x_recon_background.shape[1], x_recon_background.shape[2],1)
    x_true_signal = x_true_signal.reshape(x_true_signal.shape[0], x_true_signal.shape[1], x_true_signal.shape[2],1)
    x_recon_signal = x_recon_signal.reshape(x_recon_signal.shape[0], x_recon_signal.shape[1], x_recon_signal.shape[2],1)

    bce_background = tf.keras.losses.binary_crossentropy(x_true_background, x_recon_background, axis=(1,2,3)).numpy()
    bce_signal = tf.keras.losses.binary_crossentropy(x_true_signal, x_recon_signal, axis=(1,2,3)).numpy()

    fig, axs = plt.subplots(1,3, figsize=(11, 4))

    print(f'Median background: {np.median(bce_background):.3}')
    print(f'Median signal: {np.median(bce_signal):.3}')
    bins = np.histogram(np.hstack((bce_background, bce_signal)), bins=25)[1]
    axs[0].hist(bce_background, histtype='step', label="background",bins=bins)
    axs[0].hist(bce_signal, histtype='step', label="signal",bins=bins)
    axs[0].set_xlabel("loss")
    axs[0].legend()

    thresholds = np.linspace(0,max(np.max(bce_background),np.max(bce_signal)),1000)

    accs = []
    for i in thresholds:
        num_background_right = np.sum(bce_background < i)
        num_signal_right = np.sum(bce_signal > i)
        acc = (num_background_right + num_signal_right)/(len(x_recon_background) + len(x_recon_signal))
        accs.append(acc)

    print(f'Maximum accuracy: {np.max(accs):.3}')
    axs[1].plot(thresholds, accs)
    axs[1].set_xlabel("anomaly threshold")
    axs[1].set_ylabel("tagging accuracy")

    y_true = np.append(np.zeros(len(bce_background)), np.ones(len(bce_signal)))
    y_pred = np.append(bce_background, bce_signal)
    auc = roc_auc_score(y_true, y_pred)
    print(f'AUC: {auc:.4}')
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    tnr = 1 - fpr
    x = np.linspace(0,1,50)
    y_rnd = 1 - x
    axs[2].plot(tnr,tpr, label="anomaly tagging")
    axs[2].plot(x,y_rnd, label="random tagging", color='grey')
    axs[2].set_xlabel("fpr")
    axs[2].set_ylabel("tpr")
    axs[2].legend()

    fig.tight_layout()


def iforest_latent_eval(background_latent, signal_latent):
    """Evaluate the anomaly tagging capabilities by using an isolation forest on the latent space representations

    Args:
        background_latent (array): latent representation of bg events
        signal_latent (array): latent represnation of signal events

    Returns:
        Plots for tagging evaluation
    """
    clf = IsolationForest(random_state=0).fit(background_latent)

    if_pred_bg = clf.decision_function(background_latent)
    if_pred_signal = clf.decision_function(signal_latent)

    fig, axs = plt.subplots(1,3, figsize=(11, 4))

    print(f'Median background: {np.median(if_pred_bg):.3}')
    print(f'Median signal: {np.median(if_pred_signal):.3}')
    bins = np.histogram(np.hstack((if_pred_bg, if_pred_signal)), bins=25)[1]
    axs[0].hist(if_pred_bg, histtype='step', label="background",bins=bins)
    axs[0].hist(if_pred_signal, histtype='step', label="signal",bins=bins)
    axs[0].set_xlabel("loss")
    axs[0].legend()

    thresholds = np.linspace(-0.4,max(np.max(if_pred_bg),np.max(if_pred_signal)),1000)

    accs = []
    for i in thresholds:
        num_background_right = np.sum(if_pred_bg > i)
        num_signal_right = np.sum(if_pred_signal < i)
        acc = (num_background_right + num_signal_right)/(len(if_pred_bg) + len(if_pred_signal))
        accs.append(acc)

    print(f'Maximum accuracy: {np.max(accs):.3}')
    axs[1].plot(thresholds, accs)
    axs[1].set_xlabel("anomaly threshold")
    axs[1].set_ylabel("tagging accuracy")

    y_true = np.append(np.ones(len(if_pred_bg)), np.zeros(len(if_pred_signal)))
    y_pred = np.append(if_pred_bg, if_pred_signal)
    auc = roc_auc_score(y_true, y_pred)
    print(f'AUC: {auc:.3}')
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    tnr = 1 - fpr
    x = np.linspace(0,1,50)
    y_rnd = 1 - x
    axs[2].plot(tnr,tpr, label="anomaly tagging")
    axs[2].plot(x,y_rnd, label="random tagging", color='grey')
    axs[2].set_xlabel("fpr")
    axs[2].set_ylabel("tpr")
    axs[2].legend()

    fig.tight_layout()


def img_to_event(img):
    """Convert event image to event format for energyflow (list of non zero pixels with position and intensity as features)

    Args:
        img (array): single jet image

    Returns:
        jet event as 2d np array
    """
    x_dim, y_dim = img.shape
    y_pos = np.indices((x_dim, y_dim))[0]
    x_pos = np.indices((x_dim, y_dim))[1]
    stacked = np.dstack((img, x_pos, y_pos))
    stacked = stacked.reshape((x_dim*y_dim, 3))
    stacked = stacked[stacked[:,0]!=0]
    return stacked


def img_emd(img1, img2, R=0.4):
    """Compute EMD for two images

    Args:
        img1 (array): first event image
        img2 (array): second event image
        R (float): Radius for emd computation

    Returns:
        EMD
    """
    return ef.emd.emd(img_to_event(img1.reshape((img1.shape[0],img1.shape[1]))), img_to_event(img2.reshape((img2.shape[0],img2.shape[1]))), R=R)


def avg_emd(x_true, x_recon, R=0.4):
    """Compute Average EMD for batch of images

    Args:
        x_true (array): batch of true event images
        x_recon (array): batch of reconstructed event images
        R (float): Radius for emd computation

    Returns:
        avg EMD
    """
    return np.mean([img_emd(x,y) for x,y in zip(x_true, x_recon)])


def imgs_to_events(imgs):
    """Convert batch of event images to event format for energyflow (list of non zero pixels with position and intensity as features)

    Args:
        img (array): batch of jet images

    Returns:
        batch of jet events as 3d np array
    """
    x_dim, y_dim = imgs.shape[1], imgs.shape[2]
    y_pos = np.indices((x_dim, y_dim))[0]
    x_pos = np.indices((x_dim, y_dim))[1]
    y_posN = np.repeat(y_pos.reshape((1,x_dim,y_dim)), imgs.shape[0], axis=0)
    x_posN = np.repeat(x_pos.reshape((1,x_dim,y_dim)), imgs.shape[0], axis=0)
    stacked = np.stack((imgs, x_pos, y_pos), axis=3)
    stacked = stacked.reshape((imgs.shape[0], x_dim*y_dim, 3))
    Gs = []
    for event in stacked:
        Gs.append(event[event[:,0]!=0])
    return Gs
