Here is the current progress on the Variational Autoencoder (VAE) applied to Quark-Gluon Jet images.
For this implementation, we adapted the base architecture from the Cats vs. Dogs model, introducing minor modifications. The primary changes focus on the metrics used to evaluate the network's performance. In addition to the standard loss functions—Binary Cross-Entropy (BCE) and Kullback-Leibler Divergence (KLD)—we have included the following evaluation metrics:
- Wasserstein-1 Distance
- RDF (Radial Distribution Function)
- SSIM (Structural Similarity Index Measure)

For this new program, we tested different latent spaces:

- 8
- 16
- 32
- 64
- 128
- 256

The best results were obtained with latent spaces of 32 or 64.

Here, the results for a latent space of 64 are shown. As can be seen, the model is able to predict the position of the most active pixels, as well as a somewhat blurry distribution. However, the reconstruction process still generates images that are quite incomplete and blurry, so work on this part is still ongoing.

