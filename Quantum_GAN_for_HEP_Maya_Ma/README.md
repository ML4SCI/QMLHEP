## This is a project on Quantum (and hybrid) GAN for high energy physics. (Quark and Gluon Jet dataset)

### Description of the files:
# Midterm:
classical.ipynb -  classical GAN, serving as benchmark to be compared with other Quantum GAN models. Tested on MNIST dataset. 

hybrid.ipynb - a hybrid quantum GAN, with both the generator and discriminator being composed of a Variational Quantum Layer (VQC) embedded in classical Neural Network. Tested on MNIST dataset. 

q_gen.ipynb - a hybrid quantum GAN with only the generator being VQC embedded in classical NN. The discriminator remains fully classical. Tested on MNIST dataset. 

hybrid_quark.ipynb - testing the same hybrid GAN on the quark and gluon jet dataset.

# Final:
iqgan_minst.ipynb - recreating IQGAN on MINST dataset for single digits. It performs well for single digit, but suffers mode collapse for the entire dataset.

conditional.ipynb - based on the result of iqgan_minst.ipynb, attempted to use quantum conditional GAN to solve mode collapse with IQGAN, while leveraging the fact that it works for single digits.

fullyQuantum.ipynb - Attempts on varying different fully quantum GAN to work on MNIST dataset. End with an initial attempt in a hybrid quantum conditional GAN that performs way better. It also includes code to plot the PCA component distribution as well as the gradient for generator and discriminator during training for a clear view of the model performance.

qghybrid.ipynb - Extending the work of hybrid_quark.ipynb after midterm

WQGAN_GP.ipynb - Based on previouse experiments, I think wassertein hybrid GAN performs the best. Thus, I'm using the archetecture to test on the quark and gluon jet dataset. As wellas making improvements to it. 
