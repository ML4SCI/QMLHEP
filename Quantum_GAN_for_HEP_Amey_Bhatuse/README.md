# Quantum Generative Adversarial Neural Networks for High Energy Physics Analysis at the LHC

<p>
<!-- [<img src="https://raw.githubusercontent.com/Amey-2002/GSoC_2022_QMLHEP/main/assets/gsoc%40ml4sci.jpeg" title="Electron" />](https://ml4sci.org/) -->
<a href="https://ml4sci.org/" target="_blank"><img alt="gsoc@ml4sci" height="350px" width="1000" src="https://raw.githubusercontent.com/Amey-2002/GSoC_2022_QMLHEP/main/assets/gsoc%40ml4sci.jpeg" /></a>
</p>

This project is an official submission to the [Google Summer of Code 2022](https://summerofcode.withgoogle.com/) program carried out under the supervision of mentors from the ML4SCI organization.<br>
The official project webpage can be found [here](https://summerofcode.withgoogle.com/programs/2022/projects/jp4vG7tW).
***

## Content
- [Setup](#Setup)
- [Project](#Project)
  - [Datasets](#Datasets)
    - [Electron Photon](#Photon-Electron-Electromagnetic-Calorimeter-(ECAL)-Dataset)
    - [Jet Mass](#Jet-Mass-Image-Dataset)
  - [Models](#Models)
    - [Hybrid Quantum GAN](#Hybrid-Quantum-GAN)
    - [Fully Quantum GAN](#Fully-Quantum-GAN)
    - [Entangled Quantum GAN](#Entangled-Quantum-GAN)
- [Results](#Results)

## Setup

It is preferable to set up a virtual environment so that you do not face any package version conflicts.
```shell
python -m venv env_name
cd env_name
.\Scripts\activate
```

Clone the repository and navigate to the required folder.
```shell
git clone https://github.com/Amey-2002/GSoC_2022_QMLHEP
```
Naviagate to the folder
```shell
cd GSoC_2022_QMLHEP
```
Install the necessary libraries and frameworks.
```shell
pip install -r requirements.txt
```
Install the QGANSHEP package
```shell
setup.py install
```

_Note: If the code does not run or gives errors, try using google colab and using the following versions for tf and tfq.
<br>
tensorflow==2.7.0
<br>
tensorflow-quantum==0.6.0 --use-deprecated=legacy-resolver_
***

## Project
The proposed project introduces an novel approach for Quantum Generative Adversarial Networks, a quantum machine learning technique to solve high energy physics problems like regeneration and classification of large data. This project aims to demonstrate quantum machine learning's potential in solving critical high energy physics problems that seem to be intractable using classical techniques.

## Datasets
### Photon-Electron Electromagnetic Calorimeter (ECAL) Dataset
<p align="middle">
  <img src="https://raw.githubusercontent.com/Amey-2002/GSoC_2022_QMLHEP/main/assets/photon%20full.png" title="Photon" />
  <img src="https://raw.githubusercontent.com/Amey-2002/GSoC_2022_QMLHEP/main/assets/electron%20full.png" title="Electron" /> <br>
  Averages of Photon (left) and Electron (right) image samples from the dataset.
</p>

### Jet Mass Image Dataset

<p align="middle">
  <img src="https://raw.githubusercontent.com/Amey-2002/GSoC_2022_QMLHEP/main/assets/jet_mass.png" title="Jet Image" /> 
   <br>
   A typical Jet Image <br>
</p> 
 
<p align="middle">
  <img src="https://raw.githubusercontent.com/Amey-2002/GSoC_2022_QMLHEP/main/assets/normalized_jet_mass.png" title="Normalized Jet Image" /> 
  <br>
  A normalized Jet Image<br>
</p>

## Models
### Hybrid Quantum GAN
<p>
<img src="https://raw.githubusercontent.com/Amey-2002/GSoC_2022_QMLHEP/main/assets/hybrid_qgans.png" title="Hybrid Quantum GANS" />
</p><br>
This model can have either a quantum generator-classical discriminator or a classical generator-quantum discriminator. The generator and the discriminator have convolutional layers either quantum or classical. The classical convolutional layers can be used from a high level API. The QGANSHEP package has its own quantum convolutional layer implemented using tensorflow quantum and cirq.
<br>
<br>

### Fully Quantum GAN
<p>
<img src="https://raw.githubusercontent.com/Amey-2002/GSoC_2022_QMLHEP/main/assets/fully_quantum_gans.png" title="Hybrid Quantum GANS" />
</p>
<br>
This model will have both generator and discriminator as quantum convolutional layers. The QGANSHEP package has its own quantum convolutional layer implemented using tensorflow quantum and cirq.
<br>
<br>

### Entangled Quantum GAN
<p>
<img src="https://raw.githubusercontent.com/Amey-2002/GSoC_2022_QMLHEP/main/assets/entangled_qgans.png" title="Hybrid Quantum GANS" />
</p>
<br>

This is a new approach inspired by this [paper](https://arxiv.org/abs/2105.00080) in which a fidelity test is used. A [Swap test](https://en.wikipedia.org/wiki/Swap_test)(fidelity test) is a quantum computation technique through which we can measure the closeness of two states, i.e., how much two quantum states are similar to each other. The other uniqueness of this approach is that we upload the random data and the real data both at the same time in one train step instead of one following the other. As intended, the network is presumed to encode the required information in a latent space following a adversarial strategy. Then, this latent space is applied an Inverse PCA transform which is first fit on real data and it is thought that this will output images similar to real images. Various manipulations were tried, however the training for this network is unstable and it is difficult to interpret the overall outcome, hence it will require more study to check the validity of the idea.

<br>
<br>

_Note: There is also an implementation of the fully quantum model using pennylane library, however, due to some technical issue it could not be completed(Otherwise, implementation of a QGAN in pennylane and tensorflow was also in plans to be integrated into the package). Please do check it once if the issue can be resolved and if you can play around with it._

## Tutorials/Docs
The [demo notebooks](https://github.com/Amey-2002/GSoC_2022_QMLHEP/tree/main/demo%20notebooks) serve as tutorials as well as documentation. It is also possible to dive right into the actual code as all the necessary docstrings also help to understand the code.

## Results

The training for generative adversarial networks is a difficult task to accomplish even for the classical paradigm, the addition of quantum computation makes it more complicated to analyse and interpret the results. Also, because these networks are actually quantum circuits getting simulated in the backend, it takes a lot of time and computational overhead when deep quantum circuits are employed as well as when the network itself is large enough in size as the number of simulations increase. In this project, due to limited computational resources, networks could be trained using only upto a certain number of parameters and depth of quantum circuits as well as for limited number of samples and small number of epochs. It is a request to anyone who inspects this project must definitely try the models with high values of parameters and samples if enough computational resources are available.
<br>
 A few results for each model can be found in the respective folders with folder name as the model name.