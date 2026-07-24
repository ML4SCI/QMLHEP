<div align="center">
  
  # Quantum Convolutional Neural Networks<br>for High-Energy Physics Analysis at the LHC

A Google Summer of Code 2022 Project Repository.<br>
The goal of this study is to show the capabilities of QML especially QCNN for classifying the HEP image datasets.
</div>

## Table of (Main) Contents
- [Introduction](#introduction)
  - [Synopsis](#synopsis)
- [Usage](#usage)
  - [Code Description](#code-description)
  - [Installation](#installation)
  - [Documentation Tutorials and Development](#documentation-tutorials-and-development)
- [Datasets](#datasets)
- [Experiments and Results](#experiments-and-results)
- [References](#references)
- [Future work](#future-work)

## Introduction

- **Organization**
  - [Machine Learning for Science (ML4Sci)](https://ml4sci.org/)
- **Contributor**
  - [Gopal Ramesh Dahale](https://www.linkedin.com/in/gopal-ramesh-dahale-7a3087198/)
- **Mentors**
  - [Prof. Sergei V. Gleyzer](http://sergeigleyzer.com/), [Dr. Emanuele Usai](https://orcid.org/0000-0001-9323-2107), and [Raphael Koh](https://www.raphaelkoh.me/)
- **Project Details**
  - [Project](https://ml4sci.org/gsoc/2022/proposal_QMLHEP2.html)
  - [Proposal](https://github.com/Gopal-Dahale/qml-hep-lhc/blob/main/QCNN%20Proposal.pdf)
  - [GSoC Project Page](https://summerofcode.withgoogle.com/programs/2022/projects/0gbpQgKv)


### Synopsis
Determining whether an image of a jet particle corresponds to signals or background signals is one of the many challenges faced in High Energy Physics. CNNs have been effective against jet particle images as well for classification purposes. Quantum computing is promising in this regard and as the QML field is evolving, this project aims to understand and implement QCNN and gain some enhancement.

The goal of this study is to show the capabilities of QML especially QCNN for classifying the HEP image datasets. QCNN can be completely quantum or can be a hybrid with classical. The aim is to implement both. We will use quantum variational classification instead of the final FC classical layers in the quantum setting. This will give more depth about the quantum power that can be used in the near term future.


## Usage

### Code Description
The repository contains [TensorFlow Quantum](https://www.tensorflow.org/quantum) implementation of quantum convolution and classifier with various data encoding schemes and ansatzes including data reuploading. Models in [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) and [Pennylane](https://pennylane.ai/) are also added as they have significant speed up during the training. Hybrid as well as fully quantum models can be created using the layers implemented. JAX models can be trained on TPUs as well.

### Installation

Tested on Ubuntu 22.04.1 LTS
```
git clone https://github.com/Gopal-Dahale/qml-hep-lhc.git
cd qml-hep-lhc
python -m venv qenv
source qenv/bin/activate
export PYTHONPATH=.
pip install -r requirements.txt
```

### Documentation Tutorials and Developement
- Documentation: Work in progress.
- Tutorials: [Link](https://github.com/Gopal-Dahale/qml-hep-lhc/tree/main/notebooks/Tutorials). Work in progress.
- Development notebooks: [Link](https://github.com/Gopal-Dahale/qml-hep-lhc/tree/main/notebooks/Dev). These notebooks were used during the period of GSoC.

## Datasets

### MNIST

<p align="middle">
  <img height="300 px" src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" title="MNIST Dataset" /> <br>
  <a>MNIST Dataset</a>
</p>

Single channel images of handwritten digits of size 28 x 28 pixels.

- Train size: 60000
- Test size: 10000

### Electron Photon Electromagnetic Calorimeter (ECAL)

<p align="middle">
  <img src="https://github.com/Gopal-Dahale/qml-hep-lhc/blob/main/assets/ep-mean.png" title="Electron Photon" />
  <p>Mean image of Electron and Photon. Mean of energy channel (left). Mean of time channel (right).</p>
</p>
  
The dataset contains images electrons and photons captured by the ECAL detector.
- Each pixel corresponds to a detector cell.
- The intensity of the pixel corresponds to how much energy is measured in that cell.
- 498k samples, equally distributed between the two classes.
- The size of the images are 32x32.

### Quark-Gluon

<p align="middle">
  <img src="https://github.com/Gopal-Dahale/qml-hep-lhc/blob/main/assets/gluon-mean.png" title="Gluon" /><br>
  <img src="https://github.com/Gopal-Dahale/qml-hep-lhc/blob/main/assets/quark-mean.png" title="Quark" /><br>
  <br>
  <a>Mean image of Gluon (Top) and Quark (Bottom) for all 3 channels (Tracks, ECAL and HCAL respectively) over full dataset.</a>
  <img src="https://github.com/Gopal-Dahale/qml-hep-lhc/blob/main/assets/gluon-mean-cropped.png" title="Gluon cropped" /><br>
  <img src="https://github.com/Gopal-Dahale/qml-hep-lhc/blob/main/assets/quark-mean-cropped.png" title="Quark cropped" /><br>
  <a>Mean of cropped image of Gluon (Top) and Quark (Bottom) for all 3 channels (Tracks, ECAL and HCAL respectively) over full dataset. </a>
</p>

The dataset contains images of simulated quark and gluon jets. The image has three channels, the first channel is the reconstructed tracks of the jet, the second channel is the images captured by the electromagnetic calorimeter (ECAL) detector, and the third channel is the images captured by the hadronic calorimeter (HCAL) detector.
  
- A total of 700k samples, equally distributed between the two classes.
- The images has size of 125 x 125 pixels (for every channel).
- Since the original size of 125 x 125 pixels is too large for quantum computing simulation, we cropped the images into certain size. For now, we limit the current size to 40 x 40 pixels.
- In this study, we focus on the ECAL channel.

## Experiments and Results

The results of all experiments can be obtained from [wandb](https://wandb.ai/gopald/qml-hep-lhc/overview?workspace=user-gopald)

### Electron Photon

#### Data
- Train: 90k
- Val: 10k
- Test: 20k
- Center crop: (8,8,1)
- Standardize

#### Model
- loss: CategoricalCrossentropy
- optimizer: Adam
- initialize: he_uniform
- lr: 1e-3
- scheduler: After every 100 epochs $lr = lr \times \sqrt(0.1)$
- batch_size: 128
- Measuring all qubits in Pauli Z basis.

##### Architecture
```
y = x
x = qconv(x)
x += y
x = relu(x)
x = Linear(x) # 8 classical neurons 
x = Linear(x) # 2 classical neurons
```

#### NQubitPQCSparse

| Qubits  | Layers | Trainable Params (classical) |Trainable Params (quantum) | Test AUC |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | 
| 1 | 1 | 538 |30 | 0.734 |  
| 1 | 2 | 538 |60 | 0.7284 | 
| 2 | 1 | 1050 |60 | 0.7397 | 
| 2 | 2 | 1050 |120 | 0.7464 | 
| 3 | 1 | 1562 |90 | 0.7518 | 
| 3 | 2 | 1562 |180 | 0.7359 | 
| 4 | 1 | 2074 |120 | 0.7331 | 
| 4 | 2 | 2074 |240 | 0.7406 | 

#### Classical CNN

| Trainable Params (conv) | Trainable Params (fc)| Test AUC |
| :-------------: | :-------------: |:-------------: |
| 30 |538 | 0.7197 |  
| 58 |538 | 0.7403 | 
| 58 |1050 | 0.7469 | 
| 124 |1050 | 0.7405 | 
| 87 |1562 | 0.7273 | 
| 188 |1562 | 0.7599 | 
| 116 |2074 | 0.7222 | 
| 236 |2074 | 0.7374 |

#### NQubitPQC

| Qubits  | Layers | Trainable Params (classical) |Trainable Params (quantum) | Test AUC |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | 
| 1 | 1 | 538 |18 | 0.7405 |  
| 1 | 2 | 538 |34 | 0.7265 | 
| 2 | 1 | 1050 |34 | 0.7458 | 
| 2 | 2 | 1050 |72 | 0.7443 | 
| 3 | 1 | 1562 |54 | 0.6753 | 
| 3 | 2 | 1562 |108 | 0.7342 | 
| 4 | 1 | 2074 |72 | 0.7284 | 
| 4 | 2 | 2074 |144 | 0.7230 | 

#### Classical CNN

| Trainable Params (conv) | Trainable Params (fc)| Test AUC |
| :-------------: | :-------------: |:-------------: |
| 20 |538 | 0.7263 |  
| 39 |538 | 0.7515 | 
| 40 |1050 | 0.7408 | 
| 78 |1050 | 0.7068 | 
| 50 |1562 | 0.7241 | 
| 107 |1562 | 0.7257 | 
| 70 |2074 | 0.7259 | 
| 142 |2074 | 0.7426 |

#### Data
- Train: 380k
- Val: 20k
- Test: 98k
- Center crop: (8,8,1)
- Standardize

#### Model
- loss: CategoricalCrossentropy
- optimizer: Adam
- initialize: he_uniform
- lr: 1e-3
- scheduler: After every 100 epochs $lr = lr \times \sqrt(0.1)$
- batch_size: 1024
- Measuring all qubits in Pauli Z basis.

#### NQubitPQC

| Qubits  | Layers | Train AUC|  Test AUC |
| :-------------: | :-------------: | :-------------: | :-------------: | 
| 1 | 2 | 0.77 | 0.7684 |  

### Quark Gluon

#### Data
- Train: 90k
- Val: 10k
- Test: 20k
- Center crop: (40,40,1)
- Power transform
- Standardize

#### Model
- loss: CategoricalCrossentropy
- optimizer: Adam
- initialize: he_uniform
- lr: 1e-3
- scheduler: After every 100 epochs $lr = lr \times \sqrt(0.1)$
- batch_size: 128
- Measuring all qubits in Pauli Z basis.

##### Architecture
```
y = x
x = qconv(x)
x += y
x = relu(x)
x = Linear(x) # 8 classical neurons 
x = Linear(x) # 2 classical neurons
```

#### NQubitPQCSparse

| Qubits  | Layers | Trainable Params (classical) |Trainable Params (quantum) | Test AUC |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | 
| 1 | 1 | - | - | 0.6743 |  
| 1 | 2 | - | - | 0.6515 | 
| 2 | 1 |  - | - | 0.6862 | 
| 2 | 2 |  - | - | 0.6820 | 
| 3 | 1 |  - | - | 0.6853 | 
| 3 | 2 |  - | - | 0.6853 | 
| 4 | 1 |  - | - | 0.6887 | 
| 4 | 2 |  - | - | 0.6870 | 

#### Data
- Train: 600k
- Val: 50k
- Test: 50k
- Center crop: (40,40,1)
- Standardize

#### Model
- loss: CategoricalCrossentropy
- optimizer: Adam
- initialize: he_uniform
- lr: 1e-3
- scheduler: After every 100 epochs $lr = lr \times \sqrt(0.1)$
- batch_size: 2048
- Measuring all qubits in Pauli Z basis.

#### NQubitPQC

| Qubits  | Layers | Train AUC|  Test AUC |
| :-------------: | :-------------: | :-------------: | :-------------: | 
| 1 | 2 | 0.723 | 0.699 |

## References

- A Tutorial on Quantum Convolutional Neural Networks (QCNN) ◦Seunghyeok Oh, † Jaeho Choi, and ◦ Joongheon Kim
- Cong, I., Choi, S. & Lukin, M.D. Quantum convolutional neural networks. Nat. Phys. 15 1273–1278 (2019). https://doi.org/10.1038/s41567-019-0648-8
- S.~Y.~C.~Chen, T.~C.~Wei, C.~Zhang, H.~Yu and S.~Yoo, ``Quantum convolutional neural networks for high energy physics data analysis,'' Phys. Rev. Res. \textbf{4} (2022) no.1, 013231 doi:10.1103/PhysRevResearch.4.013231
= Liu, J., Lim, K.H., Wood, K.L. et al. Hybrid quantum-classical convolutional neural networks. Sci. China Phys. Mech. Astron. 64, 290311 (2021). https://doi.org/10.1007/s11433-021-1734-3
- Tüysüz, C., Rieger, C., Novotny, K. et al. Hybrid quantum classical graph neural networks for particle track reconstruction. Quantum Mach. Intell. 3, 29 (2021). https://doi.org/10.1007/s42484-021-00055-9
- Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., & Latorre, J. (2020). Data re-uploading for a universal quantum classifier. Quantum, 4, 226. 
- Chen, Yusui & Hu, Wenhao & Li, Xiang. (2021). Feasible Architecture for Quantum Fully Convolutional Networks.
- Chen, Yixiong. (2021). QDCNN: Quantum Dilated Convolutional Neural Network.

## Future Work

QML is a new and evolving field and this work is just a beginning in the exploration of QML for HEP. Any contributions or issues in the existing work are invited.

In future, I aim to use 

- gradient free methods for training like COBYLA, SPSA etc.
- quantum contrastive learning.
- quantum self attention network.
