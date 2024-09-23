# Equivariant Quantum Neural Networks for High Energy Physics Analysis at the LHC

<p align="center">
  <img src="https://github.com/LazaroR-u/EQNN/assets/80428982/63d3cc7b-f42f-4989-b203-4cce5eaff822" alt="image">
</p>


## Organization
[Machine Learning For Science (ML4SCI)](https://ml4sci.org/)

## Contributor
[Lazaro Diaz](https://www.linkedin.com/in/lazaro-raul-diaz-lievano/)

## Mentors
- [KC Kong](https://physics.ku.edu/people/kong-kyoungchul)
- [Konstantin Matchev](https://www.phys.ufl.edu/~matchev/)
- [Katia Matcheva](https://www.phys.ufl.edu/wp/index.php/people/faculty/katia-matcheva/)
- [Sergei Gleyzer](http://sergeigleyzer.com/)


## Project blog post 
[Equivariant Quantum Neural Networks for High Energy Physics Analysis at the LHC. Mid-Evaluation Report.](https://medium.com/@214lievano/equivariant-quantum-neural-networks-for-high-energy-physics-analysis-at-the-lhc-59b55ed3d43e)

## Summary

We propose an Equivariant Quantum Neural Network (EQNN) and an Equivariant Hybrid Quantum-Classical Neural Network architecture that leverages symmetries commonly present in image data, specifically roto-reflection symmetries. By incorporating symmetries such as rotations and reflections into the quantum neural network's design, we can significantly reduce the number of trainable parameters, thereby decreasing the model's complexity and improving its efficiency. This method enhances learning capabilities with smaller datasets while also promoting better generalization. We evaluate the performance of our model using standard benchmark datasets for image classification and compare it against other quantum models.

## Introduction

In this project, we introduce Equivariant Quantum Neural Networks (EQNNs) and equivariant hybrid quantum-classical architectures designed to exploit symmetries in image data, such as rotations and reflections. By embedding these geometric transformations directly into the neural network architecture, we aim to reduce the number of trainable parameters and increase computational efficiency. This approach is particularly suited for tasks like image classification, where symmetries play a crucial role in recognizing patterns. Equivariant models make it possible to achieve better performance with smaller datasets by reducing the model's complexity and improving generalization.

One of the key advantages of using equivariant models is their ability to streamline the learning process by minimizing the number of parameters, which leads to faster training and a reduced risk of overfitting. These models also benefit from weight-sharing, allowing the same set of weights to be applied across different symmetric transformations, further boosting computational efficiency. By being invariant to specific symmetries, the model can effectively learn from limited data without relying on additional data augmentation techniques.

However, it is important to ensure that the data reflects the symmetries embedded in the model. If the data does not possess these symmetries, enforcing equivariance may limit the model’s expressivity, potentially leading to suboptimal results. Therefore, careful alignment between the symmetries in the model and the properties of the data is essential for optimal performance.


## Datasets

- **MNIST**: A large database of handwritten digits commonly used for training and testing various image processing systems. It consists of 60,000 training images and 10,000 testing images, each of size 28x28 pixels in grayscale.

- **Fashion MNIST**: Similar to the MNIST dataset but contains images of clothing items such as shirts, trousers, and shoes. It also consists of 60,000 training images and 10,000 testing images, each of size 28x28 pixels in grayscale.

- **Quark-Gluon**: We use simulated 2012 CMS Open Data. This dataset is used in high-energy physics to distinguish between quark-initiated and gluon-initiated particle jets. The images have three channels corresponding to the Tracker, the Electromagnetic Calorimeter (ECAL), and the Hadronic Calorimeter (HCAL).

- **Electron-Photon**: Consists of the 2012 CMS Simulated Open Data for the decay of the Standard Model (SM) Higgs boson to two photons versus its leading backgrounds. This dataset is used to classify electron-initiated and photon-initiated events.



## Structure

### Equivariant_QCNN/
It Contains the modules to develop Equivariant Quantum Convolutional Neural Networks (EQCNN) and generic Quantum Convolutional Neural Networks (QCNN) with different quantum convolutional filters.

Modules:
- **data/data.py**: defines a function to load and process datasets corresponding to quatum models.
- **data/hybrid_data.py** defines a function to load and process datasets corresponding to hybrid models.
- **hep_processing/hep_dataset_processing.py**: loads and preprocesses the HEP datasets.
- **hep_processing/electron_photon_processing.ipynb**: loads and preprocesses the electron_photon dataset.
- **models/utils/embedding.py**: matches the embedding type to encode classical data into quantum states.
- **models/utils/unitary.py**: contains the set of convolutional filters and pooling layers.
- **models/QCNN_circuit.py**: constructs the EQCNN and QCNN architectures.
- **models/hybird_models.py**: constructs the hybrid models: both equivariant and no-equivariant.
- **training/Training.py**: defines cost functions (MSE and BCE) and trains the model for a given architecture, dataset, and hyperparameters.
- **benchmarking/Benchmarking.py**: defines the accuracy and a function to benchmark different given models.
- **benchmarking/result.py**: executes the benchmarking function for a given set of models, types of encodings, a dataset, and a cost function. Returns the loss history and accuracy for each model and setting.
- **benchmarking/CNN.py**: performs benchmarking for different settings for classical Convolutional Neural Networks.

## demos_notebooks/
- **Classical/classical_euivariant_CNN_Fashion_MNIST.ipynb**
- **Classical/classical_equivariant_CNN_MNIST.ipynb**
- **Classical/electron_photon_CNN.ipynb**
- **Classical/Quark_gluon_CNN.ipynb**
- **Hybrid/Equiv_Hybrid_EQCNN_Fashion_MNIST.ipynb**
- **Hybrid/Equiv_Hybrid_EQCNN_MNIST.ipynb**
- **Hybrid/Hybrid_QNN_Electron_photon.ipynb**
- **Hybrid/Hybrid_QNN_Fashion_MNIST.ipynb**
- **Hybrid/Hybrid_QNN_MNIST.ipynb**
- **Quantum/Electron-Photon_EQCNN.ipynb**
- **Quantum/Fashion_MNIST_EQCNN.ipynb**
- **Quantum/MNIST_EQCNN.ipynb**
- **Quantum/p4m_EQNN_mnist.ipynb**
- **Quantum/Quark_Gluon_EQCNN.ipynb**
- **Quantum/REFLECTION_EQNN_mnist.ipynb**


## test/
- **Electron-Photon_EQCNN.ipynb** 
- **Fashion_MNIST_EQCNN.ipynb**
- **hybrid_models_Fashion_MNIST.ipynb**
- **hybrid_models_MNIST.ipynb**
- **MNIST_EQCNN.ipynb**
- **Quark_Gluon_EQCNN.ipynb**
  
Additional:
- **requirements.txt**: Lists all the necessary packages to run the code.
- **GSOC_2024_EQNN_midterm.pdf**: Contains a presentation of the project's progress up to the midterm evaluation of the Google Summer of Code 2024 program.


## Installation

````
git clone https://github.com/LazaroR-u/EQNN_for_HEP.git
cd Equivariant_QCNN
python3 -m venv env
source env/bin/activate
pip install -r ../requirements.txt
````



## References

1. Andrews, M., Alison, J., An, S., Bryant, P., Burkle, B., Gleyzer, S., Narain, M., Paulini, M., Poczos, B., & Usai, E. (2019). End-to-End Jet Classification of Quarks and Gluons with the CMS Open Data. Nucl. Instrum. Methods Phys. Res. A 977, 164304 (2020). [https://doi.org/10.1016/j.nima.2020.164304](https://doi.org/10.1016/j.nima.2020.164304)

2. Andrews, M., Paulini, M., Gleyzer, S., & Poczos, B. (2018). End-to-End Event Classification of High-Energy Physics Data. Journal of Physics: Conference Series, Volume 1085, Issue 4. [https://dx.doi.org/10.1088/1742-6596/1085/4/042022](https://dx.doi.org/10.1088/1742-6596/1085/4/042022)

3. Chang, Su Yeon, Michele Grossi, Bertrand Le Saux, y Sofia Vallecorsa. Approximately Equivariant Quantum Neural Network for $p4m$ Group Symmetries in Images. En 2023 IEEE International Conference on Quantum Computing and Engineering (QCE), 229-35, 2023. [https://doi.org/10.1109/QCE57702.2023.00033](https://doi.org/10.1109/QCE57702.2023.00033).

4. Hur, Tak, Leeseok Kim, y Daniel K. Park. Quantum convolutional neural network for classical data classification. Quantum Machine Intelligence 4, n.o 1 (junio de 2022): 3. [https://doi.org/10.1007/s42484-021-00061-x](https://doi.org/10.1007/s42484-021-00061-x).

5. West, Maxwell T., Martin Sevior, y Muhammad Usman. Reflection Equivariant Quantum Neural Networks for Enhanced Image Classification. Machine Learning: Science and Technology 4, n.o 3 (1 de septiembre de 2023): 035027. [https://doi.org/10.1088/2632-2153/acf096](https://doi.org/10.1088/2632-2153/acf096).

6.  Nguyen, Quynh T., Louis Schatzki, Paolo Braccia, Michael Ragone, Patrick J. Coles, Frederic Sauvage, Martin Larocca, y M. Cerezo. Theory for Equivariant Quantum Neural Networks. PRX Quantum 5, n.o 2 (6 de mayo de 2024): 020328. [https://doi.org/10.1103/PRXQuantum.5.020328](https://doi.org/10.1103/PRXQuantum.5.020328).



