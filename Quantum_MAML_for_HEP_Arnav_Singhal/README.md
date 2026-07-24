# QMAML: Quantum Model-Agnostic Meta-Learning
QMAML combines classical meta-learning (MAML) with parameterized quantum circuits (PQC) to enable rapid adaptation on new tasks with few examples. This project implements and benchmarks QMAML on three datasets: MNIST, Higgs Boson, and Quark–Gluon jets. We analyze the effects of different quantum parameter initializations and demonstrate that QMAML achieves strong performance in complex scientific classification problems.

Validated on:
- **Image classification:** [MNIST](http://yann.lecun.com/exdb/mnist/) (few-shot digit recognition)  
- **HEP data:** Higgs Boson and [Quark–Gluon jet dataset](https://arxiv.org/abs/1902.08276)  

![GSoC @ ML4Sci](images/GSOC%20ML4Sci.jpg)

*This project was developed as part of Google Summer of Code (GSoC) 2025 under the ML4Sci organization.*

**Project Title:** Quantum Model-Agnostic Meta-Learning for Variational Quantum Algorithms for High Energy Physics Tasks at LHC  
**Author:** Arnav Singhal  
**Mentors:** KC Kong, Junyong Lee, Jeihee Cho  

---

## Task Generation (Overview)
To ensure tasks are meaningful and balanced, we designed a **few-shot task generation pipeline**:
- **Preprocessing:** Images normalized with train statistics.  
- **Binning:** Dataset split by physics variables (*pt, m0*) into quantile ranges.  
- **Sampling:** Support and query sets drawn from matched bins to avoid distribution mismatch.  
- **Guards:** Multiple checks (distribution match, triviality checks, PCA sanity) ensure each task is non-trivial and representative.  

This pipeline yields high-quality few-shot tasks for robust training and evaluation.

---

## Training Architecture (Overview)
Our training setup combines **classical CNN embeddings** with **quantum circuits**:
- **Backbone:** Small-stem ResNet-18 (with frozen batch norm) extracts stable 512-D features.  
- **Quantum Layer:** Parameterized quantum circuit (RY + StronglyEntanglingLayers) processes embeddings into expressive quantum states.  
- **Meta-Learning:**  
  - **Inner Loop:** Rapid adaptation using Reptile-style updates or Q-MAML task embeddings.  
  - **Outer Loop:** Updates both CNN and PQC components to generalize across tasks.  

This hybrid design leverages classical stability with quantum adaptability.

---

## Dataset Links
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)  
- [Quark–Gluon Jet Dataset (HEPML)](https://arxiv.org/abs/1902.08276)  
- Higgs Boson Dataset (available on UCI/Kaggle)

---

## Structure

The project is organized as follows:

### Main Directory:

• **LICENSE**: Contains the licensing information for the project.

• **README**: Provides an overview of the project, its objectives, and structure.

• **config.py**: Contains configuration settings and hyperparameters for training quantum meta-learning models.

• **data.py**: Handles dataset loading, preprocessing, and few-shot task generation pipeline for MNIST, Higgs Boson, and Quark-Gluon jet datasets.

• **train.py**: Implements the main training loops for both QMAML and Reptile-style meta-learning approaches.

• **utils.py**: Contains utility functions for visualization, plotting, and performance analysis.

• **models/**: A folder dedicated to model implementations. 

  ○ **hybrid.py**: Implements the hybrid classical-quantum architecture combining CNN feature extractors with PQC.
  
  ○ **pqc.py**: Contains the implementation of parameterized quantum circuits (PQC)

This structure allows for the exploration of quantum meta-learning architectures across different scientific datasets (MNIST, Higgs Boson, Quark-Gluon jets). The project combines classical CNN embeddings with quantum circuits to enable rapid few-shot adaptation on new tasks.

---

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.