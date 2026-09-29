# Quantum Acoustic QSVM & Multi-Class QCNN Pipeline

This repository contains a hybrid Quantum Machine Learning (QML) pipeline for acoustic signal processing, statistical feature filtering, and classification using PennyLane and Scikit-Learn.

## 🔑 Key Features
* **DSP & Spectral Extraction**: Fast Mel-Spectrogram extraction and feature aggregation using `librosa`.
* **Statistical Feature Selection**: Vectorized $t$-test and ANOVA $F$-test filtering ($p < 0.01$) to compress frequency feature space.
* **PennyLane QSVM Engine**: Vectorized state-vector embeddings (`lightning.qubit`) using an entangling feature map ansatz to precompute quantum Gram matrices for SVM classification.
* **Hierarchical QCNN Architecture**: 8-qubit Quantum Convolutional and Pooling Neural Network built with custom variational ansatzes for 2D time-frequency patches and multi-class expectation decoding.

---

## 📁 Repository Structure

```text
quantum_acoustic_qsvm/
├── README.md               # Project overview & usage instructions
├── requirements.txt        # Dependencies
├── main_pipeline.py        # Quantum Kernel + SVM pipeline script
├── train_qcnn.py           # Multi-Class QCNN training script
└── demo.ipynb              # Walkthrough demo notebook