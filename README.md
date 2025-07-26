# QKAN-QMLHEP
This project presents a novel hybrid quantum-classical architecture that integrates **Quantum Singular Value Transformation (QSVT)** with **Linear Combination of Unitaries (LCU)** and a **quantum summation layer**, followed by a classical Kolmogorov-Arnold Network (KAN).

Designed to bridge expressive quantum encoding with nonlinear classical learning, the framework enables end-to-end differentiable training using Pennylane and PyTorch. The QSVT module encodes input features via Chebyshev or Fourier polynomial transformations, while the LCU and quantum summation layers introduce trainable quantum aggregation, capturing interactions across polynomial basis functions.

The resulting representations are passed through a spline-based/sine KAN to produce interpretable and accurate predictions.

Validated on:

- Structured tabular datasets: Iris, Social_Network_Ads etc.

- High-dimensional physical data: jet images from quark-gluon classification

This work demonstrates the potential of hybrid QML pipelines in quantum-enhanced representation learning for real-world tasks.


> *This project was developed as part of Google Summer of Code (GSoC) 2025 under the ML4Sci organization.*
>
> *Project Title : Quantum Kolmogorov Arnold Networks for High Energy Physics Analysis at the LHC*
>
> *Author-Ria Khatoniar , Mentors- Eric Reinhardt, Dinesh Ramakrishnan, KC Kong*
>
> 
> ![image](https://github.com/user-attachments/assets/015756d0-1489-48f6-8549-507dc35e6d47)


