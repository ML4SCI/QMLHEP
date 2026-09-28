# Hybrid Quantum U-Net for Latent Diffusion

This notebook implements a hybrid quantum-classical U-Net for conditional latent diffusion of jet images.

```text
16×16 Quantum → 8×8 Quantum → 4×4 Classical
→ 2×2 Classical Bottleneck →
4×4 Classical → 8×8 Quantum → 16×16 Quantum
```

The quantum blocks replace the classical nonlinear activation at the `16×16` and `8×8` resolutions. Each block processes groups of eight channels using independent eight-qubit variational quantum circuits.



