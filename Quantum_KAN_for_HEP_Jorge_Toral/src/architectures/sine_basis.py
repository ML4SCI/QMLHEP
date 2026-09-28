# src/architectures/sine_basis.py
"""
SineKAN basis, univariate-edge form:

    y = sum_k  A_k * sin(freq_k * x + phase_k)

freq_k / phase_k are fixed by the grid size (ported 1:1 from the reference
SineKANLayer.__init__), so the basis functions are fixed functions of x, like
T_n(x) for Chebyshev, and the amplitudes A_k solve a plain linear least-squares
problem. Shared by the extractor (fit) and the circuit (rotation angles).
"""
import numpy as np


def forward_step(i_n, grid_size, A, K, C):
    ratio = A * grid_size ** (-K) + C
    return ratio * i_n


def build_sine_grid(grid_size: int, is_first: bool = True):
    """Returns fixed (freq, phase), each of shape (grid_size,). Input-dim-1 version
    of the reference layer (the input_phase offset linspace(0, pi, 1) is 0.0)."""
    A, K, C = 0.9724, 0.9884, 0.9994

    grid_phase = np.arange(1, grid_size + 1) / (grid_size + 1)
    phase = grid_phase + 0.0
    for i in range(1, grid_size):
        phase = forward_step(phase, i, A, K, C)

    exponent = 1 - int(is_first)
    freq = np.arange(1, grid_size + 1) / (grid_size + 1) ** exponent
    return freq, phase


def sine_design_matrix(x, freq, phase):
    """(N, grid_size) matrix with columns sin(freq_k * x + phase_k)."""
    return np.sin(np.outer(x, freq) + phase[None, :])


def sinefit(x, y, grid_size):
    """Least-squares amplitudes A_k. Returns (amplitudes, y_hat)."""
    freq, phase = build_sine_grid(grid_size, is_first=True)
    Phi = sine_design_matrix(x, freq, phase)
    amplitudes, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return amplitudes, Phi @ amplitudes
