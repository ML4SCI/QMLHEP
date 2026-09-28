# src/architectures/extractor_sine.py
import numpy as np

from src.architectures.extractor import SymbolicWarmStartExtractor
from src.architectures.sine_basis import sinefit


class SineWarmStartExtractor(SymbolicWarmStartExtractor):
    """
    Same graph extraction as SymbolicWarmStartExtractor (structure, pruning,
    wires, padding), but every edge is fit with the SineKAN basis instead of
    Chebyshev: grid_size = self.max_degree sine terms, amplitudes via lstsq.

    The saved graph carries basis="sine" so QKANModel builds the sine
    re-uploading edge. Amplitudes are returned as `coefs` (length grid_size);
    the base class pads them with a trailing 0.0 to degree+1, which is the
    identity for the final RY.
    """
    basis = "sine"

    def _fit_edge(self, model, layer_index, input_index, output_index, x_vals):
        y_vals = self._evaluate_isolated_edges(model, layer_index, input_index, output_index, x_vals)
        dynamic_range = float(np.max(y_vals) - np.min(y_vals))

        ss_tot = float(np.sum((y_vals - np.mean(y_vals)) ** 2))
        ss_tot_safe = ss_tot if ss_tot > 1e-12 else 1e-12

        amplitudes, y_pred = sinefit(x_vals, y_vals, self.max_degree)
        r2 = 1.0 - float(np.sum((y_vals - y_pred) ** 2)) / ss_tot_safe

        return amplitudes.tolist(), dynamic_range, self.max_degree, r2
