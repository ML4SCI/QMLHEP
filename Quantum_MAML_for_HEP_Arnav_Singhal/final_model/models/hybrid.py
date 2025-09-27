import torch
import torch.nn as nn
from models.cnn_extractor import CNNFeatureExtractor
from models.pqc import PQCModel
from config import config

class HybridModel(nn.Module):
    """Hybrid Quantum-Classical Model."""
    def __init__(self,
                 cnn: nn.Module = None,
                 pqc: nn.Module = None,
                 init_type: str = "qmaml"):
        super(HybridModel, self).__init__()

        # Default to config-based CNN + PQC if none provided
        self.cnn = cnn if cnn is not None else CNNFeatureExtractor(
            output_dim=config.CNN_OUTPUT_DIM,
            num_qubits=config.NUM_QUBITS
        )
        self.pqc = pqc if pqc is not None else PQCModel(
            num_qubits=config.NUM_QUBITS,
            depth=config.Q_DEPTH,
            init_type=init_type
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)     # -> (B, num_qubits)
        logits = self.pqc(features)  # -> (B, 2)
        return logits

__all__ = ["HybridModel"]
