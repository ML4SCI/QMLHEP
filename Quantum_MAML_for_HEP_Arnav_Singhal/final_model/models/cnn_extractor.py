# models/cnn_extractor.py
import torch
import torch.nn as nn
from config import config

class CNNFeatureExtractor(nn.Module):
    """
    ResNet18 with small-stem and split head:
    - embed(x) -> 512-D feature (stable for tasks)
    - forward(x) -> num_qubits angles (fed to PQC)
    """
    def __init__(self, output_dim: int, num_qubits: int):
        super().__init__()
        # Load pretrained ResNet18
        m = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        # Small stem for 125x125
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        self.backbone = m

        # Shared projector to a stable embedding
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, output_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Map embedding -> angles for PQC
        self.to_angles = nn.Linear(output_dim, num_qubits)

        # ImageNet normalization stats
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("_std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization (auto-scale if 0-255 input)."""
        if x.dim() == 3: 
            x = x.unsqueeze(0)
        x = x.float()
        m, s = float(x.mean()), float(x.std())
        if x.max() > 1.5 and not (abs(m) < 0.1 and 0.6 < s < 1.5):
            x = x / 255.0
        return (x - self._mean.to(x)) / self._std.to(x)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return 512-D embedding (no angle mapping)."""
        x = self._norm(x)
        self.backbone.eval()
        return self.backbone(x)  # (B, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._norm(x)
        feat = self.backbone(x)       # (B, output_dim)
        angles = self.to_angles(feat) # (B, num_qubits)
        return angles

# Instantiate default CNN (using config values)
cnn_extractor = CNNFeatureExtractor(config.CNN_OUTPUT_DIM, config.NUM_QUBITS)

# Freezing helpers

def freeze_bn(m: nn.Module) -> None:
    """Set all BatchNorm layers to eval mode and freeze their parameters."""
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad = False

def set_cnn_trainable(cnn: CNNFeatureExtractor,
                      train_layer4: bool = True,
                      train_to_angles: bool = True) -> None:
    """
    Freeze everything in the ResNet backbone, then explicitly unfreeze:
      • layer4 (convs only; keep BN frozen)
      • to_angles head
    BN layers stay in eval mode with requires_grad=False.
    """
    # Freeze everything
    for p in cnn.backbone.parameters():
        p.requires_grad = False

    # Unfreeze layer4 CONVs (keep BN frozen)
    if train_layer4:
        for name, p in cnn.backbone.named_parameters():
            if "layer4" in name and "bn" not in name:
                p.requires_grad = True

    # Unfreeze to_angles head
    for p in cnn.to_angles.parameters():
        p.requires_grad = bool(train_to_angles)

    # Keep BN frozen
    freeze_bn(cnn)

def freeze_cnn_except_last_block(cnn: CNNFeatureExtractor) -> None:
    """Freeze all layers except the last ResNet block (layer4)."""
    for name, param in cnn.backbone.named_parameters():
        if "layer4" not in name:
            param.requires_grad = False

__all__ = [
    "CNNFeatureExtractor",
    "cnn_extractor",
    "freeze_bn",
    "set_cnn_trainable",
    "freeze_cnn_except_last_block",
]
