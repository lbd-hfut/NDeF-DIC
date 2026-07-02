"""
Displacement field training (Step 3).

Modules:
    deformation_net:      DeformationNetwork Φ(x, t): ℝ⁴ → ℝ³.
    deformation_trainer:  Training orchestrator (multi-phase curriculum).
    dataset:              MultiCamDataset — reference + deformed image loading.
    dic_losses:           ZNSSD + deformation smoothness losses.
"""

from .deformation_net import DeformationNetwork
from .deformation_trainer import DeformationFieldTrainer, DeformationTrainerConfig, PhaseConfig
from .dataset import MultiCamDataset
from .dic_losses import znssd, deformation_smoothness_loss

__all__ = [
    "DeformationNetwork",
    "DeformationFieldTrainer",
    "DeformationTrainerConfig",
    "PhaseConfig",
    "MultiCamDataset",
    "znssd",
    "deformation_smoothness_loss",
]
