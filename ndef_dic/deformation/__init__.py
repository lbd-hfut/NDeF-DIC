"""Surface-based neural deformation field for multi-camera DIC."""

from .deformation_dataset import DeformationDatasetConfig, SurfaceDeformationDataset
from .deformation_field import DeformationFieldConfig, NeuralDisplacementField, PositionalEncoding
from .deformation_loss import deformation_photometric_mse, smoothness_loss
from .train_deformation import DeformationTrainingConfig, run_deformation_training

__all__ = [
    "DeformationDatasetConfig",
    "SurfaceDeformationDataset",
    "DeformationFieldConfig",
    "NeuralDisplacementField",
    "PositionalEncoding",
    "deformation_photometric_mse",
    "smoothness_loss",
    "DeformationTrainingConfig",
    "run_deformation_training",
]
