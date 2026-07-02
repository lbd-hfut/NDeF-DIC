"""Dense initialisation modules for NDeF-DIC.

The current dense boundary is intentionally narrow: it converts SfM sparse
observations into a camera-conditioned neural depth initialisation.  Legacy
dense experiments have been removed to avoid mixing incompatible assumptions.
"""

from .model_init import DepthInitConfig, SfMDepthFiLMNet, run_model_init
from .dense_znssd import DenseZNSSDConfig, DenseZNSSDLoss, run_dense_znssd
from .reconstruction_dataset import (
    BalancedPerCameraBatchLoader,
    ReconstructionDatasetConfig,
    ReconstructionMemmapDataset,
    run_reconstruction_dataset,
)
from .reconstruction_dense import DenseReconstructionConfig, run_dense_reconstruction
from .roi_builder import CameraMask, ROIConfig, load_mask_meta, load_masks, run_auto_roi
from .surface_sampler import SurfaceSamplerConfig, run_surface_sampler

__all__ = [
    "DepthInitConfig",
    "SfMDepthFiLMNet",
    "run_model_init",
    "DenseZNSSDConfig",
    "DenseZNSSDLoss",
    "run_dense_znssd",
    "ReconstructionDatasetConfig",
    "ReconstructionMemmapDataset",
    "BalancedPerCameraBatchLoader",
    "run_reconstruction_dataset",
    "DenseReconstructionConfig",
    "run_dense_reconstruction",
    "SurfaceSamplerConfig",
    "run_surface_sampler",
    "ROIConfig",
    "CameraMask",
    "run_auto_roi",
    "load_masks",
    "load_mask_meta",
]
