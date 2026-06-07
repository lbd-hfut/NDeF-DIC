"""
Configuration for NDeF-DIC.

All hyperparameters for the four-stage training pipeline are centralized here.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import os


@dataclass
class SDFConfig:
    """Neural SDF surface network configuration."""
    # Positional encoding
    n_freqs: int = 6  # L=6 for smooth geometry
    # MLP architecture
    hidden_layers: int = 6
    hidden_dim: int = 256
    skip_layer: int = 3  # skip connection at this layer
    activation: str = "softplus"  # softplus | relu
    softplus_beta: float = 100.0
    # Geometric initialization
    geo_init: bool = True
    init_radius: float = 1.0
    # Sphere tracing
    sphere_trace_iters: int = 20
    sphere_trace_eps: float = 1e-4
    # Loss weights
    lambda_data: float = 10.0
    lambda_eikonal: float = 0.1
    lambda_off_surface: float = 0.01
    # Near-surface sampling offset (in world units, e.g., mm)
    surface_offset: float = 0.5


@dataclass
class IntensityConfig:
    """Intensity field (speckle pattern) network configuration."""
    # Positional encoding
    n_freqs: int = 10  # L=10 for high-frequency speckle
    # MLP architecture
    hidden_layers: int = 4
    hidden_dim: int = 256
    skip_layer: int = 2
    activation: str = "relu"
    # Output
    output_range: Tuple[float, float] = (0.0, 1.0)  # grayscale range


@dataclass
class DeformationConfig:
    """3D deformation field network configuration."""
    # Positional encoding (space)
    n_freqs_space: int = 8
    # Positional encoding (time/load step)
    n_freqs_time: int = 4
    # MLP architecture
    hidden_layers: int = 6
    hidden_dim: int = 256
    skip_layer: int = 3
    activation: str = "relu"
    # No output activation (displacements can be positive or negative)
    # Loss
    lambda_smooth: float = 0.01  # displacement smoothness regularization


@dataclass
class AppearanceConfig:
    """Per-camera appearance embedding configuration."""
    embedding_dim: int = 4
    # Affine correction bounds
    scale_range: Tuple[float, float] = (0.8, 1.2)
    bias_range: Tuple[float, float] = (-0.1, 0.1)
    # Regularization
    lambda_reg: float = 1e-4


@dataclass
class SpeckleMaskConfig:
    """Speckle mask generation configuration."""
    window_size: int = 15
    close_kernel_size: int = 15
    open_kernel_size: int = 7
    vote_threshold: int = 2  # need >=2 of 3 features to agree
    keep_largest_only: bool = True


@dataclass
class COLMAPConfig:
    """COLMAP integration configuration."""
    # Point filtering
    min_visible_cameras_ratio: float = 0.5  # must be seen by >=50% of cameras
    max_reprojection_error: float = 2.0  # pixels
    # Bounding box (optional, set to None to disable)
    bbox_min: Optional[Tuple[float, float, float]] = None
    bbox_max: Optional[Tuple[float, float, float]] = None
    # Camera model
    camera_model: str = "PINHOLE"  # or "OPENCV" for distortion


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimizer
    learning_rate: float = 1e-4
    lr_joint: float = 1e-5  # LR for joint refinement stage
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    # Batch
    batch_size: int = 4096  # pixels per iteration
    # Stages
    stage1_epochs: int = 5000   # SDF surface learning
    stage2_epochs: int = 20000  # Intensity field pre-training
    stage3_epochs: int = 50000  # Deformation field training
    stage4_epochs: int = 10000  # Joint refinement
    # Epoch definition: one pass through all (camera, load_step) pairs
    # Logging
    log_interval: int = 100    # iterations
    save_interval: int = 5000  # iterations
    # Loss function for photometric term
    photometric_loss: str = "mse"  # "mse" | "znssd"
    znssd_window: int = 15  # ZNSSD window size (if enabled)


@dataclass
class NDeFDICConfig:
    """Master configuration for NDeF-DIC."""
    # Paths
    data_dir: str = ""
    work_dir: str = "output"
    colmap_dir: str = ""  # pre-computed COLMAP results (if available)

    # Image settings
    image_height: int = 1200
    image_width: int = 1920
    n_cameras: int = 4
    n_load_steps: int = 10  # including reference (t=0)

    # Sub-configs
    sdf: SDFConfig = field(default_factory=SDFConfig)
    intensity: IntensityConfig = field(default_factory=IntensityConfig)
    deformation: DeformationConfig = field(default_factory=DeformationConfig)
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    speckle_mask: SpeckleMaskConfig = field(default_factory=SpeckleMaskConfig)
    colmap: COLMAPConfig = field(default_factory=COLMAPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Device
    device: str = "cuda"

    # Seed
    random_seed: int = 42


def get_default_config() -> NDeFDICConfig:
    """Return a default configuration for quick start."""
    return NDeFDICConfig()
