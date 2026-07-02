"""NDeF-DIC research pipeline package."""

__version__ = "0.1.0"

from .dense import DepthInitConfig, SfMDepthFiLMNet, run_model_init
from .sfm.reference_sfm import reference_sfm_exists, run_reference_sfm

__all__ = [
    "__version__",
    "reference_sfm_exists",
    "run_reference_sfm",
    "DepthInitConfig",
    "SfMDepthFiLMNet",
    "run_model_init",
]
