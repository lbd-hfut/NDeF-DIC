"""Dense initialisation modules for NDeF-DIC.

The current dense boundary is intentionally narrow: it converts SfM sparse
observations into a camera-conditioned neural depth initialisation.  Legacy
dense experiments have been removed to avoid mixing incompatible assumptions.
"""

from .model_init import DepthInitConfig, SfMDepthFiLMNet, run_model_init

__all__ = [
    "DepthInitConfig",
    "SfMDepthFiLMNet",
    "run_model_init",
]
