"""
NDeF-DIC: Neural Deformation Field for Multi-Camera Digital Image Correlation.

A unified 3D neural field framework that represents multi-camera DIC observations
without stitching, using:
  - Neural SDF for implicit surface representation
  - Intensity field for speckle pattern on the surface
  - 3D deformation field for displacement measurement
  - Per-camera appearance embeddings for exposure compensation
  - Differentiable surface rendering (no volumetric rendering needed)
"""

__version__ = "0.1.0"
