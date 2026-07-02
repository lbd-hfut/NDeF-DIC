"""
MATLAB .mat I/O utilities.

COLMAP calibration files are saved via ``scipy.io.savemat`` and loaded via
``scipy.io.loadmat``.  Depending on the MATLAB / scipy version and the shape
of the data, ``loadmat`` may return:

* **Numeric arrays** — regular ``float64`` ndarrays (newer format, preferred).
* **Object arrays** — ``dtype=object`` with nested 0-d arrays (older format,
  produced when array elements have heterogeneous shapes or when
  ``savemat(..., oned_as='column')`` interacts poorly with cell-like nesting).

This module provides two thin helpers that normalize both formats into clean
``float64`` ndarrays so downstream code can work with a single representation.
"""

import numpy as np
from typing import Tuple


def unwrap_mat_cell(obj: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Extract a single element from a MATLAB object array into a clean array.

    Handles the common pattern where ``loadmat`` returns ``dtype=object``
    nested arrays, e.g. a ``(3, 3)`` calibration matrix stored as a 0-d
    object array of 9 floats.

    If *obj* is already a numeric array, it is returned as ``float64``
    reshaped to *shape* (no-op when the shape already matches).

    Args:
        obj:   A single element from a ``loadmat`` object array, or a
               regular numeric array.
        shape: Expected output shape, e.g. ``(3, 3)`` for K / R,
               ``(3, 1)`` for t, ``(5,)`` for dist.

    Returns:
        ``float64`` ndarray with the requested *shape*.
    """
    if obj.dtype != np.dtype("object"):
        return np.asarray(obj, dtype=np.float64).reshape(shape)

    n_expected = int(np.prod(shape))
    actual = obj.size
    if actual != n_expected:
        raise ValueError(
            f"Object cell has {actual} elements but shape {shape} "
            f"requires {n_expected}"
        )

    flat = np.array(
        [float(obj.flat[i].item()) for i in range(actual)],
        dtype=np.float64,
    )
    return flat.reshape(shape)


def unwrap_mat_batch(
    arr: np.ndarray,
    per_element_shape: Tuple[int, ...],
) -> np.ndarray:
    """Extract a batch of MATLAB object-array elements into a stacked array.

    If *arr* is already numeric, it is returned as ``float64`` unchanged.
    If *arr* has ``dtype=object``, each element is recursively extracted
    and the results are stacked into ``(arr.shape[0], *per_element_shape)``.

    This is the batch version of :func:`unwrap_mat_cell` — it processes an
    entire ``loadmat`` array at once rather than one element at a time.

    Args:
        arr:               1-D MATLAB array from ``loadmat`` (object or numeric).
        per_element_shape: Shape of each element, e.g. ``(3,)`` for 3-D points,
                           ``(3, 3)`` for rotation matrices.

    Returns:
        ``float64`` ndarray with shape ``(N, *per_element_shape)`` where
        ``N = arr.shape[0]``.
    """
    if arr.dtype != np.dtype("object"):
        return np.asarray(arr, dtype=np.float64)

    n = arr.shape[0]
    n_elements = int(np.prod(per_element_shape))
    flat = np.array(
        [
            [float(arr[i].flat[j].item()) for j in range(n_elements)]
            for i in range(n)
        ],
        dtype=np.float64,
    )
    return flat.reshape(n, *per_element_shape)
