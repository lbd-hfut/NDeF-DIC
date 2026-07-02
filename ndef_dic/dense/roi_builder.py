"""
ROI / mask builder via Delaunay triangulation of COLMAP sparse projections.

Section 3 of ``multiview_selfcalib_dic_neural_depth_plan.md``:

    1. Project COLMAP sparse 3D points → per-camera 2D point set.
    2. 2D Delaunay triangulation.
    3. Filter triangles by max edge length and circumradius.
    4. Rasterise valid triangles → binary mask M_i.
    5. Extract per-camera mask bounds (u_min, u_max, v_min, v_max).

No dilation — the mask strictly follows the triangle-supported region.
"""

from __future__ import annotations

import os
import json
import time
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections.abc import Sequence

from .camera_model import CameraGeometry, project as project_np


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class ROIConfig:
    """Parameters for ROI mask construction.

    Two modes:

    - **Delaunay** (``use_external=False``, default):
      Project COLMAP sparse points, triangulate, filter, rasterise.
      Controlled by ``edge_scale``, ``radius_scale``, ``max_points_per_camera``.

    - **External** (``use_external=True``):
      Load pre-provided ROI images — one per camera, the **last** image file
      in each camera folder.  Thresholded at ``external_threshold`` to
      produce a binary mask.  Delaunay parameters are ignored.

    Attributes:
        use_external:       If True, load masks from per-camera image folders
                            instead of building via Delaunay.
        external_threshold: Pixel value threshold for external mask images
                            (0-255, default 127).  Pixels > threshold = valid.
        edge_scale  (c_l):  max edge / d_nn threshold  (default 8.0).
        radius_scale (c_R): circumradius / d_nn threshold (default 6.0).
        max_points_per_camera: subsample sparse points if above this
                               (0 = no limit).
    """

    use_external: bool = False
    external_threshold: int = 127
    edge_scale: float = 8.0
    radius_scale: float = 6.0
    max_points_per_camera: int = 30000


# =========================================================================
# Per-camera result
# =========================================================================

@dataclass
class CameraMask:
    """Mask and metadata for one camera."""

    cam_id: int
    mask: np.ndarray                              # (H, W) bool
    u_min: float
    u_max: float
    v_min: float
    v_max: float
    n_sparse_projected: int = 0
    n_triangles_raw: int = 0
    n_triangles_valid: int = 0


# =========================================================================
# Main entry point
# =========================================================================

def build_roi_masks(
    sparse_points: Optional[np.ndarray] = None,    # (M, 3) — unused when use_external=True
    geometries: Optional[Sequence[CameraGeometry]] = None,
    config: Optional[ROIConfig] = None,
    output_dir: Optional[str] = None,
    ref_images: Optional[List[np.ndarray]] = None,  # (H, W) uint8, for debug overlay
    image_dir: Optional[str] = None,                 # needed when use_external=True
    verbose: bool = True,
) -> List[CameraMask]:
    """Build per-camera ROI masks.

    Two modes, controlled by ``config.use_external``:

    **Delaunay mode** (default):
        Project COLMAP sparse points → 2D Delaunay → filter → rasterise.
        Requires ``sparse_points`` and ``geometries``.

    **External mode** (``config.use_external=True``):
        Load the last image file from each camera folder under ``image_dir``,
        threshold to binary, and extract bounds.
        ``sparse_points`` and ``geometries`` are ignored except for
        ``geo.width`` / ``geo.height`` which are used if ``geometries``
        is provided (otherwise inferred from the loaded mask).

    Args:
        sparse_points: (M, 3) COLMAP sparse 3D points (Delaunay mode).
        geometries:    list of N CameraGeometry objects.
        config:        ROIConfig (uses defaults if None).
        output_dir:    if set, saves .npy + .png + meta.json under
                       ``<output_dir>/mask/``.
        ref_images:    optional reference images (uint8) for debug overlay.
        image_dir:     root directory containing ``cam_0/``, ``cam_1/``, …
                       subdirectories (external mode).
        verbose:       print per-camera statistics.

    Returns:
        List of CameraMask, one per camera.
    """
    cfg = config or ROIConfig()

    if cfg.use_external:
        return _build_masks_from_external(
            image_dir=image_dir,
            geometries=geometries,
            threshold=cfg.external_threshold,
            output_dir=output_dir,
            ref_images=ref_images,
            verbose=verbose,
        )

    # ---- Delaunay mode ----
    if sparse_points is None or geometries is None:
        raise ValueError(
            "Delaunay mode requires sparse_points and geometries. "
            "Set config.use_external=True to load pre-provided mask images."
        )

    n_cam = len(geometries)
    masks: List[CameraMask] = []

    if verbose:
        print(f"[ROI] Building masks for {n_cam} cameras "
              f"from {len(sparse_points)} sparse points "
              f"(c_l={cfg.edge_scale}, c_R={cfg.radius_scale})")

    t0 = time.time()

    for cam_id, geo in enumerate(geometries):
        cm = _build_single_delaunay_mask(
            cam_id, geo, sparse_points, cfg,
        )
        masks.append(cm)

        if verbose:
            fill = cm.mask.sum() / (geo.height * geo.width) * 100
            print(f"  Cam {cam_id:2d}: {cm.n_sparse_projected:5d} projected → "
                  f"{cm.n_triangles_raw}△ raw → {cm.n_triangles_valid}△ valid "
                  f"→ {cm.mask.sum()//1000}K px ({fill:.1f}%) "
                  f"bounds=[{cm.u_min:.0f},{cm.u_max:.0f}]×[{cm.v_min:.0f},{cm.v_max:.0f}]")

    if verbose:
        print(f"[ROI] Built {n_cam} masks in {time.time() - t0:.1f}s")

    if output_dir is not None:
        _save_masks(masks, cfg, output_dir, ref_images, verbose)

    return masks


# =========================================================================
# Single-camera mask construction
# =========================================================================

def _build_single_delaunay_mask(
    cam_id: int,
    geo: CameraGeometry,
    sparse_pts: np.ndarray,
    cfg: ROIConfig,
    verbose: bool,
) -> CameraMask:
    """Build the mask for one camera."""
    H, W = geo.height, geo.width

    # 1. Project sparse points → 2D, keep only in-frame + positive depth
    uv_all, depth = project_np(sparse_pts, geo.K, geo.R, geo.t)
    in_frame = (
        (uv_all[:, 0] >= 0) & (uv_all[:, 0] < W) &
        (uv_all[:, 1] >= 0) & (uv_all[:, 1] < H) &
        (depth > 1e-6)
    )
    uv = uv_all[in_frame]
    n_proj = len(uv)

    if n_proj < 3:
        # Not enough points to triangulate — return empty mask
        return CameraMask(
            cam_id=cam_id,
            mask=np.zeros((H, W), dtype=bool),
            u_min=0.0, u_max=float(W),
            v_min=0.0, v_max=float(H),
            n_sparse_projected=n_proj,
        )

    # 2. Subsample if needed
    if cfg.max_points_per_camera > 0 and n_proj > cfg.max_points_per_camera:
        rng = np.random.RandomState(cam_id)
        idx = rng.choice(n_proj, cfg.max_points_per_camera, replace=False)
        uv = uv[idx]
        n_proj = len(uv)

    # 3. Nearest-neighbour distance (median) for scale reference
    d_nn = _median_nn_distance(uv)
    if d_nn <= 0:
        return CameraMask(
            cam_id=cam_id, mask=np.zeros((H, W), dtype=bool),
            u_min=0.0, u_max=float(W), v_min=0.0, v_max=float(H),
            n_sparse_projected=n_proj,
        )

    # 4. 2D Delaunay triangulation
    from scipy.spatial import Delaunay
    try:
        tri = Delaunay(uv)                           # (n_tri, 3) indices
    except Exception:
        return CameraMask(
            cam_id=cam_id, mask=np.zeros((H, W), dtype=bool),
            u_min=0.0, u_max=float(W), v_min=0.0, v_max=float(H),
            n_sparse_projected=n_proj,
        )

    n_tri_raw = len(tri.simplices)

    # 5. Filter valid triangles
    valid_mask = _filter_triangles(uv, tri.simplices, d_nn, cfg)
    valid_tris = tri.simplices[valid_mask]
    n_tri_valid = len(valid_tris)

    # 6. Rasterise valid triangles → binary mask
    if n_tri_valid > 0:
        mask = _rasterize_triangles(uv, valid_tris, H, W)
    else:
        mask = np.zeros((H, W), dtype=bool)

    # 7. Extract mask bounds
    u_min, u_max, v_min, v_max = _compute_bounds(mask)

    return CameraMask(
        cam_id=cam_id,
        mask=mask,
        u_min=u_min, u_max=u_max,
        v_min=v_min, v_max=v_max,
        n_sparse_projected=n_proj,
        n_triangles_raw=n_tri_raw,
        n_triangles_valid=n_tri_valid,
    )


# =========================================================================
# Triangle filtering
# =========================================================================

def _median_nn_distance(uv: np.ndarray) -> float:
    """Median nearest-neighbour distance among 2D points."""
    from scipy.spatial import cKDTree
    if len(uv) < 2:
        return 0.0
    tree = cKDTree(uv)
    # k=2 because the first neighbour is the point itself (distance 0)
    dists, _ = tree.query(uv, k=2)
    nn = dists[:, 1] if dists.ndim == 2 else np.array([dists[1]])
    return float(np.median(nn))


def _filter_triangles(
    uv: np.ndarray,                          # (M, 2)
    tri_indices: np.ndarray,                 # (T, 3) indices into uv
    d_nn: float,
    cfg: ROIConfig,
) -> np.ndarray:
    """Return boolean mask of triangles that pass the geometric filters.

    For each triangle:
      - max edge length  l_max < cfg.edge_scale  * d_nn
      - circumradius     R     < cfg.radius_scale * d_nn
    """
    T = len(tri_indices)

    # Triangle vertices: (T, 3, 2)
    v = uv[tri_indices]                       # (T, 3, 2)

    # Edge vectors: (T, 3, 2)
    e0 = v[:, 1] - v[:, 0]                    # (T, 2)
    e1 = v[:, 2] - v[:, 1]                    # (T, 2)
    e2 = v[:, 0] - v[:, 2]                    # (T, 2)

    # Edge lengths
    l0 = np.linalg.norm(e0, axis=-1)          # (T,)
    l1 = np.linalg.norm(e1, axis=-1)
    l2 = np.linalg.norm(e2, axis=-1)
    l_max = np.maximum(np.maximum(l0, l1), l2)

    # Area = 0.5 * |cross(e0, -e2)| = 0.5 * |cross(e0, e0 + e1)|
    # cross2D(a, b) = a_x * b_y - a_y * b_x
    cross = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
    area = 0.5 * np.abs(cross)                 # (T,)
    area_safe = np.maximum(area, 1e-12)

    # Circumradius: R = (l0 * l1 * l2) / (4 * area)
    R = (l0 * l1 * l2) / (4.0 * area_safe)

    # Filters
    edge_ok = l_max < cfg.edge_scale * d_nn
    radius_ok = R < cfg.radius_scale * d_nn

    return edge_ok & radius_ok


# =========================================================================
# Rasterisation
# =========================================================================

def _rasterize_triangles(
    uv: np.ndarray,              # (M, 2) all point coords
    tri_indices: np.ndarray,     # (T, 3) indices of valid triangles
    H: int,
    W: int,
) -> np.ndarray:
    """Rasterise valid triangles into a (H, W) binary mask via OpenCV.

    Uses ``cv2.fillPoly`` for efficiency — it handles thousands of
    triangles in a single call.
    """
    import cv2

    # Build (T, 3, 1, 2) contour array for fillPoly
    vertices = uv[tri_indices]                     # (T, 3, 2)
    contours = vertices[:, :, None, :].astype(np.int32)  # (T, 3, 1, 2)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, contours, color=1)

    return mask.astype(bool)


# =========================================================================
# Mask bounds
# =========================================================================

def _compute_bounds(mask: np.ndarray) -> Tuple[float, float, float, float]:
    """Axis-aligned bounding box of the mask's True region.

    If the mask is empty, returns the full image extent as fallback.
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return (0.0, float(mask.shape[1]),
                0.0, float(mask.shape[0]))
    return (float(cols.min()), float(cols.max()),
            float(rows.min()), float(rows.max()))


# =========================================================================
# External mask loading
# =========================================================================

_IMAGE_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _is_image(fname: str) -> bool:
    return fname.lower().endswith(_IMAGE_EXTS)


def _list_image_files(directory: str) -> List[str]:
    """Sorted list of image files in a directory."""
    files = sorted([f for f in os.listdir(directory) if _is_image(f)])
    if not files:
        raise FileNotFoundError(f"No image files in {directory}")
    return files


def _build_masks_from_external(
    image_dir: Optional[str],
    geometries: Optional[Sequence[CameraGeometry]],
    threshold: int,
    output_dir: Optional[str],
    ref_images: Optional[List[np.ndarray]],
    verbose: bool,
) -> List[CameraMask]:
    """Load ROI masks from the last image in each camera folder.

    Expects ``image_dir/cam_0/``, ``image_dir/cam_1/``, … each containing
    a series of images whose **last** file is the ROI mask.
    """
    if image_dir is None:
        raise ValueError(
            "External mask mode requires image_dir. "
            "Set config.use_external=False to build masks via Delaunay."
        )

    import cv2

    # Discover camera folders
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    cam_names = sorted([
        d for d in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, d)) and d.startswith("cam_")
    ])
    if not cam_names:
        raise FileNotFoundError(f"No cam_* folders under {image_dir}")

    n_cam = len(cam_names)
    masks: List[CameraMask] = []

    if verbose:
        print(f"[ROI] Loading external masks for {n_cam} cameras "
              f"(threshold={threshold})")

    t0 = time.time()

    for cam_id, cam_name in enumerate(cam_names):
        cam_dir = os.path.join(image_dir, cam_name)
        files = _list_image_files(cam_dir)

        # Last image file is the ROI mask
        mask_path = os.path.join(cam_dir, files[-1])

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise RuntimeError(f"Failed to read external mask: {mask_path}")

        H, W = mask_img.shape

        # Validate dimensions against geometry (if provided)
        if geometries is not None and cam_id < len(geometries):
            geo = geometries[cam_id]
            if (H != geo.height or W != geo.width):
                raise ValueError(
                    f"Cam {cam_id} ({cam_name}): mask dimensions {W}×{H} "
                    f"don't match geometry {geo.width}×{geo.height}. "
                    f"Check that the last image in {cam_dir} is the ROI mask."
                )

        # Threshold → binary
        mask = mask_img > threshold

        # Extract bounds
        u_min, u_max, v_min, v_max = _compute_bounds(mask)

        cm = CameraMask(
            cam_id=cam_id,
            mask=mask,
            u_min=u_min, u_max=u_max,
            v_min=v_min, v_max=v_max,
            n_sparse_projected=0,
            n_triangles_raw=0,
            n_triangles_valid=0,
        )
        masks.append(cm)

        if verbose:
            fill = mask.sum() / (H * W) * 100
            print(f"  Cam {cam_id:2d} ({cam_name}): {os.path.basename(mask_path)} "
                  f"→ {mask.sum()//1000}K px ({fill:.1f}%) "
                  f"bounds=[{u_min:.0f},{u_max:.0f}]×[{v_min:.0f},{v_max:.0f}]")

    if verbose:
        print(f"[ROI] Loaded {n_cam} external masks in {time.time() - t0:.1f}s")

    if output_dir is not None:
        _save_masks(masks, ROIConfig(
            use_external=True, external_threshold=threshold,
        ), output_dir, ref_images, verbose)

    return masks


# =========================================================================
# Persistence
# =========================================================================

def _save_masks(
    masks: List[CameraMask],
    cfg: ROIConfig,
    output_dir: str,
    ref_images: Optional[List[np.ndarray]] = None,
    verbose: bool = True,
):
    """Save mask arrays, metadata JSON, and visualisation PNGs.

    All mask files go under ``output_dir/mask/``::

        mask/
          cam_0_mask.npy   cam_0_mask.png   cam_0_overlay.png
          ...
          mask_meta.json
    """
    mask_dir = os.path.join(output_dir, "mask")
    os.makedirs(mask_dir, exist_ok=True)

    # ---- Save .npy masks ----
    for cm in masks:
        np.save(os.path.join(mask_dir, f"cam_{cm.cam_id}_mask.npy"), cm.mask)

    # ---- Save metadata JSON ----
    meta = {
        "mode": "external" if cfg.use_external else "delaunay",
        "params": {
            "use_external": cfg.use_external,
            "external_threshold": cfg.external_threshold,
            "edge_scale": cfg.edge_scale,
            "radius_scale": cfg.radius_scale,
            "max_points_per_camera": cfg.max_points_per_camera,
        },
        "cameras": [
            {
                "cam_id": cm.cam_id,
                "n_sparse_projected": cm.n_sparse_projected,
                "n_triangles_raw": cm.n_triangles_raw,
                "n_triangles_valid": cm.n_triangles_valid,
                "mask_pixels": int(cm.mask.sum()),
                "u_min": cm.u_min, "u_max": cm.u_max,
                "v_min": cm.v_min, "v_max": cm.v_max,
            }
            for cm in masks
        ],
    }
    meta_path = os.path.join(mask_dir, "mask_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"[ROI] Saved {len(masks)} masks + meta → {mask_dir}/")

    # ---- Save visualisation PNGs ----
    _save_mask_pngs(masks, mask_dir, ref_images, verbose)


def _save_mask_pngs(
    masks: List[CameraMask],
    mask_dir: str,
    ref_images: Optional[List[np.ndarray]] = None,
    verbose: bool = True,
):
    """Save mask .png and overlay .png alongside the .npy files."""
    import cv2

    for cm in masks:
        H, W = cm.mask.shape

        # Plain binary mask
        cv2.imwrite(
            os.path.join(mask_dir, f"cam_{cm.cam_id}_mask.png"),
            cm.mask.astype(np.uint8) * 255,
        )

        # Overlay: green mask on dimmed reference
        if ref_images is not None and cm.cam_id < len(ref_images):
            ref = ref_images[cm.cam_id]
            if ref.ndim == 2:
                ref_rgb = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
            else:
                ref_rgb = ref.copy()

            vis = np.zeros((H, W, 3), dtype=np.uint8)
            vis[cm.mask] = [0, 200, 0]                # green
            alpha = 0.4
            blended = cv2.addWeighted(ref_rgb, alpha, vis, 1.0 - alpha, 0)
            cv2.imwrite(os.path.join(mask_dir, f"cam_{cm.cam_id}_overlay.png"), blended)

    if verbose:
        print(f"[ROI] PNG visualisations → {mask_dir}/")


# =========================================================================
# Convenience: load saved masks back
# =========================================================================

def load_masks(dense_dir: str, n_cameras: int) -> List[np.ndarray]:
    """Load saved ``cam_{i}_mask.npy`` files from ``<dense_dir>/mask/``.

    Args:
        dense_dir:  the dense output directory (contains ``mask/`` subdir).
        n_cameras:  expected number of cameras.
    Returns:
        list of (H, W) bool arrays.
    """
    mask_dir = os.path.join(dense_dir, "mask")
    masks = []
    for i in range(n_cameras):
        path = os.path.join(mask_dir, f"cam_{i}_mask.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mask file not found: {path}")
        masks.append(np.load(path))
    return masks


def load_mask_meta(dense_dir: str) -> Dict:
    """Load ``mask_meta.json`` from ``<dense_dir>/mask/``."""
    path = os.path.join(dense_dir, "mask", "mask_meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask meta not found: {path}")
    with open(path, "r") as f:
        return json.load(f)
