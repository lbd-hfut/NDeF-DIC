"""ROI construction from SfM observations and reference-image texture.

The ROI logic follows the current NDeF-DIC research definition:

1. Remove obvious outlier feature points before drawing the external boundary.
2. Use the convex hull of the remaining SfM feature points as the maximum
   candidate ROI envelope.
3. Use a filtered Delaunay triangulation as the internally supported region.
4. Detect unsupported holes inside the envelope.  A hole is filled back if its
   image texture still looks like speckle; otherwise it is removed from ROI.
5. If user-provided ROI masks are configured, use them directly.

This module intentionally uses only SfM ``observations.npz`` products, not
camera projection code, because the dense initialisation already receives true
per-camera feature observations from SfM.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ROIConfig:
    """Configuration for automatic or user-supplied ROI masks."""

    use_external: bool = False
    external_roi_dir: Optional[str] = None
    external_threshold: int = 127
    outlier_k: int = 6
    outlier_knn_scale: float = 4.0
    component_radius_scale: float = 8.0
    edge_scale: float = 8.0
    radius_scale: float = 6.0
    min_hole_area: int = 500
    tiny_hole_fill_area: int = 3000
    speckle_std_ratio: float = 0.35
    speckle_lap_ratio: float = 0.35
    speckle_grad_ratio: float = 0.35
    min_speckle_std: float = 6.0
    min_speckle_lap: float = 3.0
    overlay_alpha: float = 0.45


@dataclass
class CameraMask:
    """ROI mask and diagnostics for one camera."""

    cam_id: int
    cam_name: str
    mask: np.ndarray
    hull_mask: np.ndarray
    supported_mask: np.ndarray
    rejected_hole_mask: np.ndarray
    hull: np.ndarray
    u_min: float
    u_max: float
    v_min: float
    v_max: float
    n_observations: int = 0
    n_points_after_outlier_filter: int = 0
    n_triangles_raw: int = 0
    n_triangles_valid: int = 0
    n_holes_detected: int = 0
    n_holes_filled_as_speckle: int = 0
    n_holes_rejected: int = 0
    reference_texture: Dict[str, float] | None = None


def run_auto_roi(
    data_dir: str = "case/CylinderDIC",
    sfm_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    config: Optional[ROIConfig] = None,
    verbose: bool = True,
) -> List[CameraMask]:
    """Build ROI masks for a case from SfM products."""
    cfg = config or ROIConfig()
    data_path = Path(data_dir)
    sfm_path = Path(sfm_dir) if sfm_dir else data_path / "result" / "sfm"
    out_path = Path(output_dir) if output_dir else data_path / "result" / "dense" / "auto_roi"

    cameras = _load_npz(sfm_path / "cameras.npz")
    observations = _load_npz(sfm_path / "observations.npz")
    cam_names = [str(x) for x in cameras["cam_names"]]

    if cfg.use_external:
        masks = _load_external_roi_masks(cam_names, cameras["image_paths"], cfg, out_path, verbose)
    else:
        ref_images = _load_reference_images(cameras["image_paths"])
        masks = build_roi_masks_from_observations(
            cam_names=cam_names,
            ref_images=ref_images,
            observations=observations,
            config=cfg,
            verbose=verbose,
        )
        _save_masks(masks, cfg, out_path, ref_images, verbose)

    return masks


def build_roi_masks_from_observations(
    cam_names: List[str],
    ref_images: List[np.ndarray],
    observations: Dict[str, np.ndarray],
    config: Optional[ROIConfig] = None,
    verbose: bool = True,
) -> List[CameraMask]:
    cfg = config or ROIConfig()
    masks: List[CameraMask] = []
    t0 = time.time()
    if verbose:
        print(f"[ROI] Building automatic ROI for {len(cam_names)} cameras")

    for cam_id, cam_name in enumerate(cam_names):
        image = ref_images[cam_id]
        height, width = image.shape[:2]
        uv = observations["uv"][observations["cam_indices"] == cam_id].astype(np.float64)
        cm = _build_single_auto_roi(cam_id, cam_name, image, uv, cfg)
        masks.append(cm)
        if verbose:
            fill = cm.mask.sum() / (height * width) * 100.0
            print(
                f"  {cam_name}: obs={cm.n_observations}, kept={cm.n_points_after_outlier_filter}, "
                f"tri={cm.n_triangles_valid}/{cm.n_triangles_raw}, holes={cm.n_holes_detected}, "
                f"filled={cm.n_holes_filled_as_speckle}, rejected={cm.n_holes_rejected}, "
                f"roi={cm.mask.sum()//1000}K px ({fill:.1f}%)"
            )

    if verbose:
        print(f"[ROI] Automatic ROI complete in {time.time() - t0:.1f}s")
    return masks


def _build_single_auto_roi(
    cam_id: int,
    cam_name: str,
    image: np.ndarray,
    uv: np.ndarray,
    cfg: ROIConfig,
) -> CameraMask:
    import cv2

    height, width = image.shape[:2]
    in_frame = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    uv_in = uv[in_frame]
    uv_clean = _remove_feature_outliers(uv_in, cfg)

    if len(uv_clean) < 3:
        empty = np.zeros((height, width), dtype=bool)
        return CameraMask(
            cam_id=cam_id,
            cam_name=cam_name,
            mask=empty,
            hull_mask=empty.copy(),
            supported_mask=empty.copy(),
            rejected_hole_mask=empty.copy(),
            hull=np.empty((0, 2), dtype=np.float32),
            u_min=0.0,
            u_max=0.0,
            v_min=0.0,
            v_max=0.0,
            n_observations=len(uv_in),
        )

    hull = cv2.convexHull(uv_clean.astype(np.float32)).reshape(-1, 2)
    hull_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(hull_mask, [np.round(hull).astype(np.int32)], 1)
    hull_mask = hull_mask.astype(bool)

    supported_mask, n_tri_raw, n_tri_valid = _build_delaunay_support_mask(
        uv_clean, height, width, cfg
    )
    supported_mask &= hull_mask

    final_mask, rejected_hole_mask, hole_stats, ref_texture = _classify_and_fill_holes(
        image=image,
        hull_mask=hull_mask,
        supported_mask=supported_mask,
        cfg=cfg,
    )
    u_min, u_max, v_min, v_max = _compute_bounds(final_mask)

    return CameraMask(
        cam_id=cam_id,
        cam_name=cam_name,
        mask=final_mask,
        hull_mask=hull_mask,
        supported_mask=supported_mask,
        rejected_hole_mask=rejected_hole_mask,
        hull=hull.astype(np.float32),
        u_min=u_min,
        u_max=u_max,
        v_min=v_min,
        v_max=v_max,
        n_observations=len(uv_in),
        n_points_after_outlier_filter=len(uv_clean),
        n_triangles_raw=n_tri_raw,
        n_triangles_valid=n_tri_valid,
        n_holes_detected=hole_stats["detected"],
        n_holes_filled_as_speckle=hole_stats["filled"],
        n_holes_rejected=hole_stats["rejected"],
        reference_texture=ref_texture,
    )


def _remove_feature_outliers(uv: np.ndarray, cfg: ROIConfig) -> np.ndarray:
    from scipy.spatial import cKDTree

    if len(uv) <= max(3, cfg.outlier_k + 1):
        return uv

    tree = cKDTree(uv)
    dists, _ = tree.query(uv, k=cfg.outlier_k + 1)
    nn = dists[:, 1]
    kth = dists[:, -1]
    median_nn = float(np.median(nn[nn > 0])) if np.any(nn > 0) else float(np.median(kth))
    if not np.isfinite(median_nn) or median_nn <= 0:
        return uv

    density_keep = kth <= cfg.outlier_knn_scale * median_nn
    uv_density = uv[density_keep]
    if len(uv_density) < 3:
        uv_density = uv

    return _keep_largest_radius_component(uv_density, cfg.component_radius_scale * median_nn)


def _keep_largest_radius_component(uv: np.ndarray, radius: float) -> np.ndarray:
    from scipy.spatial import cKDTree

    if len(uv) < 3 or radius <= 0:
        return uv

    tree = cKDTree(uv)
    pairs = list(tree.query_pairs(radius))
    if not pairs:
        return uv

    parent = np.arange(len(uv))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        union(i, j)

    roots = np.array([find(i) for i in range(len(uv))])
    labels, counts = np.unique(roots, return_counts=True)
    largest = labels[np.argmax(counts)]
    keep = roots == largest
    if keep.sum() < 3:
        return uv
    return uv[keep]


def _build_delaunay_support_mask(
    uv: np.ndarray,
    height: int,
    width: int,
    cfg: ROIConfig,
) -> Tuple[np.ndarray, int, int]:
    from scipy.spatial import Delaunay

    d_nn = _median_nn_distance(uv)
    if d_nn <= 0:
        return np.zeros((height, width), dtype=bool), 0, 0
    try:
        tri = Delaunay(uv)
    except Exception:
        return np.zeros((height, width), dtype=bool), 0, 0

    valid = _filter_triangles(uv, tri.simplices, d_nn, cfg)
    valid_tris = tri.simplices[valid]
    if len(valid_tris) == 0:
        return np.zeros((height, width), dtype=bool), len(tri.simplices), 0
    return _rasterize_triangles(uv, valid_tris, height, width), len(tri.simplices), len(valid_tris)


def _median_nn_distance(uv: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    if len(uv) < 2:
        return 0.0
    tree = cKDTree(uv)
    dists, _ = tree.query(uv, k=2)
    nn = dists[:, 1]
    positive = nn[nn > 0]
    if len(positive) == 0:
        return 0.0
    return float(np.median(positive))


def _filter_triangles(uv: np.ndarray, tri_indices: np.ndarray, d_nn: float, cfg: ROIConfig) -> np.ndarray:
    v = uv[tri_indices]
    e0 = v[:, 1] - v[:, 0]
    e1 = v[:, 2] - v[:, 1]
    e2 = v[:, 0] - v[:, 2]
    l0 = np.linalg.norm(e0, axis=-1)
    l1 = np.linalg.norm(e1, axis=-1)
    l2 = np.linalg.norm(e2, axis=-1)
    l_max = np.maximum(np.maximum(l0, l1), l2)
    cross = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
    area = 0.5 * np.abs(cross)
    area_safe = np.maximum(area, 1e-12)
    radius = (l0 * l1 * l2) / (4.0 * area_safe)
    return (l_max < cfg.edge_scale * d_nn) & (radius < cfg.radius_scale * d_nn)


def _rasterize_triangles(
    uv: np.ndarray,
    tri_indices: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    import cv2

    vertices = uv[tri_indices]
    contours = vertices[:, :, None, :].astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, contours, color=1)
    return mask.astype(bool)


def _classify_and_fill_holes(
    image: np.ndarray,
    hull_mask: np.ndarray,
    supported_mask: np.ndarray,
    cfg: ROIConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, float]]:
    import cv2

    candidate_holes = hull_mask & ~supported_mask
    final = supported_mask.copy()
    rejected = np.zeros_like(hull_mask, dtype=bool)
    ref_texture = _texture_metrics(image, supported_mask)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_holes.astype(np.uint8), connectivity=8
    )
    counts = {"detected": 0, "filled": 0, "rejected": 0}
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < cfg.min_hole_area:
            continue
        counts["detected"] += 1
        hole = labels == label
        if area <= cfg.tiny_hole_fill_area:
            final[hole] = True
            counts["filled"] += 1
            continue
        hole_texture = _texture_metrics(image, hole)
        if _is_speckle_like(hole_texture, ref_texture, cfg):
            final[hole] = True
            counts["filled"] += 1
        else:
            rejected[hole] = True
            counts["rejected"] += 1

    final &= hull_mask
    return final, rejected, counts, ref_texture


def _texture_metrics(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    import cv2

    gray = image
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    if mask.sum() == 0:
        return {"std": 0.0, "lap_std": 0.0, "grad_mean": 0.0}
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    values = gray[mask]
    return {
        "std": float(np.std(values)),
        "lap_std": float(np.std(lap[mask])),
        "grad_mean": float(np.mean(grad[mask])),
    }


def _is_speckle_like(hole: Dict[str, float], ref: Dict[str, float], cfg: ROIConfig) -> bool:
    std_ok = hole["std"] >= max(cfg.min_speckle_std, cfg.speckle_std_ratio * ref["std"])
    lap_ok = hole["lap_std"] >= max(cfg.min_speckle_lap, cfg.speckle_lap_ratio * ref["lap_std"])
    grad_ok = hole["grad_mean"] >= cfg.speckle_grad_ratio * ref["grad_mean"]
    return std_ok and lap_ok and grad_ok


def _load_external_roi_masks(
    cam_names: List[str],
    image_paths: np.ndarray,
    cfg: ROIConfig,
    output_dir: Path,
    verbose: bool,
) -> List[CameraMask]:
    import cv2

    if not cfg.external_roi_dir:
        raise ValueError("external_roi_dir must be set when use_external=True")
    roi_dir = Path(cfg.external_roi_dir)
    ref_images = _load_reference_images(image_paths)
    masks = []
    for cam_id, cam_name in enumerate(cam_names):
        candidates = sorted(roi_dir.glob(f"{cam_name}*"))
        if not candidates:
            raise FileNotFoundError(f"No external ROI image for {cam_name} in {roi_dir}")
        raw = cv2.imread(str(candidates[0]), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(candidates[0])
        mask = raw > cfg.external_threshold
        u_min, u_max, v_min, v_max = _compute_bounds(mask)
        masks.append(
            CameraMask(
                cam_id=cam_id,
                cam_name=cam_name,
                mask=mask,
                hull_mask=mask.copy(),
                supported_mask=mask.copy(),
                rejected_hole_mask=np.zeros_like(mask, dtype=bool),
                hull=np.empty((0, 2), dtype=np.float32),
                u_min=u_min,
                u_max=u_max,
                v_min=v_min,
                v_max=v_max,
            )
        )
    _save_masks(masks, cfg, output_dir, ref_images, verbose)
    return masks


def _save_masks(
    masks: List[CameraMask],
    cfg: ROIConfig,
    output_dir: Path,
    ref_images: List[np.ndarray],
    verbose: bool = True,
) -> None:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "mask"
    overlay_dir = output_dir / "overlay"
    debug_dir = output_dir / "debug"
    mask_dir.mkdir(exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)
    debug_dir.mkdir(exist_ok=True)

    meta = {"mode": "external" if cfg.use_external else "auto", "config": asdict(cfg), "cameras": []}
    for cm, ref in zip(masks, ref_images):
        np.save(mask_dir / f"{cm.cam_name}_mask.npy", cm.mask)
        cv2.imwrite(str(mask_dir / f"{cm.cam_name}_mask.png"), cm.mask.astype(np.uint8) * 255)
        cv2.imwrite(str(debug_dir / f"{cm.cam_name}_hull.png"), cm.hull_mask.astype(np.uint8) * 255)
        cv2.imwrite(
            str(debug_dir / f"{cm.cam_name}_delaunay_supported.png"),
            cm.supported_mask.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(debug_dir / f"{cm.cam_name}_rejected_holes.png"),
            cm.rejected_hole_mask.astype(np.uint8) * 255,
        )
        cv2.imwrite(str(overlay_dir / f"{cm.cam_name}_overlay.png"), _make_overlay(ref, cm, cfg))

        meta["cameras"].append(
            {
                "cam_id": cm.cam_id,
                "cam_name": cm.cam_name,
                "mask_pixels": int(cm.mask.sum()),
                "hull_pixels": int(cm.hull_mask.sum()),
                "supported_pixels": int(cm.supported_mask.sum()),
                "rejected_hole_pixels": int(cm.rejected_hole_mask.sum()),
                "u_min": cm.u_min,
                "u_max": cm.u_max,
                "v_min": cm.v_min,
                "v_max": cm.v_max,
                "n_observations": cm.n_observations,
                "n_points_after_outlier_filter": cm.n_points_after_outlier_filter,
                "n_triangles_raw": cm.n_triangles_raw,
                "n_triangles_valid": cm.n_triangles_valid,
                "n_holes_detected": cm.n_holes_detected,
                "n_holes_filled_as_speckle": cm.n_holes_filled_as_speckle,
                "n_holes_rejected": cm.n_holes_rejected,
                "reference_texture": cm.reference_texture,
            }
        )

    with open(output_dir / "auto_roi_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    _save_summary_grid(output_dir / "auto_roi_summary.png", masks, ref_images, cfg)
    if verbose:
        print(f"[ROI] Saved masks, overlays and debug images to {output_dir}")


def _make_overlay(image: np.ndarray, cm: CameraMask, cfg: ROIConfig) -> np.ndarray:
    import cv2

    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()
    color = np.zeros_like(base)
    color[cm.mask] = (0, 180, 0)
    color[cm.rejected_hole_mask] = (0, 0, 220)
    overlay = cv2.addWeighted(base, 1.0 - cfg.overlay_alpha, color, cfg.overlay_alpha, 0)
    if len(cm.hull) >= 3:
        cv2.polylines(overlay, [np.round(cm.hull).astype(np.int32)], True, (255, 255, 255), 2)
    return overlay


def _save_summary_grid(
    path: Path,
    masks: List[CameraMask],
    ref_images: List[np.ndarray],
    cfg: ROIConfig,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), dpi=180, constrained_layout=True)
    fig.suptitle("Automatic ROI: green=ROI, red=rejected holes, white=max feature hull")
    for ax, cm, ref in zip(axes.ravel(), masks, ref_images):
        overlay = _make_overlay(ref, cm, cfg)
        ax.imshow(overlay[..., ::-1])
        ax.set_title(
            f"{cm.cam_name}  holes {cm.n_holes_filled_as_speckle}/{cm.n_holes_detected} filled"
        )
        ax.set_axis_off()
    fig.savefig(path)
    plt.close(fig)


def _compute_bounds(mask: np.ndarray) -> Tuple[float, float, float, float]:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(cols.min()), float(cols.max()), float(rows.min()), float(rows.max())


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _load_reference_images(image_paths: np.ndarray) -> List[np.ndarray]:
    import cv2

    images = []
    for raw_path in image_paths:
        path = Path(str(raw_path))
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        images.append(image)
    return images


def load_masks(dense_dir: str, n_cameras: int) -> List[np.ndarray]:
    """Load saved masks from ``<dense_dir>/mask``.

    The loader accepts both ``cam_0_mask.npy`` and older
    ``cam_0_mask.npy``-style numeric naming patterns.
    """
    mask_dir = Path(dense_dir) / "mask"
    if not mask_dir.exists():
        raise FileNotFoundError(mask_dir)
    masks = []
    for i in range(n_cameras):
        candidates = [
            mask_dir / f"cam_{i}_mask.npy",
            mask_dir / f"{i}_mask.npy",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            matches = sorted(mask_dir.glob(f"*{i}*_mask.npy"), key=lambda p: _natural_key(p.name))
            path = matches[0] if matches else None
        if path is None:
            raise FileNotFoundError(f"Mask file for camera {i} not found under {mask_dir}")
        masks.append(np.load(path).astype(bool))
    return masks


def load_mask_meta(dense_dir: str) -> Dict:
    for name in ("auto_roi_meta.json", "mask_meta.json"):
        path = Path(dense_dir) / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    path = Path(dense_dir) / "mask" / "mask_meta.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"No ROI metadata found under {dense_dir}")


def _natural_key(text: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]
