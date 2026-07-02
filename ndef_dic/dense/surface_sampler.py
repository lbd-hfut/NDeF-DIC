"""Visibility-aware reference surface sampling for the deformation stage."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SurfaceSamplerConfig:
    data_dir: str = "case/CylinderDIC"
    sfm_dir: str | None = None
    reconstruction_dense_dir: str | None = None
    model_init_dir: str | None = None
    output_dir: str | None = None
    relative_sample_spacing: float = 0.005
    robust_percentile_low: float = 1.0
    robust_percentile_high: float = 99.0
    candidate_spacing_factor: float = 0.5
    max_candidate_points: int = 1200000
    min_visible_cameras: int = 2
    depth_tolerance_factor: float = 1.0
    max_final_points: int = 100000
    max_plot_dense_points: int = 60000
    max_plot_visibility_lines: int = 2500
    seed: int = 17


def run_surface_sampler(config: SurfaceSamplerConfig | None = None) -> Dict[str, str]:
    cfg = config or SurfaceSamplerConfig()
    rng = np.random.default_rng(cfg.seed)
    data_dir = Path(cfg.data_dir)
    sfm_dir = Path(cfg.sfm_dir) if cfg.sfm_dir else data_dir / "result" / "sfm"
    reconstruction_dir = (
        Path(cfg.reconstruction_dense_dir)
        if cfg.reconstruction_dense_dir
        else data_dir / "result" / "dense" / "reconstruction_dense"
    )
    model_init_dir = (
        Path(cfg.model_init_dir)
        if cfg.model_init_dir
        else data_dir / "result" / "dense" / "model_init"
    )
    output_dir = (
        Path(cfg.output_dir)
        if cfg.output_dir
        else data_dir / "result" / "dense" / "surface_sampler"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = _load_npz(sfm_dir / "cameras.npz")
    cam_names = [str(x) for x in cameras["cam_names"]]
    K = cameras["K"].astype(np.float64)
    dist = cameras["dist"].astype(np.float64)
    R = cameras["R"].astype(np.float64)
    t = cameras["t"].astype(np.float64).reshape(len(cam_names), 3)
    centres = np.asarray([-R[i].T @ t[i] for i in range(len(cam_names))], dtype=np.float64)
    image_sizes = _load_image_sizes(cameras["image_paths"])

    dense_items = _load_reconstruction_dense(reconstruction_dir, cam_names)
    roi_masks = _load_roi_masks(model_init_dir, cam_names)
    dense_points = np.concatenate([item["world"] for item in dense_items], axis=0)
    bbox_low, bbox_high, object_scale = _robust_bbox(dense_points, cfg)
    spacing = float(cfg.relative_sample_spacing * object_scale)
    if spacing <= 0:
        raise ValueError("Computed non-positive sample spacing.")
    print(f"[SurfaceSampler] object_scale={object_scale:.6g}, sample_spacing={spacing:.6g}")

    surface_maps = _build_surface_maps(dense_items, image_sizes, cam_names)
    candidate_points, candidate_source_cam, candidate_area_stats = _sample_chart_candidates(
        surface_maps=surface_maps,
        cam_names=cam_names,
        final_spacing=spacing,
        cfg=cfg,
        rng=rng,
    )
    print(f"[SurfaceSampler] 2.5D chart candidates: {len(candidate_points)}")

    depth_maps = [item["depth_map"] for item in surface_maps]
    depth_tol = _estimate_depth_tolerance(dense_items, spacing, cfg.depth_tolerance_factor)
    visibility = _compute_visibility(
        points=candidate_points,
        K=K,
        dist=dist,
        R=R,
        t=t,
        roi_masks=roi_masks,
        depth_maps=depth_maps,
        depth_tolerance=depth_tol,
    )
    keep = visibility["visibility_mask"].sum(axis=1) >= cfg.min_visible_cameras
    visible_points = candidate_points[keep]
    visible_source_cam = candidate_source_cam[keep]
    visible_mask = visibility["visibility_mask"][keep]
    visible_uv = visibility["projected_uv"][keep]
    visible_depth = visibility["projected_depth"][keep]
    visible_depth_error = visibility["depth_abs_error"][keep]
    print(f"[SurfaceSampler] visible candidates: {len(visible_points)}")

    final_idx = _select_global_uniform_points(
        points=visible_points,
        visibility=visible_mask,
        depth_abs_error=visible_depth_error,
        spacing=spacing,
        max_points=cfg.max_final_points,
        rng=rng,
    )
    surface_points = visible_points[final_idx]
    surface_normals = _estimate_output_normals(surface_points, dense_points)
    surface_source_cam = visible_source_cam[final_idx]
    surface_visibility = visible_mask[final_idx]
    surface_uv = visible_uv[final_idx]
    surface_depth = visible_depth[final_idx]
    surface_depth_error = visible_depth_error[final_idx]
    visible_counts = surface_visibility.sum(axis=1)
    print(f"[SurfaceSampler] final surface points: {len(surface_points)}")

    np.save(output_dir / "surface_points.npy", surface_points.astype(np.float32))
    np.save(output_dir / "surface_normals.npy", surface_normals.astype(np.float32))
    np.save(output_dir / "source_camera.npy", surface_source_cam.astype(np.int16))
    np.save(output_dir / "visibility_mask.npy", surface_visibility.astype(bool))
    np.save(output_dir / "projected_uv.npy", surface_uv.astype(np.float32))
    np.save(output_dir / "projected_depth.npy", surface_depth.astype(np.float32))
    np.save(output_dir / "depth_abs_error.npy", surface_depth_error.astype(np.float32))
    np.savez_compressed(
        output_dir / "deformation_surface_dataset.npz",
        points=surface_points.astype(np.float32),
        normals=surface_normals.astype(np.float32),
        source_camera=surface_source_cam.astype(np.int16),
        visibility_mask=surface_visibility.astype(bool),
        projected_uv=surface_uv.astype(np.float32),
        projected_depth=surface_depth.astype(np.float32),
        depth_abs_error=surface_depth_error.astype(np.float32),
        visible_counts=visible_counts.astype(np.int16),
        cam_names=np.asarray(cam_names),
    )
    with open(output_dir / "surface_sampler_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "object_scale": object_scale,
                "sample_spacing": spacing,
                "candidate_spacing": spacing * cfg.candidate_spacing_factor,
                "depth_tolerance": depth_tol,
                "bbox_low": bbox_low.tolist(),
                "bbox_high": bbox_high.tolist(),
                "n_dense_input_points": int(len(dense_points)),
                "n_chart_candidates": int(len(candidate_points)),
                "n_visible_candidates": int(len(visible_points)),
                "n_surface_points": int(len(surface_points)),
                "visible_count_min": int(visible_counts.min()) if len(visible_counts) else 0,
                "visible_count_max": int(visible_counts.max()) if len(visible_counts) else 0,
                "visible_count_mean": float(visible_counts.mean()) if len(visible_counts) else 0.0,
                "depth_abs_error_mean": float(np.nanmean(surface_depth_error[surface_visibility])) if surface_visibility.any() else 0.0,
                "chart_area_stats": candidate_area_stats,
                "cam_names": cam_names,
            },
            f,
            indent=2,
        )

    fig3d = output_dir / "surface_visibility_3d.png"
    fig2d = output_dir / "surface_visibility_2d_by_camera.png"
    _plot_surface_visibility_3d(fig3d, dense_points, surface_points, surface_visibility, centres, cam_names, cfg, rng)
    _plot_surface_visibility_2d(fig2d, surface_uv, surface_visibility, roi_masks, cam_names)
    return {
        "dataset": str(output_dir / "deformation_surface_dataset.npz"),
        "figure_3d": str(fig3d),
        "figure_2d": str(fig2d),
        "meta": str(output_dir / "surface_sampler_meta.json"),
    }


def _load_reconstruction_dense(reconstruction_dir: Path, cam_names: List[str]) -> List[Dict[str, np.ndarray]]:
    items = []
    for cam_name in cam_names:
        path = reconstruction_dir / "per_camera_dense" / f"{cam_name}_znssd_dense.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True)
        items.append(
            {
                "pixels": data["pixels"].astype(np.float64),
                "depth": data["pred_depth"].astype(np.float64),
                "world": data["world"].astype(np.float64),
                "roi_mask": data["roi_mask"].astype(bool),
            }
        )
    return items


def _load_roi_masks(model_init_dir: Path, cam_names: List[str]) -> List[np.ndarray]:
    masks = []
    for cam_name in cam_names:
        data = np.load(model_init_dir / "per_camera_dense" / f"{cam_name}_dense_init.npz")
        masks.append(data["roi_mask"].astype(bool))
    return masks


def _robust_bbox(points: np.ndarray, cfg: SurfaceSamplerConfig) -> Tuple[np.ndarray, np.ndarray, float]:
    low = np.percentile(points, cfg.robust_percentile_low, axis=0)
    high = np.percentile(points, cfg.robust_percentile_high, axis=0)
    return low, high, float(np.linalg.norm(high - low))


def _estimate_output_normals(samples: np.ndarray, dense_points: np.ndarray, k: int = 24) -> np.ndarray:
    if len(samples) == 0:
        return np.empty((0, 3), dtype=np.float64)
    from scipy.spatial import cKDTree

    tree = cKDTree(dense_points)
    k_eff = min(max(3, k), len(dense_points))
    _, idx = tree.query(samples, k=k_eff, workers=-1)
    normals = np.zeros_like(samples)
    for i in range(len(samples)):
        nbrs = dense_points[idx[i]]
        centered = nbrs - nbrs.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(1, len(nbrs) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, np.argmin(eigvals)]
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-12)


def _build_surface_maps(
    dense_items: List[Dict[str, np.ndarray]],
    image_sizes: np.ndarray,
    cam_names: List[str],
) -> List[Dict[str, np.ndarray]]:
    maps = []
    for item, (width, height), cam_name in zip(dense_items, image_sizes, cam_names):
        depth_map = np.full((int(height), int(width)), np.nan, dtype=np.float32)
        world_map = np.full((int(height), int(width), 3), np.nan, dtype=np.float32)
        pix = np.rint(item["pixels"]).astype(np.int64)
        valid = (
            (pix[:, 0] >= 0)
            & (pix[:, 0] < width)
            & (pix[:, 1] >= 0)
            & (pix[:, 1] < height)
            & np.isfinite(item["depth"])
        )
        yy = pix[valid, 1]
        xx = pix[valid, 0]
        depth_map[yy, xx] = item["depth"][valid].astype(np.float32)
        world_map[yy, xx] = item["world"][valid].astype(np.float32)
        area = _estimate_pixel_surface_area(world_map, np.isfinite(depth_map))
        maps.append({"depth_map": depth_map, "world_map": world_map, "area": area})
        finite = np.isfinite(depth_map).sum()
        total_area = float(np.nansum(area))
        print(f"[SurfaceSampler] {cam_name}: depth surface pixels={finite}, area={total_area:.6g}")
    return maps


def _estimate_pixel_surface_area(world_map: np.ndarray, valid: np.ndarray) -> np.ndarray:
    area = np.zeros(valid.shape, dtype=np.float64)
    inner = valid[1:-1, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:] & valid[:-2, 1:-1] & valid[2:, 1:-1]
    xu = world_map[1:-1, 2:] - world_map[1:-1, :-2]
    xv = world_map[2:, 1:-1] - world_map[:-2, 1:-1]
    cross = np.cross(xu, xv)
    local_area = 0.25 * np.linalg.norm(cross, axis=2)
    area[1:-1, 1:-1] = np.where(inner & np.isfinite(local_area), local_area, 0.0)
    if np.count_nonzero(area) == 0 and np.any(valid):
        area[valid] = 1.0
    return area


def _sample_chart_candidates(
    surface_maps: List[Dict[str, np.ndarray]],
    cam_names: List[str],
    final_spacing: float,
    cfg: SurfaceSamplerConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    candidate_spacing = final_spacing * cfg.candidate_spacing_factor
    if candidate_spacing <= 0:
        raise ValueError("candidate_spacing_factor produced a non-positive candidate spacing.")
    target_area = candidate_spacing**2

    stats = []
    per_cam_targets = []
    total_requested = 0
    for cam_id, item in enumerate(surface_maps):
        area = item["area"]
        valid_idx = np.flatnonzero(area.reshape(-1) > 0)
        area_sum = float(area.reshape(-1)[valid_idx].sum()) if len(valid_idx) else 0.0
        requested = int(math.ceil(area_sum / target_area)) if area_sum > 0 else 0
        requested = min(requested, len(valid_idx))
        per_cam_targets.append(requested)
        total_requested += requested
        stats.append(
            {
                "cam_name": cam_names[cam_id],
                "surface_area": area_sum,
                "valid_area_pixels": int(len(valid_idx)),
                "requested_candidates": int(requested),
            }
        )

    if total_requested > cfg.max_candidate_points:
        scale = cfg.max_candidate_points / max(total_requested, 1)
        per_cam_targets = [max(1, int(math.floor(n * scale))) if n > 0 else 0 for n in per_cam_targets]

    all_points = []
    all_source = []
    for cam_id, (item, n_pick) in enumerate(zip(surface_maps, per_cam_targets)):
        area_flat = item["area"].reshape(-1)
        valid_idx = np.flatnonzero(area_flat > 0)
        if n_pick <= 0 or len(valid_idx) == 0:
            stats[cam_id]["sampled_candidates"] = 0
            continue
        weights = area_flat[valid_idx]
        weights = weights / max(float(weights.sum()), 1e-12)
        chosen = rng.choice(valid_idx, size=min(n_pick, len(valid_idx)), replace=False, p=weights)
        h, w = item["area"].shape
        y, x = np.unravel_index(chosen, (h, w))
        points = item["world_map"][y, x].astype(np.float64)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        all_points.append(points)
        all_source.append(np.full(len(points), cam_id, dtype=np.int16))
        stats[cam_id]["sampled_candidates"] = int(len(points))

    if not all_points:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.int16), stats
    return np.concatenate(all_points, axis=0), np.concatenate(all_source, axis=0), stats


def _estimate_depth_tolerance(
    dense_items: List[Dict[str, np.ndarray]],
    spacing: float,
    factor: float,
) -> float:
    all_depth = np.concatenate([item["depth"] for item in dense_items])
    spread = float(np.nanpercentile(all_depth, 99) - np.nanpercentile(all_depth, 1))
    return max(spacing * factor, spread * 1e-3)


def _compute_visibility(
    points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    roi_masks: List[np.ndarray],
    depth_maps: List[np.ndarray],
    depth_tolerance: float,
) -> Dict[str, np.ndarray]:
    n_points, n_cams = len(points), len(K)
    visibility = np.zeros((n_points, n_cams), dtype=bool)
    projected_uv = np.full((n_points, n_cams, 2), np.nan, dtype=np.float32)
    projected_depth = np.full((n_points, n_cams), np.nan, dtype=np.float32)
    depth_abs_error = np.full((n_points, n_cams), np.nan, dtype=np.float32)
    for cam_id in range(n_cams):
        uv, depth = _project_points(points, K[cam_id], dist[cam_id], R[cam_id], t[cam_id])
        projected_uv[:, cam_id] = uv.astype(np.float32)
        projected_depth[:, cam_id] = depth.astype(np.float32)
        mask = roi_masks[cam_id]
        height, width = mask.shape
        pix = np.rint(uv).astype(np.int64)
        in_img = (
            (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= width - 1)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= height - 1)
            & (depth > 1e-8)
        )
        in_roi = np.zeros(n_points, dtype=bool)
        idx = np.where(in_img)[0]
        valid_pix = (
            (pix[idx, 0] >= 0)
            & (pix[idx, 0] < width)
            & (pix[idx, 1] >= 0)
            & (pix[idx, 1] < height)
        )
        idx_pix = idx[valid_pix]
        in_roi[idx_pix] = mask[pix[idx_pix, 1], pix[idx_pix, 0]]
        z_ref = _bilinear_sample_depth(depth_maps[cam_id], uv)
        depth_ok = np.zeros(n_points, dtype=bool)
        err = np.abs(depth - z_ref)
        depth_abs_error[:, cam_id] = err.astype(np.float32)
        depth_ok[np.isfinite(err)] = err[np.isfinite(err)] <= depth_tolerance
        visibility[:, cam_id] = in_img & in_roi & depth_ok
    return {
        "visibility_mask": visibility,
        "projected_uv": projected_uv,
        "projected_depth": projected_depth,
        "depth_abs_error": depth_abs_error,
    }


def _bilinear_sample_depth(depth_map: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = depth_map.shape
    x = uv[:, 0]
    y = uv[:, 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
    out = np.full(len(uv), np.nan, dtype=np.float64)
    if not np.any(valid):
        return out
    idx = np.where(valid)[0]
    x0i, x1i = x0[idx], x1[idx]
    y0i, y1i = y0[idx], y1[idx]
    q00 = depth_map[y0i, x0i].astype(np.float64)
    q10 = depth_map[y0i, x1i].astype(np.float64)
    q01 = depth_map[y1i, x0i].astype(np.float64)
    q11 = depth_map[y1i, x1i].astype(np.float64)
    finite = np.isfinite(q00) & np.isfinite(q10) & np.isfinite(q01) & np.isfinite(q11)
    if not np.any(finite):
        return out
    idx_f = idx[finite]
    dx = x[idx_f] - x0[idx_f]
    dy = y[idx_f] - y0[idx_f]
    out[idx_f] = (
        q00[finite] * (1.0 - dx) * (1.0 - dy)
        + q10[finite] * dx * (1.0 - dy)
        + q01[finite] * (1.0 - dx) * dy
        + q11[finite] * dx * dy
    )
    return out


def _project_points(points: np.ndarray, K: np.ndarray, dist: np.ndarray, R: np.ndarray, t: np.ndarray):
    import cv2

    rvec, _ = cv2.Rodrigues(R)
    uv, _ = cv2.projectPoints(points.reshape(-1, 1, 3), rvec, t.reshape(3, 1), K, dist)
    depth = (R @ points.T + t.reshape(3, 1))[2]
    return uv.reshape(-1, 2), depth


def _select_global_uniform_points(
    points: np.ndarray,
    visibility: np.ndarray,
    depth_abs_error: np.ndarray,
    spacing: float,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0,), dtype=np.int64)
    visible_count = visibility.sum(axis=1)
    masked_error = np.where(visibility, depth_abs_error, np.nan)
    mean_error = np.nanmean(masked_error, axis=1)
    mean_error = np.where(np.isfinite(mean_error), mean_error, np.inf)

    vox = np.floor(points / spacing).astype(np.int64)
    order = np.lexsort((mean_error, -visible_count))
    _, first_in_order = np.unique(vox[order], axis=0, return_index=True)
    selected = order[first_in_order]

    if len(selected) > max_points:
        fps_local = _farthest_point_indices(points[selected], max_points, rng)
        selected = selected[fps_local]
    return selected.astype(np.int64)


def _voxel_downsample_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    vox = np.floor(points / voxel_size).astype(np.int64)
    _, first = np.unique(vox, axis=0, return_index=True)
    return np.sort(first.astype(np.int64))


def _farthest_point_indices(points: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    selected = np.empty(n_samples, dtype=np.int64)
    selected[0] = int(rng.integers(0, len(points)))
    min_dist2 = np.full(len(points), np.inf, dtype=np.float64)
    for i in range(1, n_samples):
        diff = points - points[selected[i - 1]]
        min_dist2 = np.minimum(min_dist2, np.einsum("ij,ij->i", diff, diff))
        selected[i] = int(np.argmax(min_dist2))
    return selected


def _plot_surface_visibility_3d(
    path: Path,
    dense_points: np.ndarray,
    surface_points: np.ndarray,
    visibility: np.ndarray,
    centres: np.ndarray,
    cam_names: List[str],
    cfg: SurfaceSamplerConfig,
    rng: np.random.Generator,
) -> None:
    import matplotlib.pyplot as plt

    dense_stride = max(1, len(dense_points) // cfg.max_plot_dense_points)
    dense_plot = dense_points[::dense_stride]
    fig = plt.figure(figsize=(10, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(dense_plot[:, 0], dense_plot[:, 1], dense_plot[:, 2], s=0.08, c="0.78", alpha=0.35)
    ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], s=1.2, c=surface_points[:, 2], cmap="viridis")
    ax.scatter(centres[:, 0], centres[:, 1], centres[:, 2], s=35, c="black", marker="^")
    colors = plt.cm.tab20(np.linspace(0, 1, len(cam_names)))
    visible_pairs = np.argwhere(visibility)
    if len(visible_pairs) > cfg.max_plot_visibility_lines:
        idx = rng.choice(len(visible_pairs), cfg.max_plot_visibility_lines, replace=False)
        visible_pairs = visible_pairs[idx]
    for point_id, cam_id in visible_pairs:
        p = surface_points[point_id]
        c = centres[cam_id]
        ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]], color=colors[cam_id], linewidth=0.18, alpha=0.25)
    for cam_id, name in enumerate(cam_names):
        ax.text(centres[cam_id, 0], centres[cam_id, 1], centres[cam_id, 2], name, fontsize=7)
    ax.set_title("Visibility-aware sampled surface and camera observation rays")
    ax.set_xlabel("SfM world X")
    ax.set_ylabel("SfM world Y")
    ax.set_zlabel("SfM world Z")
    _set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_surface_visibility_2d(
    path: Path,
    projected_uv: np.ndarray,
    visibility: np.ndarray,
    roi_masks: List[np.ndarray],
    cam_names: List[str],
) -> None:
    import cv2
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), dpi=180, constrained_layout=True)
    fig.suptitle("Observed sampled surface points in each camera ROI")
    for ax, cam_id in zip(axes.ravel(), range(len(cam_names))):
        mask = roi_masks[cam_id]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ax.imshow(np.zeros_like(mask), cmap="gray", vmin=0, vmax=1)
        for contour in contours:
            c = contour.reshape(-1, 2)
            ax.plot(c[:, 0], c[:, 1], color="white", linewidth=0.8)
            ax.plot([c[-1, 0], c[0, 0]], [c[-1, 1], c[0, 1]], color="white", linewidth=0.8)
        pts = projected_uv[visibility[:, cam_id], cam_id]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.8, c="#2ca25f", alpha=0.75)
        ax.set_title(f"{cam_names[cam_id]}  n={len(pts)}")
        ax.set_xlim(0, mask.shape[1])
        ax.set_ylim(mask.shape[0], 0)
        ax.set_axis_off()
    fig.savefig(path)
    plt.close(fig)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _load_image_sizes(image_paths: np.ndarray) -> np.ndarray:
    import cv2

    sizes = []
    for raw in image_paths:
        image = cv2.imread(str(Path(str(raw))), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(str(raw))
        height, width = image.shape[:2]
        sizes.append((width, height))
    return np.asarray(sizes, dtype=np.int64)


def _set_axes_equal(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = np.abs(limits[:, 1] - limits[:, 0])
    centres = np.mean(limits, axis=1)
    radius = 0.5 * max(spans)
    ax.set_xlim3d([centres[0] - radius, centres[0] + radius])
    ax.set_ylim3d([centres[1] - radius, centres[1] + radius])
    ax.set_zlim3d([centres[2] - radius, centres[2] + radius])
