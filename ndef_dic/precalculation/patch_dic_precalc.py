"""Sparse multi-view patch-DIC precalculation for deformation initialization."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PatchDICPrecalcConfig:
    data_dir: str = "case/CylinderDIC"
    sfm_dir: str | None = None
    model_init_dir: str | None = None
    surface_dataset_path: str | None = None
    output_dir: str | None = None
    image_dir: str = "images"
    reference_name: str = "001"
    current_name: str = "002"
    points_per_camera: int = 300
    neighbors_per_camera: int = 2
    patch_radius: int = 10
    cross_search_radius: int = 40
    temporal_search_radius: int = 8
    ncc_threshold_cross: float = 0.45
    ncc_threshold_temporal: float = 0.55
    min_texture_std: float = 0.02
    max_reprojection_error_px: float = 3.0
    displacement_mad_thresh: float = 5.0
    match_batch_size: int = 64
    max_visualization_points: int = 60000
    seed: int = 23
    device: str = "auto"


def run_patch_dic_precalculation(config: PatchDICPrecalcConfig | None = None) -> Dict[str, str]:
    cfg = config or PatchDICPrecalcConfig()
    rng = np.random.default_rng(cfg.seed)
    data_dir = Path(cfg.data_dir)
    sfm_dir = Path(cfg.sfm_dir) if cfg.sfm_dir else data_dir / "result" / "sfm"
    model_init_dir = (
        Path(cfg.model_init_dir)
        if cfg.model_init_dir
        else data_dir / "result" / "dense" / "model_init"
    )
    surface_path = (
        Path(cfg.surface_dataset_path)
        if cfg.surface_dataset_path
        else data_dir / "result" / "dense" / "surface_sampler" / "deformation_surface_dataset.npz"
    )
    output_dir = (
        Path(cfg.output_dir)
        if cfg.output_dir
        else data_dir / "result" / "deformation" / "precalculation" / "patch_dic_sparse"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _select_device(cfg.device)

    cameras = _load_npz(sfm_dir / "cameras.npz")
    surface = np.load(surface_path, allow_pickle=True)
    cam_names = [str(x) for x in surface["cam_names"]]
    K = cameras["K"].astype(np.float64)
    dist = cameras["dist"].astype(np.float64)
    R = cameras["R"].astype(np.float64)
    t = cameras["t"].reshape(len(cam_names), 3).astype(np.float64)

    ref_images_np, cur_images_np, image_sizes = _load_image_pairs(data_dir, cfg.image_dir, cam_names, cfg.reference_name, cfg.current_name)
    roi_masks = _load_roi_masks(model_init_dir, cam_names)
    ref_images = [torch.as_tensor(img, dtype=torch.float32, device=device) for img in ref_images_np]
    cur_images = [torch.as_tensor(img, dtype=torch.float32, device=device) for img in cur_images_np]

    projected_uv = surface["projected_uv"].astype(np.float64)
    visibility = surface["visibility_mask"].astype(bool)
    median_offsets = _estimate_pair_offsets(projected_uv, visibility)
    neighbors = _camera_neighbors(len(cam_names), cfg.neighbors_per_camera)

    all_records = []
    per_camera_stats = []
    for source_cam, cam_name in enumerate(cam_names):
        seeds = _sample_roi_points(
            roi_masks[source_cam],
            ref_images_np[source_cam],
            cfg.points_per_camera,
            cfg.patch_radius,
            cfg.min_texture_std,
            rng,
        )
        if len(seeds) == 0:
            per_camera_stats.append({"cam_name": cam_name, "n_seeds": 0, "n_triangulated": 0})
            continue

        source_current_uv, source_temp_score, source_temp_valid = _match_points_ncc(
            ref_images[source_cam],
            cur_images[source_cam],
            torch.as_tensor(seeds, dtype=torch.float32, device=device),
            torch.as_tensor(seeds, dtype=torch.float32, device=device),
            cfg.patch_radius,
            cfg.temporal_search_radius,
            cfg.match_batch_size,
        )
        source_temp_valid &= source_temp_score >= cfg.ncc_threshold_temporal
        source_current_uv_np = source_current_uv.detach().cpu().numpy()
        source_temp_score_np = source_temp_score.detach().cpu().numpy()
        source_temp_valid_np = source_temp_valid.detach().cpu().numpy()

        ref_observations: List[Dict[int, np.ndarray]] = [{source_cam: seeds[i].astype(np.float64)} for i in range(len(seeds))]
        cur_observations: List[Dict[int, np.ndarray]] = [
            {source_cam: source_current_uv_np[i].astype(np.float64)} if source_temp_valid_np[i] else {}
            for i in range(len(seeds))
        ]
        match_scores: List[List[float]] = [[float(source_temp_score_np[i])] if source_temp_valid_np[i] else [] for i in range(len(seeds))]

        for target_cam in neighbors[source_cam]:
            predicted = seeds + median_offsets[source_cam, target_cam]
            target_ref_uv, cross_score, cross_valid = _match_points_ncc(
                ref_images[source_cam],
                ref_images[target_cam],
                torch.as_tensor(seeds, dtype=torch.float32, device=device),
                torch.as_tensor(predicted, dtype=torch.float32, device=device),
                cfg.patch_radius,
                cfg.cross_search_radius,
                cfg.match_batch_size,
            )
            cross_valid &= cross_score >= cfg.ncc_threshold_cross
            if not torch.any(cross_valid):
                continue
            target_cur_uv, target_temp_score, target_temp_valid = _match_points_ncc(
                ref_images[target_cam],
                cur_images[target_cam],
                target_ref_uv.detach(),
                target_ref_uv.detach(),
                cfg.patch_radius,
                cfg.temporal_search_radius,
                cfg.match_batch_size,
            )
            target_valid = (cross_valid & target_temp_valid & (target_temp_score >= cfg.ncc_threshold_temporal)).detach().cpu().numpy()
            target_ref_np = target_ref_uv.detach().cpu().numpy()
            target_cur_np = target_cur_uv.detach().cpu().numpy()
            cross_score_np = cross_score.detach().cpu().numpy()
            temp_score_np = target_temp_score.detach().cpu().numpy()
            for i in np.where(target_valid)[0]:
                ref_observations[i][target_cam] = target_ref_np[i].astype(np.float64)
                cur_observations[i][target_cam] = target_cur_np[i].astype(np.float64)
                match_scores[i].extend([float(cross_score_np[i]), float(temp_score_np[i])])

        source_records = _triangulate_seed_records(
            source_cam,
            seeds,
            ref_observations,
            cur_observations,
            match_scores,
            K,
            dist,
            R,
            t,
            cfg.max_reprojection_error_px,
        )
        all_records.extend(source_records)
        per_camera_stats.append(
            {
                "cam_name": cam_name,
                "n_seeds": int(len(seeds)),
                "source_temporal_valid": int(source_temp_valid_np.sum()),
                "n_triangulated": int(len(source_records)),
            }
        )
        print(
            f"[PatchDICPrecalc] {cam_name}: seeds={len(seeds)} "
            f"source_temporal={int(source_temp_valid_np.sum())} triangulated={len(source_records)}"
        )

    sparse = _records_to_arrays(all_records)
    filtered = _robust_filter_sparse(sparse, cfg.displacement_mad_thresh)
    scale_stats = _scale_stats(filtered["displacement_magnitude"])

    sparse_path = output_dir / "sparse_displacement.npz"
    np.savez_compressed(sparse_path, **sparse, inlier_mask=filtered["inlier_mask"], **{f"filtered_{k}": v for k, v in filtered.items() if k != "inlier_mask"})
    scale_path = output_dir / "displacement_scale.json"
    with open(scale_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "per_camera_stats": per_camera_stats,
                "n_sparse": int(len(sparse["displacement_magnitude"])),
                "n_inlier": int(len(filtered["displacement_magnitude"])),
                "scale_stats": scale_stats,
            },
            f,
            indent=2,
        )
    fig_path = output_dir / "patch_dic_displacement_3d.png"
    _plot_displacement_components(
        filtered["reference_points"],
        filtered["displacement"],
        fig_path,
        cfg.max_visualization_points,
        cfg.seed,
    )
    return {
        "sparse_displacement": str(sparse_path),
        "scale": str(scale_path),
        "figure": str(fig_path),
    }


def _match_points_ncc(
    ref_image: torch.Tensor,
    target_image: torch.Tensor,
    ref_centers_xy: torch.Tensor,
    target_centers_xy: torch.Tensor,
    patch_radius: int,
    search_radius: int,
    batch_size: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_uv = torch.empty_like(target_centers_xy)
    out_score = torch.full((len(ref_centers_xy),), -1.0, dtype=torch.float32, device=ref_centers_xy.device)
    out_valid = torch.zeros((len(ref_centers_xy),), dtype=torch.bool, device=ref_centers_xy.device)
    offsets = _search_offsets(search_radius, ref_centers_xy.device, ref_centers_xy.dtype)
    for start in range(0, len(ref_centers_xy), batch_size):
        stop = min(start + batch_size, len(ref_centers_xy))
        ref_c = ref_centers_xy[start:stop]
        tgt_c = target_centers_xy[start:stop]
        ref_patch, ref_valid = _extract_centered_windows(ref_image, ref_c, patch_radius)
        target_window, target_valid = _extract_centered_windows(target_image, tgt_c, patch_radius + search_radius)
        if ref_patch.numel() == 0:
            continue
        candidates = F.unfold(target_window, kernel_size=2 * patch_radius + 1)
        ref_flat = ref_patch.reshape(ref_patch.shape[0], -1)
        ref_zero = ref_flat - ref_flat.mean(dim=1, keepdim=True)
        cand_zero = candidates - candidates.mean(dim=1, keepdim=True)
        numerator = (cand_zero * ref_zero[:, :, None]).sum(dim=1)
        denom = torch.sqrt(ref_zero.square().sum(dim=1, keepdim=True) * cand_zero.square().sum(dim=1) + eps)
        score = numerator / denom.clamp_min(eps)
        best_score, best_idx = torch.max(score, dim=1)
        best_offsets = offsets[best_idx]
        valid = ref_valid & target_valid & torch.isfinite(best_score)
        out_uv[start:stop] = tgt_c + best_offsets
        out_score[start:stop] = best_score
        out_valid[start:stop] = valid
    return out_uv, out_score, out_valid


def _extract_centered_windows(image: torch.Tensor, centers_xy: torch.Tensor, radius: int) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = image.shape
    centers = torch.round(centers_xy).long()
    valid = (
        (centers[:, 0] >= radius)
        & (centers[:, 0] < width - radius)
        & (centers[:, 1] >= radius)
        & (centers[:, 1] < height - radius)
    )
    safe_x = centers[:, 0].clamp(radius, width - radius - 1)
    safe_y = centers[:, 1].clamp(radius, height - radius - 1)
    offsets = torch.arange(-radius, radius + 1, device=image.device, dtype=torch.long)
    yy = safe_y[:, None, None] + offsets[None, :, None]
    xx = safe_x[:, None, None] + offsets[None, None, :]
    windows = image[yy, xx].unsqueeze(1)
    return windows, valid


def _search_offsets(radius: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    vals = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    dy, dx = torch.meshgrid(vals, vals, indexing="ij")
    return torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=1)


def _sample_roi_points(
    mask: np.ndarray,
    image: np.ndarray,
    n_points: int,
    margin: int,
    min_texture_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = mask.shape
    valid = mask.copy()
    valid[:margin, :] = False
    valid[-margin:, :] = False
    valid[:, :margin] = False
    valid[:, -margin:] = False
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)
    points = np.column_stack([xs, ys])
    n_side = max(1, int(math.ceil(math.sqrt(n_points))))
    x_edges = np.linspace(points[:, 0].min(), points[:, 0].max() + 1, n_side + 1)
    y_edges = np.linspace(points[:, 1].min(), points[:, 1].max() + 1, n_side + 1)
    selected = []
    for yi in range(n_side):
        for xi in range(n_side):
            in_cell = (
                (points[:, 0] >= x_edges[xi])
                & (points[:, 0] < x_edges[xi + 1])
                & (points[:, 1] >= y_edges[yi])
                & (points[:, 1] < y_edges[yi + 1])
            )
            cell = points[in_cell]
            if len(cell) == 0:
                continue
            order = rng.permutation(len(cell))
            chosen = None
            for idx in order[: min(20, len(order))]:
                x, y = cell[idx]
                patch = image[y - margin : y + margin + 1, x - margin : x + margin + 1]
                if float(patch.std()) >= min_texture_std:
                    chosen = (x, y)
                    break
            if chosen is not None:
                selected.append(chosen)
            if len(selected) >= n_points:
                break
        if len(selected) >= n_points:
            break
    if len(selected) < n_points:
        order = rng.permutation(len(points))
        existing = set(selected)
        for idx in order:
            x, y = map(int, points[idx])
            if (x, y) in existing:
                continue
            patch = image[y - margin : y + margin + 1, x - margin : x + margin + 1]
            if float(patch.std()) >= min_texture_std:
                selected.append((x, y))
            if len(selected) >= n_points:
                break
    return np.asarray(selected, dtype=np.float32)


def _triangulate_seed_records(
    source_cam: int,
    seeds: np.ndarray,
    ref_observations: List[Dict[int, np.ndarray]],
    cur_observations: List[Dict[int, np.ndarray]],
    match_scores: List[List[float]],
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    max_reprojection_error: float,
) -> List[Dict]:
    records = []
    for seed_id, (ref_obs, cur_obs) in enumerate(zip(ref_observations, cur_observations)):
        common_cams = sorted(set(ref_obs.keys()) & set(cur_obs.keys()))
        if len(common_cams) < 2:
            continue
        cam_ids = np.asarray(common_cams, dtype=np.int64)
        ref_uv = np.asarray([ref_obs[c] for c in cam_ids], dtype=np.float64)
        cur_uv = np.asarray([cur_obs[c] for c in cam_ids], dtype=np.float64)
        ref_point = _triangulate_multiview(ref_uv, cam_ids, K, dist, R, t)
        cur_point = _triangulate_multiview(cur_uv, cam_ids, K, dist, R, t)
        if ref_point is None or cur_point is None:
            continue
        ref_reproj = _mean_reprojection_error(ref_point, ref_uv, cam_ids, K, dist, R, t)
        cur_reproj = _mean_reprojection_error(cur_point, cur_uv, cam_ids, K, dist, R, t)
        if max(ref_reproj, cur_reproj) > max_reprojection_error:
            continue
        disp = cur_point - ref_point
        records.append(
            {
                "source_cam": source_cam,
                "seed_id": seed_id,
                "source_uv": seeds[seed_id].astype(np.float64),
                "reference_point": ref_point,
                "current_point": cur_point,
                "displacement": disp,
                "displacement_magnitude": float(np.linalg.norm(disp)),
                "camera_count": len(cam_ids),
                "ref_reprojection_error": ref_reproj,
                "cur_reprojection_error": cur_reproj,
                "mean_match_score": float(np.mean(match_scores[seed_id])) if match_scores[seed_id] else np.nan,
            }
        )
    return records


def _triangulate_multiview(
    uv: np.ndarray,
    cam_ids: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray | None:
    import cv2

    rows = []
    for point_uv, cam_id in zip(uv, cam_ids):
        undist = cv2.undistortPoints(point_uv.reshape(1, 1, 2), K[int(cam_id)], dist[int(cam_id)]).reshape(2)
        P = np.concatenate([R[int(cam_id)], t[int(cam_id)].reshape(3, 1)], axis=1)
        x, y = undist
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    A = np.asarray(rows, dtype=np.float64)
    try:
        _, _, vh = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    Xh = vh[-1]
    if abs(Xh[3]) < 1e-12:
        return None
    return Xh[:3] / Xh[3]


def _mean_reprojection_error(
    point: np.ndarray,
    uv: np.ndarray,
    cam_ids: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> float:
    import cv2

    errs = []
    for target_uv, cam_id in zip(uv, cam_ids):
        rvec, _ = cv2.Rodrigues(R[int(cam_id)])
        proj, _ = cv2.projectPoints(point.reshape(1, 1, 3), rvec, t[int(cam_id)].reshape(3, 1), K[int(cam_id)], dist[int(cam_id)])
        errs.append(float(np.linalg.norm(proj.reshape(2) - target_uv)))
    return float(np.mean(errs)) if errs else float("inf")


def _records_to_arrays(records: List[Dict]) -> Dict[str, np.ndarray]:
    if not records:
        return {
            "source_cam": np.empty((0,), dtype=np.int16),
            "seed_id": np.empty((0,), dtype=np.int32),
            "source_uv": np.empty((0, 2), dtype=np.float32),
            "reference_points": np.empty((0, 3), dtype=np.float32),
            "current_points": np.empty((0, 3), dtype=np.float32),
            "displacement": np.empty((0, 3), dtype=np.float32),
            "displacement_magnitude": np.empty((0,), dtype=np.float32),
            "camera_count": np.empty((0,), dtype=np.int16),
            "ref_reprojection_error": np.empty((0,), dtype=np.float32),
            "cur_reprojection_error": np.empty((0,), dtype=np.float32),
            "mean_match_score": np.empty((0,), dtype=np.float32),
        }
    return {
        "source_cam": np.asarray([r["source_cam"] for r in records], dtype=np.int16),
        "seed_id": np.asarray([r["seed_id"] for r in records], dtype=np.int32),
        "source_uv": np.asarray([r["source_uv"] for r in records], dtype=np.float32),
        "reference_points": np.asarray([r["reference_point"] for r in records], dtype=np.float32),
        "current_points": np.asarray([r["current_point"] for r in records], dtype=np.float32),
        "displacement": np.asarray([r["displacement"] for r in records], dtype=np.float32),
        "displacement_magnitude": np.asarray([r["displacement_magnitude"] for r in records], dtype=np.float32),
        "camera_count": np.asarray([r["camera_count"] for r in records], dtype=np.int16),
        "ref_reprojection_error": np.asarray([r["ref_reprojection_error"] for r in records], dtype=np.float32),
        "cur_reprojection_error": np.asarray([r["cur_reprojection_error"] for r in records], dtype=np.float32),
        "mean_match_score": np.asarray([r["mean_match_score"] for r in records], dtype=np.float32),
    }


def _robust_filter_sparse(sparse: Dict[str, np.ndarray], thresh: float) -> Dict[str, np.ndarray]:
    mag = sparse["displacement_magnitude"]
    if len(mag) == 0:
        out = {k: v.copy() for k, v in sparse.items()}
        out["inlier_mask"] = np.zeros((0,), dtype=bool)
        return out
    med = np.median(mag)
    mad = np.median(np.abs(mag - med))
    keep = np.ones(len(mag), dtype=bool) if mad < 1e-12 else np.abs(mag - med) <= thresh * 1.4826 * mad
    return {k: v[keep] for k, v in sparse.items()} | {"inlier_mask": keep}


def _scale_stats(mag: np.ndarray) -> Dict:
    if len(mag) == 0:
        return {"median": None, "mean": None, "p75": None, "p90": None, "max": None}
    return {
        "median": float(np.median(mag)),
        "mean": float(np.mean(mag)),
        "p75": float(np.percentile(mag, 75)),
        "p90": float(np.percentile(mag, 90)),
        "max": float(np.max(mag)),
    }


def _camera_neighbors(n_cameras: int, neighbors_per_camera: int) -> List[List[int]]:
    neighbors = []
    for i in range(n_cameras):
        row = []
        for offset in range(1, neighbors_per_camera + 1):
            row.extend([(i - offset) % n_cameras, (i + offset) % n_cameras])
        dedup = []
        for item in row:
            if item != i and item not in dedup:
                dedup.append(item)
        neighbors.append(dedup[:neighbors_per_camera])
    return neighbors


def _estimate_pair_offsets(projected_uv: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    n_cams = projected_uv.shape[1]
    offsets = np.zeros((n_cams, n_cams, 2), dtype=np.float32)
    for i in range(n_cams):
        for j in range(n_cams):
            both = visibility[:, i] & visibility[:, j] & np.isfinite(projected_uv[:, i]).all(axis=1) & np.isfinite(projected_uv[:, j]).all(axis=1)
            if np.any(both):
                offsets[i, j] = np.median(projected_uv[both, j] - projected_uv[both, i], axis=0).astype(np.float32)
    return offsets


def _load_image_pairs(
    data_dir: Path,
    image_dir: str,
    cam_names: List[str],
    reference_name: str,
    current_name: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    import cv2

    refs, curs, sizes = [], [], []
    for cam_name in cam_names:
        ref_path = _find_named_image(data_dir / image_dir / cam_name, reference_name)
        cur_path = _find_named_image(data_dir / image_dir / cam_name, current_name)
        ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
        cur = cv2.imread(str(cur_path), cv2.IMREAD_GRAYSCALE)
        if ref is None or cur is None:
            raise FileNotFoundError(f"Failed to read {ref_path} or {cur_path}")
        refs.append(ref.astype(np.float32) / 255.0)
        curs.append(cur.astype(np.float32) / 255.0)
        sizes.append((ref.shape[1], ref.shape[0]))
    return refs, curs, np.asarray(sizes, dtype=np.int64)


def _load_roi_masks(model_init_dir: Path, cam_names: List[str]) -> List[np.ndarray]:
    masks = []
    for cam_name in cam_names:
        data = np.load(model_init_dir / "per_camera_dense" / f"{cam_name}_dense_init.npz")
        masks.append(data["roi_mask"].astype(bool))
    return masks


def _find_named_image(cam_dir: Path, stem: str) -> Path:
    for ext in (".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg"):
        path = cam_dir / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image {stem} in {cam_dir}")


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[PatchDICPrecalc] CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _plot_displacement_components(
    points: np.ndarray,
    displacement: np.ndarray,
    path: Path,
    max_points: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    if len(points) == 0:
        return
    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        pts = points[idx]
        disp = displacement[idx]
    else:
        pts = points
        disp = displacement
    magnitude = np.linalg.norm(disp, axis=1)
    values = [magnitude, disp[:, 0], disp[:, 1], disp[:, 2]]
    titles = ["Total displacement", "U displacement", "V displacement", "W displacement"]
    labels = ["|u|", "U", "V", "W"]
    fig = plt.figure(figsize=(14, 11), dpi=180)
    for i, (value, title, label) in enumerate(zip(values, titles, labels), start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        scatter = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=value, s=6.0, cmap="viridis" if i == 1 else "coolwarm", linewidths=0.0)
        ax.set_title(title)
        ax.set_xlabel("SfM world X")
        ax.set_ylabel("SfM world Y")
        ax.set_zlabel("SfM world Z")
        _set_axes_equal(ax, pts)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.08)
        cbar.set_label(label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _set_axes_equal(ax, points: np.ndarray) -> None:
    limits = np.array(
        [
            [float(points[:, 0].min()), float(points[:, 0].max())],
            [float(points[:, 1].min()), float(points[:, 1].max())],
            [float(points[:, 2].min()), float(points[:, 2].max())],
        ]
    )
    spans = np.maximum(limits[:, 1] - limits[:, 0], 1e-12)
    centers = limits.mean(axis=1)
    radius = 0.5 * float(spans.max())
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])
