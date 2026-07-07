"""Chessboard-based SfM-to-world scale estimation."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg")


@dataclass
class ChessboardScaleConfig:
    data_dir: str = "case/CylinderDIC"
    sfm_dir: str | None = None
    image_dir: str = "calibrate_images"
    image_name: str | None = None
    output_dir: str | None = None
    inner_cols: int = 9
    inner_rows: int = 7
    square_size: float = 10.0
    pair_selection: str = "middle"  # "middle" | "max_baseline" | "all_pairs"
    scale_stat: str = "trimmed_mean"  # "mean" | "median" | "trimmed_mean" | "p25"
    trim_fraction: float = 0.20
    max_pair_edge_cv: float = 0.08
    max_edge_cv_warning: float = 0.05
    corner_order_edge_cv_weight: float = 1.0
    subpix_window: int = 11
    min_common_corners: int = 12
    max_reprojection_error_px: float = 3.0
    save_overlays: bool = True


def load_sfm_cameras(sfm_dir: Path) -> Dict[str, np.ndarray]:
    path = sfm_dir / "cameras.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def find_named_image(cam_dir: Path, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        path = cam_dir / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image named {stem} with supported extension in {cam_dir}")


def find_single_image(cam_dir: Path) -> Path:
    if not cam_dir.is_dir():
        raise FileNotFoundError(cam_dir)
    images = sorted([p for p in cam_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    if len(images) != 1:
        raise FileNotFoundError(
            f"Expected exactly one calibration image in {cam_dir}, found {len(images)}"
        )
    return images[0]


def detect_chessboard_corners(
    image_path: Path,
    pattern_size: Tuple[int, int],
    subpix_window: int,
) -> tuple[bool, np.ndarray, np.ndarray]:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(img, pattern_size, flags)
    if not ok or corners is None:
        return False, np.zeros((0, 2), dtype=np.float64), img

    win = max(3, int(subpix_window) | 1)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        60,
        1e-4,
    )
    corners = cv2.cornerSubPix(img, corners, (win // 2, win // 2), (-1, -1), criteria)
    return True, corners.reshape(-1, 2).astype(np.float64), img


def _camera_order_value(name: str, fallback: int) -> int:
    if "_" in name:
        tail = name.rsplit("_", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return fallback


def _continuous_visible_arc(indices: List[int], n_cameras: int) -> List[int]:
    if not indices:
        return []
    ordered = sorted(indices)
    if len(ordered) <= 2:
        return ordered

    gaps = []
    for i, idx in enumerate(ordered):
        nxt = ordered[(i + 1) % len(ordered)]
        gap = (nxt - idx) % n_cameras
        gaps.append(gap)

    break_after = int(np.argmax(gaps))
    return ordered[break_after + 1 :] + ordered[: break_after + 1]


def select_camera_pair(
    visible_indices: List[int],
    cam_names: List[str],
    centers: np.ndarray,
    strategy: str,
) -> tuple[int, int]:
    if len(visible_indices) < 2:
        raise ValueError("Need at least two chessboard-visible cameras.")

    if strategy == "middle":
        order_numbers = [_camera_order_value(cam_names[i], i) for i in range(len(cam_names))]
        n_cameras = max(order_numbers) + 1 if order_numbers else len(cam_names)
        visible_order = [_camera_order_value(cam_names[i], i) for i in visible_indices]
        by_order = dict(zip(visible_order, visible_indices))
        arc = _continuous_visible_arc(visible_order, n_cameras)
        arc_indices = [by_order[i] for i in arc]
        mid = len(arc_indices) // 2
        if len(arc_indices) % 2 == 0:
            pair = (arc_indices[mid - 1], arc_indices[mid])
        else:
            left = max(0, mid - 1)
            right = min(len(arc_indices) - 1, mid + 1)
            pair = (arc_indices[left], arc_indices[right])
        return tuple(sorted(pair))

    if strategy == "max_baseline":
        best_pair = None
        best_dist = -np.inf
        for a_pos, i in enumerate(visible_indices):
            for j in visible_indices[a_pos + 1 :]:
                dist = float(np.linalg.norm(centers[i] - centers[j]))
                if dist > best_dist:
                    best_pair = (i, j)
                    best_dist = dist
        if best_pair is None:
            raise ValueError("Could not select a visible camera pair.")
        return best_pair

    raise ValueError(f"Unknown pair_selection={strategy!r}; expected 'middle' or 'max_baseline'.")


def undistort_to_pixel(
    uv: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    if np.all(np.abs(dist) < 1e-12):
        return uv.astype(np.float64)
    undist = cv2.undistortPoints(
        uv.reshape(-1, 1, 2).astype(np.float64),
        K.astype(np.float64),
        dist.astype(np.float64),
        P=K.astype(np.float64),
    )
    return undist.reshape(-1, 2)


def triangulate_points(
    uv_a: np.ndarray,
    uv_b: np.ndarray,
    K_a: np.ndarray,
    K_b: np.ndarray,
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    R_a: np.ndarray,
    R_b: np.ndarray,
    t_a: np.ndarray,
    t_b: np.ndarray,
) -> np.ndarray:
    uv_a = undistort_to_pixel(uv_a, K_a, dist_a)
    uv_b = undistort_to_pixel(uv_b, K_b, dist_b)
    P_a = K_a @ np.concatenate([R_a, t_a.reshape(3, 1)], axis=1)
    P_b = K_b @ np.concatenate([R_b, t_b.reshape(3, 1)], axis=1)
    homog = cv2.triangulatePoints(P_a, P_b, uv_a.T, uv_b.T)
    xyz = (homog[:3] / homog[3:4]).T
    return xyz.astype(np.float64)


def project_points(points: np.ndarray, K: np.ndarray, dist: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    p_cam = R @ points.T + t.reshape(3, 1)
    x = p_cam[0] / p_cam[2]
    y = p_cam[1] / p_cam[2]
    k1, k2 = float(dist[0]), float(dist[1])
    if abs(k1) > 1e-12 or abs(k2) > 1e-12:
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        x = x * radial
        y = y * radial
    return np.stack([K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]], axis=1)


def grid_edge_lengths(points: np.ndarray, inner_rows: int, inner_cols: int) -> tuple[np.ndarray, np.ndarray]:
    grid = points.reshape(inner_rows, inner_cols, 3)
    horizontal = np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=2).reshape(-1)
    vertical = np.linalg.norm(grid[1:, :, :] - grid[:-1, :, :], axis=2).reshape(-1)
    return horizontal, vertical


def robust_edge_size(edge_lengths: np.ndarray, scale_stat: str, trim_fraction: float) -> float:
    edge_lengths = np.asarray(edge_lengths, dtype=np.float64)
    edge_lengths = edge_lengths[np.isfinite(edge_lengths)]
    if len(edge_lengths) == 0:
        raise RuntimeError("Cannot compute chessboard scale from no valid edges.")

    stat = str(scale_stat).lower()
    if stat == "mean":
        return float(np.mean(edge_lengths))
    if stat == "median":
        return float(np.median(edge_lengths))
    if stat == "p25":
        return float(np.percentile(edge_lengths, 25))
    if stat == "trimmed_mean":
        trim = min(max(float(trim_fraction), 0.0), 0.45)
        ordered = np.sort(edge_lengths)
        lo = int(np.floor(len(ordered) * trim))
        hi = int(np.ceil(len(ordered) * (1.0 - trim)))
        trimmed = ordered[lo:hi]
        if len(trimmed) == 0:
            trimmed = ordered
        return float(np.mean(trimmed))
    raise ValueError(f"Unknown scale_stat={scale_stat!r}; expected mean, median, trimmed_mean, or p25.")


def edge_length_stats(edge_lengths: np.ndarray, scale_stat: str, trim_fraction: float) -> dict:
    edge_lengths = np.asarray(edge_lengths, dtype=np.float64)
    selected = robust_edge_size(edge_lengths, scale_stat, trim_fraction)
    mean = float(np.mean(edge_lengths))
    median = float(np.median(edge_lengths))
    std = float(np.std(edge_lengths))
    return {
        "count": int(len(edge_lengths)),
        "selected": selected,
        "mean": mean,
        "median": median,
        "std": std,
        "min": float(np.min(edge_lengths)),
        "p05": float(np.percentile(edge_lengths, 5)),
        "p25": float(np.percentile(edge_lengths, 25)),
        "p75": float(np.percentile(edge_lengths, 75)),
        "p90": float(np.percentile(edge_lengths, 90)),
        "p95": float(np.percentile(edge_lengths, 95)),
        "max": float(np.max(edge_lengths)),
        "cv_mean": float(std / max(abs(mean), 1e-12)),
        "cv_selected": float(std / max(abs(selected), 1e-12)),
    }


def corner_order_candidates(corners: np.ndarray, inner_rows: int, inner_cols: int) -> List[tuple[str, np.ndarray]]:
    """Return plausible chessboard ordering variants for multi-view matching."""
    grid = corners.reshape(inner_rows, inner_cols, 2)
    return [
        ("as_detected", grid.reshape(-1, 2)),
        ("reverse_all", grid[::-1, ::-1].reshape(-1, 2)),
        ("flip_rows", grid[::-1, :].reshape(-1, 2)),
        ("flip_cols", grid[:, ::-1].reshape(-1, 2)),
    ]


def triangulate_with_best_corner_order(
    uv_a: np.ndarray,
    uv_b: np.ndarray,
    cfg: ChessboardScaleConfig,
    K_a: np.ndarray,
    K_b: np.ndarray,
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    R_a: np.ndarray,
    R_b: np.ndarray,
    t_a: np.ndarray,
    t_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    """Pick the second-view corner ordering with reprojection and grid-shape checks."""
    best = None
    for name, candidate_b in corner_order_candidates(uv_b, cfg.inner_rows, cfg.inner_cols):
        points = triangulate_points(uv_a, candidate_b, K_a, K_b, dist_a, dist_b, R_a, R_b, t_a, t_b)
        reproj_a = project_points(points, K_a, dist_a, R_a, t_a)
        reproj_b = project_points(points, K_b, dist_b, R_b, t_b)
        err = 0.5 * (
            np.linalg.norm(reproj_a - uv_a, axis=1)
            + np.linalg.norm(reproj_b - candidate_b, axis=1)
        )
        horizontal, vertical = grid_edge_lengths(points, cfg.inner_rows, cfg.inner_cols)
        candidate_edges = np.concatenate([horizontal, vertical])
        edge_cv = float(np.std(candidate_edges) / max(abs(np.mean(candidate_edges)), 1e-12))
        score = float(np.median(err)) + float(cfg.corner_order_edge_cv_weight) * edge_cv
        if best is None or score < best[0]:
            best = (score, points, candidate_b, name, err)

    if best is None:
        raise RuntimeError("Failed to evaluate chessboard corner order candidates.")
    _, points, aligned_b, order_name, mean_err = best
    return points, aligned_b, order_name, mean_err


def save_detection_overlay(path: Path, image: np.ndarray, pattern_size: tuple[int, int], ok: bool, corners: np.ndarray) -> None:
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    draw_corners = corners.reshape(-1, 1, 2).astype(np.float32) if len(corners) else None
    cv2.drawChessboardCorners(vis, pattern_size, draw_corners, ok)
    cv2.imwrite(str(path), vis)


def evaluate_camera_pair(
    cam_a: int,
    cam_b: int,
    detections: Dict[int, np.ndarray],
    cam_names: list[str],
    centers: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    cfg: ChessboardScaleConfig,
) -> dict:
    uv_a = detections[cam_a]
    uv_b = detections[cam_b]
    if len(uv_a) < cfg.min_common_corners or len(uv_b) < cfg.min_common_corners:
        raise RuntimeError(
            f"Pair {cam_names[cam_a]}-{cam_names[cam_b]} has too few common corners: "
            f"{len(uv_a)} and {len(uv_b)}; min_common_corners={cfg.min_common_corners}"
        )

    points, uv_b_aligned, corner_order_b, _ = triangulate_with_best_corner_order(
        uv_a,
        uv_b,
        cfg,
        K[cam_a],
        K[cam_b],
        dist[cam_a],
        dist[cam_b],
        R[cam_a],
        R[cam_b],
        t[cam_a],
        t[cam_b],
    )

    reproj_a = project_points(points, K[cam_a], dist[cam_a], R[cam_a], t[cam_a])
    reproj_b = project_points(points, K[cam_b], dist[cam_b], R[cam_b], t[cam_b])
    err_a = np.linalg.norm(reproj_a - uv_a, axis=1)
    err_b = np.linalg.norm(reproj_b - uv_b_aligned, axis=1)
    mean_reproj = 0.5 * (err_a + err_b)
    valid = mean_reproj <= float(cfg.max_reprojection_error_px)
    if np.count_nonzero(valid) < cfg.min_common_corners:
        raise RuntimeError(
            f"Pair {cam_names[cam_a]}-{cam_names[cam_b]} only has "
            f"{np.count_nonzero(valid)} valid corners after reprojection filtering."
        )

    horizontal, vertical = grid_edge_lengths(points, cfg.inner_rows, cfg.inner_cols)
    valid_grid = valid.reshape(cfg.inner_rows, cfg.inner_cols)
    valid_h = (valid_grid[:, 1:] & valid_grid[:, :-1]).reshape(-1)
    valid_v = (valid_grid[1:, :] & valid_grid[:-1, :]).reshape(-1)
    edge_lengths = np.concatenate([horizontal[valid_h], vertical[valid_v]])
    if len(edge_lengths) == 0:
        raise RuntimeError(f"Pair {cam_names[cam_a]}-{cam_names[cam_b]} has no valid chessboard edges.")

    edge_stats = edge_length_stats(edge_lengths, cfg.scale_stat, cfg.trim_fraction)
    sfm_square_size = float(edge_stats["selected"])
    sfm_to_world_scale = float(cfg.square_size / sfm_square_size)
    edge_cv = float(edge_stats["cv_mean"])
    baseline = float(np.linalg.norm(centers[cam_a] - centers[cam_b]))
    quality_score = edge_cv + 0.01 * float(np.median(mean_reproj)) + 1.0 / max(baseline, 1e-12)
    return {
        "camera_ids": [int(cam_a), int(cam_b)],
        "camera_names": [cam_names[cam_a], cam_names[cam_b]],
        "corner_order_second_view": corner_order_b,
        "baseline_sfm": baseline,
        "points": points,
        "corners_uv_a": uv_a,
        "corners_uv_b": uv_b_aligned,
        "valid_corners": valid,
        "edge_lengths_sfm": edge_lengths,
        "scale": sfm_to_world_scale,
        "sfm_square_size": sfm_square_size,
        "edge_stats": edge_stats,
        "quality": {
            "num_triangulated_corners": int(len(points)),
            "num_valid_corners": int(np.count_nonzero(valid)),
            "num_valid_edges": int(len(edge_lengths)),
            "reprojection_error_px_mean": float(np.mean(mean_reproj)),
            "reprojection_error_px_median": float(np.median(mean_reproj)),
            "reprojection_error_px_max": float(np.max(mean_reproj)),
            "edge_cv": edge_cv,
            "quality_score": float(quality_score),
        },
    }


def pair_record_for_json(pair: dict, *, reliable: bool) -> dict:
    return {
        "camera_ids": pair["camera_ids"],
        "camera_names": pair["camera_names"],
        "corner_order_second_view": pair["corner_order_second_view"],
        "baseline_sfm": pair["baseline_sfm"],
        "sfm_square_size": pair["sfm_square_size"],
        "sfm_to_world_scale": pair["scale"],
        "edge_stats": pair["edge_stats"],
        "quality": pair["quality"],
        "reliable": bool(reliable),
    }


def run_chessboard_scale(config: ChessboardScaleConfig | None = None) -> Dict[str, str]:
    cfg = config or ChessboardScaleConfig()
    data_dir = Path(cfg.data_dir)
    sfm_dir = Path(cfg.sfm_dir) if cfg.sfm_dir else data_dir / "result" / "sfm"
    image_root = data_dir / cfg.image_dir
    output_dir = Path(cfg.output_dir) if cfg.output_dir else data_dir / "result" / "sfm2world"
    overlay_dir = output_dir / "detections"
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    cameras = load_sfm_cameras(sfm_dir)
    cam_names = [str(x) for x in cameras["cam_names"]]
    K = np.asarray(cameras["K"], dtype=np.float64)
    dist = np.asarray(cameras["dist"], dtype=np.float64)
    R = np.asarray(cameras["R"], dtype=np.float64)
    t = np.asarray(cameras["t"], dtype=np.float64).reshape(len(cam_names), 3)
    centers = np.asarray(cameras["camera_centers_world"], dtype=np.float64)

    pattern_size = (int(cfg.inner_cols), int(cfg.inner_rows))
    n_corners = cfg.inner_cols * cfg.inner_rows

    detections: Dict[int, np.ndarray] = {}
    image_paths: Dict[int, str] = {}
    detection_records = []

    print("[sfm2world] Detecting chessboard corners")
    for cam_id, cam_name in enumerate(cam_names):
        cam_dir = image_root / cam_name
        img_path = find_named_image(cam_dir, cfg.image_name) if cfg.image_name else find_single_image(cam_dir)
        ok, corners, image = detect_chessboard_corners(img_path, pattern_size, cfg.subpix_window)
        if ok and len(corners) == n_corners:
            detections[cam_id] = corners
        image_paths[cam_id] = str(img_path)
        detection_records.append(
            {
                "camera_id": cam_id,
                "camera_name": cam_name,
                "image_path": str(img_path),
                "detected": bool(ok and len(corners) == n_corners),
                "num_corners": int(len(corners)),
            }
        )
        if cfg.save_overlays:
            save_detection_overlay(overlay_dir / f"{cam_name}_corners.png", image, pattern_size, bool(ok), corners)
        print(f"  {cam_name}: detected={ok}, corners={len(corners)}")

    visible = sorted(detections.keys())
    if len(visible) < 2:
        raise RuntimeError(f"Need at least two visible cameras, got {len(visible)}.")

    pair_errors = []
    if cfg.pair_selection == "all_pairs":
        candidate_pairs = [(a, b) for pos, a in enumerate(visible) for b in visible[pos + 1 :]]
    else:
        candidate_pairs = [select_camera_pair(visible, cam_names, centers, cfg.pair_selection)]

    pair_results = []
    for cam_a, cam_b in candidate_pairs:
        try:
            pair_results.append(evaluate_camera_pair(cam_a, cam_b, detections, cam_names, centers, K, dist, R, t, cfg))
        except RuntimeError as exc:
            pair_errors.append(
                {
                    "camera_ids": [int(cam_a), int(cam_b)],
                    "camera_names": [cam_names[cam_a], cam_names[cam_b]],
                    "error": str(exc),
                }
            )

    if not pair_results:
        raise RuntimeError(f"No valid chessboard camera pair. Pair errors: {pair_errors}")

    reliable_pairs = [
        pair
        for pair in pair_results
        if pair["quality"]["edge_cv"] <= float(cfg.max_pair_edge_cv)
        and pair["quality"]["num_valid_edges"] >= int(cfg.min_common_corners)
    ]
    if not reliable_pairs:
        reliable_pairs = sorted(pair_results, key=lambda p: p["quality"]["quality_score"])[:1]
        print(
            "[sfm2world] WARNING: no pair passed max_pair_edge_cv; "
            f"falling back to best quality_score pair ({reliable_pairs[0]['camera_names']})."
        )

    selected_pair = min(reliable_pairs, key=lambda p: p["quality"]["quality_score"])
    if cfg.pair_selection == "all_pairs":
        sfm_to_world_scale = float(np.median([pair["scale"] for pair in reliable_pairs]))
        sfm_square_size = float(cfg.square_size / sfm_to_world_scale)
    else:
        sfm_to_world_scale = float(selected_pair["scale"])
        sfm_square_size = float(selected_pair["sfm_square_size"])

    world_to_sfm_scale = float(sfm_square_size / cfg.square_size)
    edge_lengths = selected_pair["edge_lengths_sfm"]
    edge_stats = selected_pair["edge_stats"]
    sfm_square_mean = float(edge_stats["mean"])
    sfm_square_median = float(edge_stats["median"])
    sfm_square_std = float(edge_stats["std"])
    points = selected_pair["points"]
    uv_a = selected_pair["corners_uv_a"]
    uv_b_aligned = selected_pair["corners_uv_b"]
    valid = selected_pair["valid_corners"]
    cam_a, cam_b = selected_pair["camera_ids"]
    corner_order_b = selected_pair["corner_order_second_view"]
    pair_records = [
        pair_record_for_json(pair, reliable=pair in reliable_pairs)
        for pair in sorted(pair_results, key=lambda p: p["quality"]["quality_score"])
    ]
    unreliable_edge_cv = selected_pair["quality"]["edge_cv"] > float(cfg.max_edge_cv_warning)
    warnings = []
    if unreliable_edge_cv:
        warnings.append(
            "Selected chessboard pair has high edge_cv; scale may be unreliable. "
            f"edge_cv={selected_pair['quality']['edge_cv']:.6g}, "
            f"warning_threshold={cfg.max_edge_cv_warning:.6g}."
        )

    result = {
        "config": asdict(cfg),
        "sfm_dir": str(sfm_dir),
        "image_root": str(image_root),
        "visible_cameras": [cam_names[i] for i in visible],
        "selected_pair": {
            "camera_ids": [int(cam_a), int(cam_b)],
            "camera_names": [cam_names[cam_a], cam_names[cam_b]],
            "pair_selection": cfg.pair_selection,
            "corner_order_second_view": corner_order_b,
            "baseline_sfm": float(np.linalg.norm(centers[cam_a] - centers[cam_b])),
        },
        "scale": {
            "scale_stat": str(cfg.scale_stat),
            "trim_fraction": float(cfg.trim_fraction),
            "sfm_square_size_mean": sfm_square_mean,
            "sfm_square_size_selected": sfm_square_size,
            "selected_pair_sfm_square_size_mean": sfm_square_mean,
            "sfm_square_size_median": sfm_square_median,
            "sfm_square_size_std": sfm_square_std,
            "physical_square_size": float(cfg.square_size),
            "sfm_to_world_scale": sfm_to_world_scale,
            "world_to_sfm_scale": world_to_sfm_scale,
            "unit": "physical_units_per_sfm_unit",
        },
        "quality": {
            "num_detected_cameras": int(len(visible)),
            "num_triangulated_corners": int(len(points)),
            "num_valid_corners": int(np.count_nonzero(valid)),
            "num_valid_edges": int(len(edge_lengths)),
            "reprojection_error_px_mean": selected_pair["quality"]["reprojection_error_px_mean"],
            "reprojection_error_px_median": selected_pair["quality"]["reprojection_error_px_median"],
            "reprojection_error_px_max": selected_pair["quality"]["reprojection_error_px_max"],
            "edge_cv": selected_pair["quality"]["edge_cv"],
            "selected_pair_quality_score": selected_pair["quality"]["quality_score"],
            "num_candidate_pairs": int(len(pair_results)),
            "num_reliable_pairs": int(len(reliable_pairs)),
            "max_pair_edge_cv": float(cfg.max_pair_edge_cv),
        },
        "pair_scale_records": pair_records,
        "pair_errors": pair_errors,
        "warnings": warnings,
        "detections": detection_records,
    }

    npz_path = output_dir / "chessboard_triangulation.npz"
    json_path = output_dir / "sfm2world_scale.json"
    np.savez_compressed(
        npz_path,
        points_sfm=points,
        corners_uv_a=uv_a,
        corners_uv_b=uv_b_aligned,
        valid_corners=valid,
        edge_lengths_sfm=edge_lengths,
        selected_camera_ids=np.asarray([cam_a, cam_b], dtype=np.int64),
        selected_camera_names=np.asarray([cam_names[cam_a], cam_names[cam_b]]),
        visible_camera_ids=np.asarray(visible, dtype=np.int64),
        visible_camera_names=np.asarray([cam_names[i] for i in visible]),
        reliable_pair_scales=np.asarray([pair["scale"] for pair in reliable_pairs], dtype=np.float64),
        candidate_pair_scales=np.asarray([pair["scale"] for pair in pair_results], dtype=np.float64),
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("[sfm2world] Selected pair:", cam_names[cam_a], cam_names[cam_b])
    print(f"[sfm2world] Scale statistic: {cfg.scale_stat}")
    print(f"[sfm2world] Square size in SfM units: {sfm_square_size:.8g}")
    print(f"[sfm2world] Reliable pairs: {len(reliable_pairs)}/{len(pair_results)}")
    if warnings:
        for warning in warnings:
            print(f"[sfm2world] WARNING: {warning}")
    print(f"[sfm2world] Physical square size: {cfg.square_size:.8g}")
    print(f"[sfm2world] sfm_to_world_scale: {sfm_to_world_scale:.8g}")
    print(f"[sfm2world] Output: {json_path}")

    outputs = {
        "scale_json": str(json_path),
        "triangulation_npz": str(npz_path),
    }
    if cfg.save_overlays:
        outputs["detection_dir"] = str(overlay_dir)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate SfM-to-world scale from calibration chessboard images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default="case/CylinderDIC")
    parser.add_argument("--sfm_dir", type=str, default="")
    parser.add_argument("--image_dir", type=str, default="calibrate_images")
    parser.add_argument("--image_name", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--inner_cols", type=int, required=True, help="Number of inner corners along board width.")
    parser.add_argument("--inner_rows", type=int, required=True, help="Number of inner corners along board height.")
    parser.add_argument("--square_size", type=float, required=True, help="Physical chessboard square size.")
    parser.add_argument("--pair_selection", choices=["middle", "max_baseline", "all_pairs"], default="middle")
    parser.add_argument("--scale_stat", choices=["mean", "median", "trimmed_mean", "p25"], default="trimmed_mean")
    parser.add_argument("--trim_fraction", type=float, default=0.20)
    parser.add_argument("--max_pair_edge_cv", type=float, default=0.08)
    parser.add_argument("--max_edge_cv_warning", type=float, default=0.05)
    parser.add_argument("--corner_order_edge_cv_weight", type=float, default=1.0)
    parser.add_argument("--subpix_window", type=int, default=11)
    parser.add_argument("--min_common_corners", type=int, default=12)
    parser.add_argument("--max_reprojection_error_px", type=float, default=3.0)
    parser.add_argument("--no_overlays", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ChessboardScaleConfig(
        data_dir=args.data_dir,
        sfm_dir=args.sfm_dir or None,
        image_dir=args.image_dir,
        image_name=args.image_name or None,
        output_dir=args.output_dir or None,
        inner_cols=args.inner_cols,
        inner_rows=args.inner_rows,
        square_size=args.square_size,
        pair_selection=args.pair_selection,
        scale_stat=args.scale_stat,
        trim_fraction=args.trim_fraction,
        max_pair_edge_cv=args.max_pair_edge_cv,
        max_edge_cv_warning=args.max_edge_cv_warning,
        corner_order_edge_cv_weight=args.corner_order_edge_cv_weight,
        subpix_window=args.subpix_window,
        min_common_corners=args.min_common_corners,
        max_reprojection_error_px=args.max_reprojection_error_px,
        save_overlays=not args.no_overlays,
    )
    run_chessboard_scale(cfg)


if __name__ == "__main__":
    main()
