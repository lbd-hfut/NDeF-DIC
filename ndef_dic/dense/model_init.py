"""SfM-guided neural depth initialisation for the dense module.

This module does one deliberately limited job: learn a continuous,
camera-conditioned Z-depth field from sparse SfM observations, then evaluate it
inside ROI masks supplied by ``roi_builder.py``.  The result is an
initialisation and diagnostic product for the later dense DIC stage, not a
standalone dense reconstruction claim.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class DepthInitConfig:
    """Configuration for SfM sparse-depth neural initialisation."""

    data_dir: str = "case/CylinderDIC"
    sfm_dir: str | None = None
    output_dir: str | None = None
    hidden_dim: int = 32
    pixel_layers: int = 3
    camera_layers: int = 2
    trunk_layers: int = 3
    camera_embedding_dim: int = 16
    positional_encoding_enabled: bool = False
    positional_encoding_num_frequencies: int = 4
    epochs: int = 3000
    lr: float = 1e-3
    weight_decay: float = 1e-6
    smooth_weight: float = 1e-4
    smooth_samples_per_camera: int = 256
    log_interval: int = 250
    prediction_batch_size: int = 262144
    point_plot_stride: int = 20
    roi_dir: str | None = None
    use_external_roi: bool = False
    external_roi_dir: str | None = None
    sparse_filter_enabled: bool = True
    sparse_filter_min_track_length: int = 2
    sparse_filter_max_reproj_error: float | None = None
    sparse_filter_radius_mad_thresh: float = 8.0
    sparse_filter_knn_k: int = 8
    sparse_filter_knn_mad_thresh: float = 8.0
    seed: int = 7
    device: str = "auto"


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        modules: List[nn.Module] = []
        last_dim = in_dim
        for _ in range(layers):
            modules.append(nn.Linear(last_dim, hidden_dim))
            modules.append(nn.Tanh())
            last_dim = hidden_dim
        modules.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierPixelEncoding(nn.Module):
    """Pure Fourier encoding for normalized pixel coordinates."""

    def __init__(self, num_frequencies: int):
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be >= 1")
        frequencies = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)
        self.output_dim = 4 * num_frequencies

    def forward(self, pixel_xy_norm: torch.Tensor) -> torch.Tensor:
        angles = math.pi * pixel_xy_norm[..., None, :] * self.frequencies[:, None]
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return encoded.flatten(start_dim=-2)


class SfMDepthFiLMNet(nn.Module):
    """Camera-conditioned sparse-depth interpolator.

    Pixel coordinates are shared across cameras.  Camera identity is encoded by
    a learnable embedding that produces FiLM modulation parameters, avoiding a
    false ordinal meaning for camera ids.
    """

    def __init__(
        self,
        n_cameras: int,
        hidden_dim: int = 32,
        camera_embedding_dim: int = 16,
        pixel_layers: int = 3,
        camera_layers: int = 2,
        trunk_layers: int = 3,
        positional_encoding_enabled: bool = False,
        positional_encoding_num_frequencies: int = 4,
    ):
        super().__init__()
        self.pixel_encoding = (
            FourierPixelEncoding(positional_encoding_num_frequencies)
            if positional_encoding_enabled
            else nn.Identity()
        )
        pixel_input_dim = self.pixel_encoding.output_dim if positional_encoding_enabled else 2
        self.pixel_head = MLP(pixel_input_dim, hidden_dim, hidden_dim, pixel_layers)
        self.camera_embedding = nn.Embedding(n_cameras, camera_embedding_dim)
        self.camera_head = MLP(camera_embedding_dim, hidden_dim, hidden_dim * 2, camera_layers)
        self.depth_head = MLP(hidden_dim, hidden_dim, 1, trunk_layers)

    def forward(self, pixel_xy_norm: torch.Tensor, cam_indices: torch.Tensor) -> torch.Tensor:
        pixel_features = self.pixel_head(self.pixel_encoding(pixel_xy_norm))
        camera_features = self.camera_head(self.camera_embedding(cam_indices.long()))
        gamma, beta = camera_features.chunk(2, dim=-1)
        fused = (1.0 + 0.1 * gamma) * pixel_features + beta
        return self.depth_head(fused).squeeze(-1)


def run_model_init(config: DepthInitConfig | None = None) -> Dict[str, str]:
    cfg = config or DepthInitConfig()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    sfm_dir = Path(cfg.sfm_dir) if cfg.sfm_dir else data_dir / "result" / "sfm"
    output_dir = Path(cfg.output_dir) if cfg.output_dir else data_dir / "result" / "dense" / "model_init"
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = _load_npz(sfm_dir / "cameras.npz")
    sparse = _load_npz(sfm_dir / "sparse_points.npz")
    obs = _load_npz(sfm_dir / "observations.npz")
    sparse, obs, sparse_filter_meta = _filter_sparse_products(sparse, obs, output_dir, cfg)

    cam_names = [str(x) for x in cameras["cam_names"]]
    K = cameras["K"].astype(np.float64)
    dist = cameras["dist"].astype(np.float64)
    R = cameras["R"].astype(np.float64)
    t = cameras["t"].astype(np.float64).reshape(len(cam_names), 3)
    image_sizes = _load_image_sizes(cameras["image_paths"])

    train_xy = _normalise_pixels(obs["uv"].astype(np.float64), obs["cam_indices"], image_sizes)
    train_cam = obs["cam_indices"].astype(np.int64)
    train_depth = obs["depth"].astype(np.float64)
    depth_mean = float(train_depth.mean())
    depth_std = float(train_depth.std() if train_depth.std() > 1e-12 else 1.0)
    train_target = ((train_depth - depth_mean) / depth_std).astype(np.float32)

    device = _select_device(cfg.device)
    model = SfMDepthFiLMNet(
        n_cameras=len(cam_names),
        hidden_dim=cfg.hidden_dim,
        camera_embedding_dim=cfg.camera_embedding_dim,
        pixel_layers=cfg.pixel_layers,
        camera_layers=cfg.camera_layers,
        trunk_layers=cfg.trunk_layers,
        positional_encoding_enabled=cfg.positional_encoding_enabled,
        positional_encoding_num_frequencies=cfg.positional_encoding_num_frequencies,
    ).to(device)

    history = _train_model(
        model=model,
        xy_norm=train_xy,
        cam_indices=train_cam,
        target=train_target,
        image_sizes=image_sizes,
        cfg=cfg,
        device=device,
    )

    roi_products = _prepare_roi_masks(
        data_dir=data_dir,
        sfm_dir=sfm_dir,
        cam_names=cam_names,
        cfg=cfg,
    )

    dense_products = _predict_inside_roi_masks(
        model=model,
        observations=obs,
        image_sizes=image_sizes,
        roi_products=roi_products,
        depth_mean=depth_mean,
        depth_std=depth_std,
        K=K,
        dist=dist,
        R=R,
        t=t,
        cfg=cfg,
        device=device,
    )

    diagnostics = _evaluate_sparse_errors(
        model=model,
        xy_norm=train_xy,
        cam_indices=train_cam,
        sparse_depth=train_depth,
        depth_mean=depth_mean,
        depth_std=depth_std,
        device=device,
    )

    _save_dense_products(output_dir, cam_names, dense_products)
    np.savez_compressed(
        output_dir / "dense_model_init.npz",
        cam_names=np.asarray(cam_names),
        depth_mean=np.asarray(depth_mean),
        depth_std=np.asarray(depth_std),
        train_loss=np.asarray(history["train_loss"]),
        sparse_error=diagnostics["error"],
        sparse_pred_depth=diagnostics["pred_depth"],
    )
    _save_depth_normalization(output_dir, depth_mean, depth_std)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "depth_mean": depth_mean,
            "depth_std": depth_std,
            "cam_names": cam_names,
            "image_sizes": image_sizes,
        },
        output_dir / "depth_film_init.pt",
    )
    _save_meta(output_dir, cfg, cam_names, image_sizes, history, depth_mean, depth_std, diagnostics, sparse_filter_meta)

    fig_paths = _save_visualisations(
        output_dir=output_dir,
        cam_names=cam_names,
        sparse_points=sparse["points3D"].astype(np.float64),
        dense_products=dense_products,
        observations=obs,
        sparse_errors=diagnostics["error"],
        cfg=cfg,
    )
    return {name: str(path) for name, path in fig_paths.items()}


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _filter_sparse_products(
    sparse: Dict[str, np.ndarray],
    observations: Dict[str, np.ndarray],
    output_dir: Path,
    cfg: DepthInitConfig,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    points = sparse["points3D"].astype(np.float64)
    point_ids = sparse["point_ids"].astype(np.int64)
    n_points = len(points)
    keep = np.ones(n_points, dtype=bool)
    reasons: Dict[str, int] = {}

    if cfg.sparse_filter_enabled and n_points:
        finite = np.isfinite(points).all(axis=1)
        keep &= finite
        reasons["nonfinite"] = int((~finite).sum())

        if "track_lengths" in sparse and cfg.sparse_filter_min_track_length > 1:
            track_ok = sparse["track_lengths"].astype(np.int64) >= int(cfg.sparse_filter_min_track_length)
            keep &= track_ok
            reasons["short_track"] = int((~track_ok).sum())

        if cfg.sparse_filter_max_reproj_error is not None and "reproj_error" in sparse:
            reproj_ok = sparse["reproj_error"].astype(np.float64) <= float(cfg.sparse_filter_max_reproj_error)
            keep &= reproj_ok
            reasons["high_reprojection_error"] = int((~reproj_ok).sum())

        radius_keep = _robust_radius_mask(points, float(cfg.sparse_filter_radius_mad_thresh))
        keep &= radius_keep
        reasons["robust_radius"] = int((~radius_keep).sum())

        knn_keep = _knn_density_mask(
            points,
            k=int(cfg.sparse_filter_knn_k),
            mad_thresh=float(cfg.sparse_filter_knn_mad_thresh),
        )
        keep &= knn_keep
        reasons["knn_density"] = int((~knn_keep).sum())

    filtered_sparse, old_to_new = _apply_sparse_point_mask(sparse, keep)
    filtered_obs = _apply_observation_point_mask(observations, old_to_new)
    filter_dir = output_dir / "sparse_filter"
    filter_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filter_dir / "sparse_points_filtered.npz", **filtered_sparse)
    np.savez_compressed(filter_dir / "observations_filtered.npz", **filtered_obs)

    meta: Dict[str, object] = {
        "enabled": bool(cfg.sparse_filter_enabled),
        "n_points_before": int(n_points),
        "n_points_after": int(len(filtered_sparse["points3D"])),
        "n_points_removed": int(n_points - len(filtered_sparse["points3D"])),
        "n_observations_before": int(len(observations["cam_indices"])),
        "n_observations_after": int(len(filtered_obs["cam_indices"])),
        "n_observations_removed": int(len(observations["cam_indices"]) - len(filtered_obs["cam_indices"])),
        "criteria": {
            "min_track_length": int(cfg.sparse_filter_min_track_length),
            "max_reproj_error": cfg.sparse_filter_max_reproj_error,
            "radius_mad_thresh": float(cfg.sparse_filter_radius_mad_thresh),
            "knn_k": int(cfg.sparse_filter_knn_k),
            "knn_mad_thresh": float(cfg.sparse_filter_knn_mad_thresh),
        },
        "raw_rejection_counts": reasons,
        "filtered_sparse_points": str(filter_dir / "sparse_points_filtered.npz"),
        "filtered_observations": str(filter_dir / "observations_filtered.npz"),
    }
    with open(filter_dir / "sparse_filter_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if cfg.sparse_filter_enabled:
        print(
            f"[SparseFilter] kept {meta['n_points_after']}/{meta['n_points_before']} "
            f"points and {meta['n_observations_after']}/{meta['n_observations_before']} observations"
        )
    else:
        print("[SparseFilter] disabled; using raw SfM sparse points")
    return filtered_sparse, filtered_obs, meta


def _robust_radius_mask(points: np.ndarray, mad_thresh: float) -> np.ndarray:
    if len(points) < 8 or mad_thresh <= 0:
        return np.ones(len(points), dtype=bool)
    centre = np.median(points, axis=0)
    radius = np.linalg.norm(points - centre, axis=1)
    median = float(np.median(radius))
    mad = float(np.median(np.abs(radius - median)))
    if mad <= 1e-12:
        return np.ones(len(points), dtype=bool)
    sigma = 1.4826 * mad
    return radius <= median + mad_thresh * sigma


def _knn_density_mask(points: np.ndarray, k: int, mad_thresh: float) -> np.ndarray:
    if len(points) < max(8, k + 2) or k <= 0 or mad_thresh <= 0:
        return np.ones(len(points), dtype=bool)
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return np.ones(len(points), dtype=bool)
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(k + 1, len(points)))
    kth = distances[:, -1]
    median = float(np.median(kth))
    mad = float(np.median(np.abs(kth - median)))
    if mad <= 1e-12:
        return np.ones(len(points), dtype=bool)
    sigma = 1.4826 * mad
    return kth <= median + mad_thresh * sigma


def _apply_sparse_point_mask(
    sparse: Dict[str, np.ndarray],
    keep: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[int, int]]:
    old_indices = np.flatnonzero(keep)
    old_to_new = {int(old): int(new) for new, old in enumerate(old_indices)}
    out: Dict[str, np.ndarray] = {}
    for key, value in sparse.items():
        if isinstance(value, np.ndarray) and len(value) == len(keep):
            out[key] = value[keep]
        else:
            out[key] = value
    return out, old_to_new


def _apply_observation_point_mask(
    observations: Dict[str, np.ndarray],
    old_to_new: Dict[int, int],
) -> Dict[str, np.ndarray]:
    point_indices = observations["point_indices"].astype(np.int64)
    keep_obs = np.fromiter((int(idx) in old_to_new for idx in point_indices), dtype=bool, count=len(point_indices))
    out: Dict[str, np.ndarray] = {}
    for key, value in observations.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == keep_obs.shape:
            out[key] = value[keep_obs]
        else:
            out[key] = value
    out["point_indices"] = np.asarray([old_to_new[int(idx)] for idx in point_indices[keep_obs]], dtype=np.int64)
    return out


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[DenseInit] CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _load_image_sizes(image_paths: np.ndarray) -> np.ndarray:
    import cv2

    sizes = []
    for raw_path in image_paths:
        path = Path(str(raw_path))
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read camera image: {path}")
        height, width = image.shape[:2]
        sizes.append((width, height))
    return np.asarray(sizes, dtype=np.int64)


def _normalise_pixels(uv: np.ndarray, cam_indices: np.ndarray, image_sizes: np.ndarray) -> np.ndarray:
    sizes = image_sizes[np.asarray(cam_indices, dtype=np.int64)]
    x = 2.0 * uv[:, 0] / np.maximum(sizes[:, 0] - 1, 1) - 1.0
    y = 2.0 * uv[:, 1] / np.maximum(sizes[:, 1] - 1, 1) - 1.0
    return np.stack([x, y], axis=1).astype(np.float32)


def _train_model(
    model: SfMDepthFiLMNet,
    xy_norm: np.ndarray,
    cam_indices: np.ndarray,
    target: np.ndarray,
    image_sizes: np.ndarray,
    cfg: DepthInitConfig,
    device: torch.device,
) -> Dict[str, List[float]]:
    x = torch.as_tensor(xy_norm, dtype=torch.float32, device=device)
    cam = torch.as_tensor(cam_indices, dtype=torch.long, device=device)
    y = torch.as_tensor(target, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = {"train_loss": []}

    print(
        f"[DenseInit] Training {len(x)} sparse depth observations on {device} "
        f"for {cfg.epochs} epochs"
    )
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pred = model(x, cam)
        data_loss = torch.mean((pred - y) ** 2)
        smooth_loss = _smoothness_loss(model, image_sizes, cfg, device)
        loss = data_loss + cfg.smooth_weight * smooth_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history["train_loss"].append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % cfg.log_interval == 0 or epoch == cfg.epochs:
            rmse_norm = math.sqrt(float(data_loss.detach().cpu()))
            print(
                f"[DenseInit] epoch {epoch:5d}/{cfg.epochs} "
                f"loss={float(loss.detach().cpu()):.6e} rmse_norm={rmse_norm:.4f}"
            )
    return history


def _smoothness_loss(
    model: SfMDepthFiLMNet,
    image_sizes: np.ndarray,
    cfg: DepthInitConfig,
    device: torch.device,
) -> torch.Tensor:
    if cfg.smooth_weight <= 0 or cfg.smooth_samples_per_camera <= 0:
        return torch.zeros((), device=device)
    n_cameras = image_sizes.shape[0]
    xy = torch.rand(n_cameras * cfg.smooth_samples_per_camera, 2, device=device) * 2.0 - 1.0
    xy.requires_grad_(True)
    cam = torch.arange(n_cameras, device=device).repeat_interleave(cfg.smooth_samples_per_camera)
    pred = model(xy, cam)
    grad = torch.autograd.grad(pred.sum(), xy, create_graph=True)[0]
    return torch.mean(grad**2)


def _prepare_roi_masks(
    data_dir: Path,
    sfm_dir: Path,
    cam_names: List[str],
    cfg: DepthInitConfig,
) -> List[Dict[str, np.ndarray]]:
    from .roi_builder import ROIConfig, run_auto_roi

    roi_output_dir = Path(cfg.roi_dir) if cfg.roi_dir else data_dir / "result" / "dense" / "auto_roi"
    roi_masks = run_auto_roi(
        data_dir=str(data_dir),
        sfm_dir=str(sfm_dir),
        output_dir=str(roi_output_dir),
        config=ROIConfig(
            use_external=cfg.use_external_roi,
            external_roi_dir=cfg.external_roi_dir,
        ),
        verbose=True,
    )
    if len(roi_masks) != len(cam_names):
        raise ValueError(f"ROI camera count {len(roi_masks)} does not match SfM camera count {len(cam_names)}")

    products = []
    for cm, expected_name in zip(roi_masks, cam_names):
        if cm.cam_name != expected_name:
            raise ValueError(f"ROI camera order mismatch: got {cm.cam_name}, expected {expected_name}")
        products.append(
            {
                "cam_id": int(cm.cam_id),
                "cam_name": cm.cam_name,
                "mask": cm.mask.astype(bool),
                "contour": _mask_outer_contour(cm.mask),
                "roi_source": "external" if cfg.use_external_roi else "auto",
                "roi_dir": str(roi_output_dir),
            }
        )
    return products


def _mask_outer_contour(mask: np.ndarray) -> np.ndarray:
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return contour.astype(np.float32)


def _predict_inside_roi_masks(
    model: SfMDepthFiLMNet,
    observations: Dict[str, np.ndarray],
    image_sizes: np.ndarray,
    roi_products: List[Dict[str, np.ndarray]],
    depth_mean: float,
    depth_std: float,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    cfg: DepthInitConfig,
    device: torch.device,
) -> List[Dict[str, np.ndarray]]:
    products = []
    for cam_id, (width, height) in enumerate(image_sizes):
        uv = observations["uv"][observations["cam_indices"] == cam_id].astype(np.float64)
        if len(uv) < 3:
            raise ValueError(f"Camera {cam_id} has fewer than 3 SfM observations.")

        mask = roi_products[cam_id]["mask"]
        if mask.shape != (int(height), int(width)):
            raise ValueError(
                f"ROI mask shape for camera {cam_id} is {mask.shape}, "
                f"expected {(int(height), int(width))}"
            )
        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            raise ValueError(f"ROI mask for camera {cam_id} is empty.")
        pixels = np.stack([cols, rows], axis=1).astype(np.float64)

        xy_norm = _normalise_pixels(
            pixels,
            np.full(len(pixels), cam_id, dtype=np.int64),
            image_sizes,
        )
        pred_depth = _predict_depth_batches(
            model,
            xy_norm,
            np.full(len(pixels), cam_id, dtype=np.int64),
            depth_mean,
            depth_std,
            cfg.prediction_batch_size,
            device,
        )
        world = _backproject_pixels(pixels, pred_depth, K[cam_id], dist[cam_id], R[cam_id], t[cam_id])
        interp_depth = _interpolate_sparse_depth(pixels, uv, observations["depth"][observations["cam_indices"] == cam_id])

        products.append(
            {
                "cam_id": np.asarray(cam_id),
                "contour": roi_products[cam_id]["contour"],
                "mask": mask,
                "pixels": pixels.astype(np.float32),
                "pred_depth": pred_depth.astype(np.float32),
                "interp_depth": interp_depth.astype(np.float32),
                "world": world.astype(np.float32),
                "roi_source": np.asarray(roi_products[cam_id]["roi_source"]),
                "roi_dir": np.asarray(roi_products[cam_id]["roi_dir"]),
            }
        )
        print(
            f"[DenseInit] cam_{cam_id}: predicted {len(pixels)} ROI pixels "
            f"from {roi_products[cam_id]['roi_source']} ROI"
        )
    return products


def _predict_depth_batches(
    model: SfMDepthFiLMNet,
    xy_norm: np.ndarray,
    cam_indices: np.ndarray,
    depth_mean: float,
    depth_std: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(xy_norm), batch_size):
            stop = min(start + batch_size, len(xy_norm))
            x = torch.as_tensor(xy_norm[start:stop], dtype=torch.float32, device=device)
            cam = torch.as_tensor(cam_indices[start:stop], dtype=torch.long, device=device)
            pred = model(x, cam).detach().cpu().numpy()
            out.append(pred * depth_std + depth_mean)
    return np.concatenate(out, axis=0)


def _backproject_pixels(
    pixels: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    import cv2

    undist = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    cam = np.column_stack([undist[:, 0] * depth, undist[:, 1] * depth, depth])
    return (R.T @ (cam - t.reshape(1, 3)).T).T


def _interpolate_sparse_depth(pixels: np.ndarray, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
    from scipy.interpolate import griddata

    interp = griddata(uv, depth, pixels, method="linear")
    missing = ~np.isfinite(interp)
    if missing.any():
        interp[missing] = griddata(uv, depth, pixels[missing], method="nearest")
    return interp


def _evaluate_sparse_errors(
    model: SfMDepthFiLMNet,
    xy_norm: np.ndarray,
    cam_indices: np.ndarray,
    sparse_depth: np.ndarray,
    depth_mean: float,
    depth_std: float,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    pred = _predict_depth_batches(
        model=model,
        xy_norm=xy_norm,
        cam_indices=cam_indices,
        depth_mean=depth_mean,
        depth_std=depth_std,
        batch_size=262144,
        device=device,
    )
    return {"pred_depth": pred, "error": pred - sparse_depth}


def _save_meta(
    output_dir: Path,
    cfg: DepthInitConfig,
    cam_names: List[str],
    image_sizes: np.ndarray,
    history: Dict[str, List[float]],
    depth_mean: float,
    depth_std: float,
    diagnostics: Dict[str, np.ndarray],
    sparse_filter_meta: Dict[str, object],
) -> None:
    error = diagnostics["error"]
    meta = {
        "purpose": "SfM-guided camera-conditioned neural depth initialisation",
        "coordinate_convention": "depth is camera-coordinate Z in x_cam = R @ x_world + t",
        "config": asdict(cfg),
        "cam_names": cam_names,
        "image_sizes_wh": image_sizes.tolist(),
        "depth_mean": depth_mean,
        "depth_std": depth_std,
        "final_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "sparse_error_mean": float(error.mean()),
        "sparse_error_std": float(error.std()),
        "sparse_error_rmse": float(np.sqrt(np.mean(error**2))),
        "sparse_filter": sparse_filter_meta,
    }
    with open(output_dir / "model_init_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _save_depth_normalization(output_dir: Path, depth_mean: float, depth_std: float) -> None:
    payload = {
        "normalization": "z_norm = (z_physical - depth_mean) / depth_std",
        "denormalization": "z_physical = z_norm * depth_std + depth_mean",
        "depth_type": "camera-coordinate Z-depth",
        "coordinate_convention": "x_cam = R @ x_world + t",
        "depth_mean": depth_mean,
        "depth_std": depth_std,
    }
    with open(output_dir / "depth_normalization.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_dense_products(
    output_dir: Path,
    cam_names: List[str],
    dense_products: List[Dict[str, np.ndarray]],
) -> None:
    dense_dir = output_dir / "per_camera_dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for cam_name, item in zip(cam_names, dense_products):
        path = dense_dir / f"{cam_name}_dense_init.npz"
        np.savez_compressed(
            path,
            roi_contour=item["contour"],
            roi_mask=item["mask"],
            pixels=item["pixels"],
            pred_depth=item["pred_depth"],
            sfm_interp_depth=item["interp_depth"],
            world=item["world"],
            roi_source=item["roi_source"],
            roi_dir=item["roi_dir"],
        )
        index.append(
            {
                "cam_name": cam_name,
                "path": str(path),
                "n_pixels": int(len(item["pixels"])),
                "roi_source": str(item["roi_source"]),
                "roi_dir": str(item["roi_dir"]),
            }
        )
    with open(output_dir / "per_camera_dense_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def _save_visualisations(
    output_dir: Path,
    cam_names: List[str],
    sparse_points: np.ndarray,
    dense_products: List[Dict[str, np.ndarray]],
    observations: Dict[str, np.ndarray],
    sparse_errors: np.ndarray,
    cfg: DepthInitConfig,
) -> Dict[str, Path]:
    import matplotlib.pyplot as plt

    paths = {
        "dense_vs_sparse": output_dir / "01_dense_init_vs_sfm_sparse.png",
        "predicted_depth": output_dir / "02_predicted_depth_maps.png",
        "sfm_interpolated_depth": output_dir / "03_sfm_interpolated_depth_maps.png",
        "sparse_error_hist": output_dir / "04_sparse_depth_error_histograms.png",
    }
    _plot_dense_vs_sparse(paths["dense_vs_sparse"], sparse_points, dense_products, cfg)
    _plot_depth_grid(
        paths["predicted_depth"],
        cam_names,
        dense_products,
        "pred_depth",
        "Predicted camera Z-depth, SfM physical scale",
    )
    _plot_depth_grid(
        paths["sfm_interpolated_depth"],
        cam_names,
        dense_products,
        "interp_depth",
        "SfM observed camera Z-depth interpolation, physical scale",
    )
    _plot_error_hist(paths["sparse_error_hist"], cam_names, observations, sparse_errors)
    plt.close("all")
    return paths


def _plot_dense_vs_sparse(
    path: Path,
    sparse_points: np.ndarray,
    dense_products: List[Dict[str, np.ndarray]],
    cfg: DepthInitConfig,
) -> None:
    import matplotlib.pyplot as plt

    dense_chunks = []
    stride = max(1, int(cfg.point_plot_stride))
    for item in dense_products:
        dense_chunks.append(item["world"][::stride])
    dense = np.concatenate(dense_chunks, axis=0)

    fig = plt.figure(figsize=(12, 5), dpi=180)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.scatter(sparse_points[:, 0], sparse_points[:, 1], sparse_points[:, 2], s=1.5, c=sparse_points[:, 2], cmap="viridis")
    ax1.set_title("SfM sparse reconstruction, world scale")
    ax2.scatter(dense[:, 0], dense[:, 1], dense[:, 2], s=0.2, c=dense[:, 2], cmap="viridis")
    ax2.set_title(f"Neural dense init, SfM world scale (1/{stride} plotted)")
    for ax in (ax1, ax2):
        ax.set_xlabel("SfM world X")
        ax.set_ylabel("SfM world Y")
        ax.set_zlabel("SfM world Z")
        _set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(path)


def _plot_depth_grid(
    path: Path,
    cam_names: List[str],
    dense_products: List[Dict[str, np.ndarray]],
    key: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    values = np.concatenate([item[key] for item in dense_products])
    vmin, vmax = np.nanpercentile(values, [1, 99])
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), dpi=180, constrained_layout=True)
    fig.suptitle(title)
    last_im = None
    for ax, cam_name, item in zip(axes.ravel(), cam_names, dense_products):
        image = np.full(item["mask"].shape, np.nan, dtype=np.float32)
        pixels = item["pixels"].astype(np.int64)
        image[pixels[:, 1], pixels[:, 0]] = item[key]
        last_im = ax.imshow(image, cmap="turbo", vmin=vmin, vmax=vmax)
        contour = item["contour"]
        if len(contour) >= 2:
            ax.plot(contour[:, 0], contour[:, 1], "w-", linewidth=0.6)
            ax.plot(
                [contour[-1, 0], contour[0, 0]],
                [contour[-1, 1], contour[0, 1]],
                "w-",
                linewidth=0.6,
            )
        ax.set_title(cam_name)
        ax.set_axis_off()
    if last_im is not None:
        fig.colorbar(
            last_im,
            ax=axes.ravel().tolist(),
            shrink=0.75,
            label="Camera Z-depth in SfM physical scale",
        )
    fig.savefig(path)


def _plot_error_hist(
    path: Path,
    cam_names: List[str],
    observations: Dict[str, np.ndarray],
    sparse_errors: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), dpi=180, constrained_layout=True)
    fig.suptitle("Sparse observation Z-depth residuals in SfM physical scale: predicted - SfM")
    for ax, cam_name, cam_id in zip(axes.ravel(), cam_names, range(len(cam_names))):
        err = sparse_errors[observations["cam_indices"] == cam_id]
        if len(err) == 0:
            ax.set_axis_off()
            continue
        weights = np.ones_like(err) / len(err)
        ax.hist(err, bins=40, weights=weights, color="#4c78a8", edgecolor="white", linewidth=0.3)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{cam_name}  RMSE={np.sqrt(np.mean(err**2)):.4g}")
        ax.set_xlabel("Camera Z-depth error in SfM physical scale")
        ax.set_ylabel("Ratio")
    fig.savefig(path)


def _set_axes_equal(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = np.abs(limits[:, 1] - limits[:, 0])
    centres = np.mean(limits, axis=1)
    radius = 0.5 * max(spans)
    ax.set_xlim3d([centres[0] - radius, centres[0] + radius])
    ax.set_ylim3d([centres[1] - radius, centres[1] + radius])
    ax.set_zlim3d([centres[2] - radius, centres[2] + radius])
