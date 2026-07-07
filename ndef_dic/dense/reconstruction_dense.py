"""Export and visualise dense reconstruction after ZNSSD optimisation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .dense_znssd import backproject_pixels_torch, load_depth_model, load_npz, normalise_pixels_torch, select_device


@dataclass
class DenseReconstructionConfig:
    data_dir: str = "case/CylinderDIC"
    sfm_dir: str | None = None
    model_init_dir: str | None = None
    znssd_dir: str | None = None
    output_dir: str | None = None
    prediction_batch_size: int = 262144
    point_plot_stride: int = 20
    device: str = "auto"


def run_dense_reconstruction(config: DenseReconstructionConfig | None = None) -> Dict[str, str]:
    cfg = config or DenseReconstructionConfig()
    data_dir = Path(cfg.data_dir)
    sfm_dir = Path(cfg.sfm_dir) if cfg.sfm_dir else data_dir / "result" / "sfm"
    model_init_dir = (
        Path(cfg.model_init_dir)
        if cfg.model_init_dir
        else data_dir / "result" / "dense" / "model_init"
    )
    znssd_dir = Path(cfg.znssd_dir) if cfg.znssd_dir else data_dir / "result" / "dense" / "znssd_opt"
    output_dir = (
        Path(cfg.output_dir)
        if cfg.output_dir
        else data_dir / "result" / "dense" / "reconstruction_dense"
    )
    per_camera_dir = output_dir / "per_camera_dense"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_camera_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(cfg.device)
    cameras = load_npz(sfm_dir / "cameras.npz")
    sparse = _load_plot_sparse(sfm_dir, model_init_dir)
    cam_names = [str(x) for x in cameras["cam_names"]]
    image_sizes = torch.as_tensor(_load_image_sizes(cameras["image_paths"]), dtype=torch.float32, device=device)
    K = torch.as_tensor(cameras["K"], dtype=torch.float32, device=device)
    dist = torch.as_tensor(cameras["dist"], dtype=torch.float32, device=device)
    R = torch.as_tensor(cameras["R"], dtype=torch.float32, device=device)
    t = torch.as_tensor(cameras["t"].reshape(len(cam_names), 3), dtype=torch.float32, device=device)

    model, depth_mean, depth_std = load_depth_model(model_init_dir, device)
    znssd_ckpt = torch.load(znssd_dir / "depth_film_znssd.pt", map_location=device, weights_only=False)
    model.load_state_dict(znssd_ckpt["model_state_dict"])
    model.eval()

    init_dense_world = []
    znssd_dense_world = []
    index = []
    for cam_id, cam_name in enumerate(cam_names):
        init_path = model_init_dir / "per_camera_dense" / f"{cam_name}_dense_init.npz"
        init_data = np.load(init_path, allow_pickle=True)
        pixels = init_data["pixels"].astype(np.float32)
        roi_mask = init_data["roi_mask"].astype(bool)
        init_world = init_data["world"].astype(np.float32)
        pred_depth, world = _predict_camera_dense(
            model=model,
            pixels=pixels,
            cam_id=cam_id,
            image_sizes=image_sizes,
            K=K,
            dist=dist,
            R=R,
            t=t,
            depth_mean=depth_mean,
            depth_std=depth_std,
            batch_size=cfg.prediction_batch_size,
            device=device,
        )
        out_path = per_camera_dir / f"{cam_name}_znssd_dense.npz"
        np.savez_compressed(
            out_path,
            roi_mask=roi_mask,
            pixels=pixels,
            pred_depth=pred_depth.astype(np.float32),
            world=world.astype(np.float32),
        )
        init_dense_world.append(init_world[:: max(1, cfg.point_plot_stride)])
        znssd_dense_world.append(world[:: max(1, cfg.point_plot_stride)])
        index.append({"cam_name": cam_name, "path": str(out_path), "n_pixels": int(len(pixels))})
        print(f"[ReconDense] {cam_name}: exported {len(pixels)} ZNSSD dense points")

    with open(output_dir / "reconstruction_dense_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    with open(output_dir / "reconstruction_dense_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "source_checkpoint": str(znssd_dir / "depth_film_znssd.pt"),
                "depth_mean": depth_mean,
                "depth_std": depth_std,
                "cam_names": cam_names,
            },
            f,
            indent=2,
        )

    fig_path = output_dir / "dense_reconstruction_comparison.png"
    _plot_three_way(
        fig_path,
        sparse["points3D"].astype(np.float64),
        np.concatenate(init_dense_world, axis=0),
        np.concatenate(znssd_dense_world, axis=0),
    )
    return {
        "figure": str(fig_path),
        "index": str(output_dir / "reconstruction_dense_index.json"),
        "meta": str(output_dir / "reconstruction_dense_meta.json"),
    }


def _load_plot_sparse(sfm_dir: Path, model_init_dir: Path) -> Dict[str, np.ndarray]:
    filtered = model_init_dir / "sparse_filter" / "sparse_points_filtered.npz"
    if filtered.exists():
        return load_npz(filtered)
    return load_npz(sfm_dir / "sparse_points.npz")


def _predict_camera_dense(
    model,
    pixels: np.ndarray,
    cam_id: int,
    image_sizes: torch.Tensor,
    K: torch.Tensor,
    dist: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    depth_mean: float,
    depth_std: float,
    batch_size: int,
    device: torch.device,
):
    depths = []
    worlds = []
    with torch.no_grad():
        for start in range(0, len(pixels), batch_size):
            stop = min(start + batch_size, len(pixels))
            uv = torch.as_tensor(pixels[start:stop], dtype=torch.float32, device=device)
            cam = torch.full((len(uv),), cam_id, dtype=torch.long, device=device)
            xy_norm = normalise_pixels_torch(uv, cam, image_sizes)
            depth_norm = model(xy_norm, cam)
            depth = depth_norm * depth_std + depth_mean
            world = backproject_pixels_torch(uv, depth, cam, K, dist, R, t)
            depths.append(depth.detach().cpu().numpy())
            worlds.append(world.detach().cpu().numpy())
    return np.concatenate(depths, axis=0), np.concatenate(worlds, axis=0)


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


def _plot_three_way(path: Path, sparse: np.ndarray, init_dense: np.ndarray, znssd_dense: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 5), dpi=180)
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]
    titles = [
        "COLMAP sparse reconstruction",
        "Camera-network initial dense reconstruction",
        "ZNSSD-refined dense reconstruction",
    ]
    point_sets = [sparse, init_dense, znssd_dense]
    sizes = [1.4, 0.2, 0.2]
    for ax, title, pts, size in zip(axes, titles, point_sets, sizes):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, c=pts[:, 2], cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("SfM world X")
        ax.set_ylabel("SfM world Y")
        ax.set_zlabel("SfM world Z")
        _set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _set_axes_equal(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = np.abs(limits[:, 1] - limits[:, 0])
    centres = np.mean(limits, axis=1)
    radius = 0.5 * max(spans)
    ax.set_xlim3d([centres[0] - radius, centres[0] + radius])
    ax.set_ylim3d([centres[1] - radius, centres[1] + radius])
    ax.set_zlim3d([centres[2] - radius, centres[2] + radius])
