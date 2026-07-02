"""Dense multi-view ZNSSD optimisation scaffold.

This stage refines the same camera-conditioned depth network used for
``model_init``.  Each batch is balanced across source cameras.  The network
predicts normalised camera Z-depth, the depth is denormalised using the saved
SfM initialisation statistics, then source pixels are back-projected to
SfM/world coordinates and projected into their neighbouring cameras for ZNSSD.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .model_init import SfMDepthFiLMNet
from .reconstruction_dataset import BalancedPerCameraBatchLoader


@dataclass
class DenseZNSSDConfig:
    data_dir: str = "case/CylinderDIC"
    sfm_dir: str | None = None
    model_init_dir: str | None = None
    dataset_dir: str | None = None
    output_dir: str | None = None
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-6
    per_camera_batch: int | None = None
    auto_batch: bool = True
    auto_batch_start: int = 1
    auto_batch_max: int | None = None
    memory_fraction: float = 0.80
    min_valid_ratio: float = 0.50
    eps: float = 1e-6
    seed: int = 11
    device: str = "auto"
    max_steps_per_epoch: int | None = 5
    log_interval: int = 10


class DenseZNSSDLoss:
    def __init__(
        self,
        cameras: Dict[str, torch.Tensor],
        images: torch.Tensor,
        roi_masks: torch.Tensor,
        patch_offsets: torch.Tensor,
        image_sizes: torch.Tensor,
        depth_mean: float,
        depth_std: float,
        min_valid_ratio: float = 0.5,
        eps: float = 1e-6,
    ):
        self.cameras = cameras
        self.images = images
        self.roi_masks = roi_masks
        self.patch_offsets = patch_offsets
        self.image_sizes = image_sizes
        self.depth_mean = float(depth_mean)
        self.depth_std = float(depth_std)
        self.min_valid_ratio = float(min_valid_ratio)
        self.eps = float(eps)

    def __call__(self, model: SfMDepthFiLMNet, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        source_cam = batch["source_cam"].long()
        source_uv = batch["source_uv"].float()
        neighbor_ids = batch["neighbor_ids"].long()

        source_xy_norm = normalise_pixels_torch(source_uv, source_cam, self.image_sizes)
        depth_norm = model(source_xy_norm, source_cam)
        depth = depth_norm * self.depth_std + self.depth_mean
        world = backproject_pixels_torch(
            source_uv,
            depth,
            source_cam,
            self.cameras["K"],
            self.cameras["dist"],
            self.cameras["R"],
            self.cameras["t"],
        )

        source_patch_uv = source_uv[:, None, :] + self.patch_offsets[None, :, :]
        source_values = sample_per_camera(self.images, source_patch_uv, source_cam, mode="bilinear")
        source_roi = sample_per_camera(self.roi_masks, source_patch_uv, source_cam, mode="nearest") > 0.5
        source_bounds = in_image_bounds(source_patch_uv, source_cam, self.image_sizes)

        losses = []
        valid_counts = []
        slot_count = neighbor_ids.shape[1]
        for slot in range(slot_count):
            target_cam = neighbor_ids[:, slot]
            slot_valid = target_cam >= 0
            if not torch.any(slot_valid):
                continue
            target_uv, target_depth = project_world_torch(
                world,
                target_cam.clamp_min(0),
                self.cameras["K"],
                self.cameras["dist"],
                self.cameras["R"],
                self.cameras["t"],
            )
            target_patch_uv = target_uv[:, None, :] + self.patch_offsets[None, :, :]
            target_values = sample_per_camera(
                self.images,
                target_patch_uv,
                target_cam.clamp_min(0),
                mode="bilinear",
            )
            target_roi = (
                sample_per_camera(
                    self.roi_masks,
                    target_patch_uv,
                    target_cam.clamp_min(0),
                    mode="nearest",
                )
                > 0.5
            )
            target_bounds = in_image_bounds(target_patch_uv, target_cam.clamp_min(0), self.image_sizes)
            valid = (
                slot_valid[:, None]
                & (target_depth[:, None] > 1e-8)
                & source_roi
                & target_roi
                & source_bounds
                & target_bounds
            )
            znssd, counts = weighted_znssd(
                source_values,
                target_values,
                valid,
                min_valid_ratio=self.min_valid_ratio,
                eps=self.eps,
            )
            if znssd.numel():
                losses.append(znssd)
                valid_counts.append(counts)

        if not losses:
            zero = depth.sum() * 0.0
            return {"loss": zero, "valid_pairs": torch.zeros((), device=depth.device), "depth_mean": depth.mean()}

        all_losses = torch.cat(losses, dim=0)
        all_counts = torch.cat(valid_counts, dim=0)
        return {
            "loss": all_losses.mean(),
            "valid_pairs": torch.as_tensor(float(len(all_losses)), device=depth.device),
            "valid_patch_points_mean": all_counts.float().mean(),
            "depth_mean": depth.mean(),
        }


def run_dense_znssd(config: DenseZNSSDConfig | None = None) -> Dict[str, str]:
    cfg = config or DenseZNSSDConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    sfm_dir = Path(cfg.sfm_dir) if cfg.sfm_dir else data_dir / "result" / "sfm"
    model_init_dir = (
        Path(cfg.model_init_dir)
        if cfg.model_init_dir
        else data_dir / "result" / "dense" / "model_init"
    )
    dataset_dir = (
        Path(cfg.dataset_dir)
        if cfg.dataset_dir
        else data_dir / "result" / "dense" / "reconstruction_dataset"
    )
    output_dir = (
        Path(cfg.output_dir)
        if cfg.output_dir
        else data_dir / "result" / "dense" / "znssd_opt"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(cfg.device)
    cameras_np = load_npz(sfm_dir / "cameras.npz")
    cam_names = [str(x) for x in cameras_np["cam_names"]]
    image_sizes_np, images = load_images(cameras_np["image_paths"], device)
    roi_masks = load_roi_masks(model_init_dir, cam_names, device)
    cameras = {
        "K": torch.as_tensor(cameras_np["K"], dtype=torch.float32, device=device),
        "dist": torch.as_tensor(cameras_np["dist"], dtype=torch.float32, device=device),
        "R": torch.as_tensor(cameras_np["R"], dtype=torch.float32, device=device),
        "t": torch.as_tensor(cameras_np["t"].reshape(len(cam_names), 3), dtype=torch.float32, device=device),
    }
    image_sizes = torch.as_tensor(image_sizes_np, dtype=torch.float32, device=device)

    model, depth_mean, depth_std = load_depth_model(model_init_dir, device)
    manifest = dataset_dir / "dataset_manifest.json"
    patch_offsets = torch.as_tensor(np.load(dataset_dir / "patch_offsets.npy"), dtype=torch.float32, device=device)

    loss_fn = DenseZNSSDLoss(
        cameras=cameras,
        images=images,
        roi_masks=roi_masks,
        patch_offsets=patch_offsets,
        image_sizes=image_sizes,
        depth_mean=depth_mean,
        depth_std=depth_std,
        min_valid_ratio=cfg.min_valid_ratio,
        eps=cfg.eps,
    )

    if cfg.auto_batch or cfg.per_camera_batch is None:
        per_camera_batch = estimate_per_camera_batch(
            manifest,
            model,
            loss_fn,
            cfg,
            device,
        )
    else:
        per_camera_batch = int(cfg.per_camera_batch)
    loader = BalancedPerCameraBatchLoader(manifest, per_camera_batch=per_camera_batch, seed=cfg.seed)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        for step, batch_np in enumerate(loader, start=1):
            if cfg.max_steps_per_epoch is not None and step > cfg.max_steps_per_epoch:
                break
            batch = batch_to_device(batch_np, device)
            metrics = loss_fn(model, batch)
            opt.zero_grad(set_to_none=True)
            metrics["loss"].backward()
            opt.step()
            record = {
                "epoch": epoch,
                "step": step,
                "loss": float(metrics["loss"].detach().cpu()),
                "valid_pairs": float(metrics["valid_pairs"].detach().cpu()),
                "depth_mean": float(metrics["depth_mean"].detach().cpu()),
            }
            if "valid_patch_points_mean" in metrics:
                record["valid_patch_points_mean"] = float(
                    metrics["valid_patch_points_mean"].detach().cpu()
                )
            history.append(record)
            if step == 1 or step % cfg.log_interval == 0:
                print(
                    f"[DenseZNSSD] epoch={epoch} step={step}/{len(loader)} "
                    f"loss={record['loss']:.6f} valid_pairs={record['valid_pairs']:.0f} "
                    f"depth_mean={record['depth_mean']:.4f}"
                )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "depth_mean": depth_mean,
            "depth_std": depth_std,
            "per_camera_batch": per_camera_batch,
            "cam_names": cam_names,
        },
        output_dir / "depth_film_znssd.pt",
    )
    with open(output_dir / "znssd_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_path = output_dir / "loss_curve.png"
    _plot_loss_curve(history, plot_path)
    with open(output_dir / "znssd_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "per_camera_batch": per_camera_batch,
                "global_batch": per_camera_batch * len(loader.active_shards),
                "depth_mean": depth_mean,
                "depth_std": depth_std,
                "history_records": len(history),
            },
            f,
            indent=2,
        )
    return {
        "checkpoint": str(output_dir / "depth_film_znssd.pt"),
        "history": str(output_dir / "znssd_history.json"),
        "loss_curve": str(plot_path),
        "meta": str(output_dir / "znssd_meta.json"),
    }


def _plot_loss_curve(history: List[Dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    if not history:
        return
    steps = np.arange(1, len(history) + 1)
    losses = np.asarray([row["loss"] for row in history], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    ax.plot(steps, losses, linewidth=1.2, color="#2f6f9f")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("ZNSSD loss")
    ax.set_title("Dense ZNSSD optimisation loss")
    ax.grid(True, alpha=0.25)
    if len(losses) >= 5:
        window = min(25, max(5, len(losses) // 10))
        kernel = np.ones(window, dtype=np.float64) / window
        smooth = np.convolve(losses, kernel, mode="valid")
        ax.plot(
            np.arange(window, window + len(smooth)),
            smooth,
            linewidth=1.4,
            color="#d55e00",
            label=f"moving avg ({window})",
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def estimate_per_camera_batch(
    manifest: Path,
    model: SfMDepthFiLMNet,
    loss_fn: DenseZNSSDLoss,
    cfg: DenseZNSSDConfig,
    device: torch.device,
) -> int:
    if device.type != "cuda":
        print("[DenseZNSSD] Non-CUDA device; using auto_batch_start as per_camera_batch.")
        return int(cfg.auto_batch_start)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    target = free_bytes * cfg.memory_fraction
    probe_dataset = BalancedPerCameraBatchLoader(manifest, per_camera_batch=1, seed=cfg.seed).dataset
    max_dataset_per_camera = max(
        int(shard["record"]["n_samples"])
        for shard in probe_dataset.shards
        if int(shard["record"]["n_samples"]) > 0
    )
    max_probe = int(cfg.auto_batch_max) if cfg.auto_batch_max is not None else max_dataset_per_camera
    max_probe = max(1, min(max_probe, max_dataset_per_camera))
    print(
        f"[DenseZNSSD] auto batch target: {cfg.memory_fraction:.0%} of free CUDA memory "
        f"({target / 1024**3:.2f}GB); search upper={max_probe}"
    )

    def try_batch(per_camera_batch: int) -> Tuple[bool, int]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        loader = BalancedPerCameraBatchLoader(manifest, per_camera_batch=per_camera_batch, seed=cfg.seed)
        batch_np = next(iter(loader))
        batch = batch_to_device(batch_np, device)
        try:
            metrics = loss_fn(model, batch)
            metrics["loss"].backward()
            model.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated(device)
            return peak <= target, int(peak)
        except RuntimeError as exc:
            model.zero_grad(set_to_none=True)
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                return False, int(torch.cuda.max_memory_allocated(device))
            raise

    low = max(1, int(cfg.auto_batch_start))
    low = min(low, max_probe)
    high = low
    last_good = 0
    while high <= max_probe:
        ok, peak = try_batch(high)
        print(f"[DenseZNSSD] dry-run per_camera_batch={high} peak={peak/1024**3:.2f}GB ok={ok}")
        if not ok:
            break
        last_good = high
        high *= 2
    high = min(high, max_probe)
    low = last_good
    while high - low > 1:
        mid = (low + high) // 2
        ok, peak = try_batch(mid)
        print(f"[DenseZNSSD] dry-run per_camera_batch={mid} peak={peak/1024**3:.2f}GB ok={ok}")
        if ok:
            low = mid
            last_good = mid
        else:
            high = mid
    if last_good <= 0:
        raise RuntimeError("Even per_camera_batch=1 exceeds the configured memory budget.")
    print(f"[DenseZNSSD] selected per_camera_batch={last_good}")
    return int(last_good)


def weighted_znssd(
    source: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    min_valid_ratio: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_f = valid.float()
    counts = valid_f.sum(dim=1)
    min_count = max(1.0, source.shape[1] * min_valid_ratio)
    sample_valid = counts >= min_count
    if not torch.any(sample_valid):
        return source.new_empty((0,)), counts.new_empty((0,))
    source = source[sample_valid]
    target = target[sample_valid]
    valid_f = valid_f[sample_valid]
    counts = counts[sample_valid]
    mu_s = (source * valid_f).sum(dim=1, keepdim=True) / counts[:, None].clamp_min(1.0)
    mu_t = (target * valid_f).sum(dim=1, keepdim=True) / counts[:, None].clamp_min(1.0)
    ds = (source - mu_s) * valid_f
    dt = (target - mu_t) * valid_f
    sig_s = torch.sqrt((ds.square().sum(dim=1, keepdim=True) / counts[:, None]) + eps)
    sig_t = torch.sqrt((dt.square().sum(dim=1, keepdim=True) / counts[:, None]) + eps)
    residual = ((source - mu_s) / sig_s - (target - mu_t) / sig_t).square() * valid_f
    return residual.sum(dim=1) / counts.clamp_min(1.0), counts


def backproject_pixels_torch(
    uv: torch.Tensor,
    depth: torch.Tensor,
    cam: torch.Tensor,
    K: torch.Tensor,
    dist: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    Ki = K[cam]
    x = (uv[:, 0] - Ki[:, 0, 2]) / Ki[:, 0, 0]
    y = (uv[:, 1] - Ki[:, 1, 2]) / Ki[:, 1, 1]
    xy = undistort_normalized_torch(torch.stack([x, y], dim=1), dist[cam])
    cam_xyz = torch.stack([xy[:, 0] * depth, xy[:, 1] * depth, depth], dim=1)
    return torch.bmm(R[cam].transpose(1, 2), (cam_xyz - t[cam]).unsqueeze(-1)).squeeze(-1)


def project_world_torch(
    world: torch.Tensor,
    cam: torch.Tensor,
    K: torch.Tensor,
    dist: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cam_xyz = torch.bmm(R[cam], world.unsqueeze(-1)).squeeze(-1) + t[cam]
    z = cam_xyz[:, 2]
    x = cam_xyz[:, 0] / z.clamp_min(1e-8)
    y = cam_xyz[:, 1] / z.clamp_min(1e-8)
    xy_d = distort_normalized_torch(torch.stack([x, y], dim=1), dist[cam])
    Ki = K[cam]
    u = Ki[:, 0, 0] * xy_d[:, 0] + Ki[:, 0, 2]
    v = Ki[:, 1, 1] * xy_d[:, 1] + Ki[:, 1, 2]
    return torch.stack([u, v], dim=1), z


def distort_normalized_torch(xy: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    k1, k2, p1, p2, k3 = [dist[:, i] for i in range(5)]
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return torch.stack([x_d, y_d], dim=1)


def undistort_normalized_torch(xy_d: torch.Tensor, dist: torch.Tensor, iterations: int = 5) -> torch.Tensor:
    xy = xy_d
    k1, k2, p1, p2, k3 = [dist[:, i] for i in range(5)]
    for _ in range(iterations):
        x, y = xy[:, 0], xy[:, 1]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        xy = torch.stack([(xy_d[:, 0] - dx) / radial, (xy_d[:, 1] - dy) / radial], dim=1)
    return xy


def sample_per_camera(
    stack: torch.Tensor,
    uv: torch.Tensor,
    cam: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    out = uv.new_empty((uv.shape[0], uv.shape[1]))
    for cam_id in torch.unique(cam):
        mask = cam == cam_id
        if mode == "bilinear":
            values = bilinear_sample_single(stack[int(cam_id.item())], uv[mask])
        elif mode == "nearest":
            values = nearest_sample_single(stack[int(cam_id.item())], uv[mask])
        else:
            raise ValueError(f"Unsupported sampling mode: {mode}")
        out[mask] = values
    return out


def bilinear_sample_single(image: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    _, height, width = image.shape
    gray = image[0]
    x = uv[..., 0]
    y = uv[..., 1]
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1
    x0c = x0.clamp(0, width - 1)
    x1c = x1.clamp(0, width - 1)
    y0c = y0.clamp(0, height - 1)
    y1c = y1.clamp(0, height - 1)
    Ia = gray[y0c, x0c]
    Ib = gray[y1c, x0c]
    Ic = gray[y0c, x1c]
    Id = gray[y1c, x1c]
    x0f = x0.float()
    y0f = y0.float()
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0f)
    wc = (x - x0f) * (y1.float() - y)
    wd = (x - x0f) * (y - y0f)
    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def nearest_sample_single(image: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    _, height, width = image.shape
    gray = image[0]
    x = torch.round(uv[..., 0]).long().clamp(0, width - 1)
    y = torch.round(uv[..., 1]).long().clamp(0, height - 1)
    return gray[y, x]


def in_image_bounds(uv: torch.Tensor, cam: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
    sizes = image_sizes[cam]
    width = sizes[:, 0][:, None]
    height = sizes[:, 1][:, None]
    return (uv[..., 0] >= 0) & (uv[..., 0] <= width - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= height - 1)


def normalise_pixels_torch(uv: torch.Tensor, cam: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
    sizes = image_sizes[cam]
    x = 2.0 * uv[:, 0] / torch.clamp(sizes[:, 0] - 1.0, min=1.0) - 1.0
    y = 2.0 * uv[:, 1] / torch.clamp(sizes[:, 1] - 1.0, min=1.0) - 1.0
    return torch.stack([x, y], dim=1)


def batch_to_device(batch_np: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    keys = ["source_cam", "source_uv", "source_world", "neighbor_ids", "neighbor_uv"]
    return {
        key: torch.as_tensor(batch_np[key], device=device)
        for key in keys
    }


def load_depth_model(model_init_dir: Path, device: torch.device) -> Tuple[SfMDepthFiLMNet, float, float]:
    ckpt = torch.load(model_init_dir / "depth_film_init.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SfMDepthFiLMNet(
        n_cameras=len(ckpt["cam_names"]),
        hidden_dim=int(cfg["hidden_dim"]),
        camera_embedding_dim=int(cfg["camera_embedding_dim"]),
        pixel_layers=int(cfg["pixel_layers"]),
        camera_layers=int(cfg["camera_layers"]),
        trunk_layers=int(cfg["trunk_layers"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, float(ckpt["depth_mean"]), float(ckpt["depth_std"])


def load_images(image_paths: np.ndarray, device: torch.device) -> Tuple[np.ndarray, torch.Tensor]:
    import cv2

    images = []
    sizes = []
    for raw in image_paths:
        image = cv2.imread(str(Path(str(raw))), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(str(raw))
        height, width = image.shape
        sizes.append((width, height))
        images.append(torch.as_tensor(image.astype(np.float32) / 255.0)[None])
    return np.asarray(sizes, dtype=np.int64), torch.stack(images, dim=0).to(device)


def load_roi_masks(model_init_dir: Path, cam_names: List[str], device: torch.device) -> torch.Tensor:
    masks = []
    for cam_name in cam_names:
        data = np.load(model_init_dir / "per_camera_dense" / f"{cam_name}_dense_init.npz")
        masks.append(torch.as_tensor(data["roi_mask"].astype(np.float32))[None])
    return torch.stack(masks, dim=0).to(device)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[DenseZNSSD] CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)
