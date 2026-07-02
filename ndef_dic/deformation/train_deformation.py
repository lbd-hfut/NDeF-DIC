"""Training entry point for the reference-surface neural deformation field."""

from __future__ import annotations

import json
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .deformation_dataset import DeformationDatasetConfig, SurfaceDeformationDataset
from .deformation_field import DeformationFieldConfig, NeuralDisplacementField
from .deformation_loss import deformation_photometric_mse, smoothness_loss


@dataclass
class DeformationTrainingConfig:
    data_dir: str = "case/CylinderDIC"
    sfm_dir: str | None = None
    surface_dataset_path: str | None = None
    output_dir: str | None = None
    image_dir: str = "images"
    reference_name: str = "001"
    current_name: str = "002"
    hidden_dim: int = 32
    hidden_layers: int = 5
    use_positional_encoding: bool = True
    positional_encoding_frequencies: int = 6
    displacement_scale: float | None = None
    displacement_scale_path: str | None = None
    displacement_scale_stat: str = "mean"
    sfm2world_scale: float | None = None
    sfm2world_scale_path: str | None = None
    lambda_smooth: float = 1e-5
    photometric_loss: str = "mse"
    patch_radius: int = 2
    min_valid_patch_ratio: float = 1.0
    invalid_patch_penalty: float = 0.05
    znssd_eps: float = 1e-6
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int | str = "auto"
    auto_batch_start: int = 1024
    auto_batch_max: int | None = None
    memory_fraction: float = 0.80
    max_steps_per_epoch: int | None = None
    log_interval: int = 10
    max_visualization_points: int = 60000
    seed: int = 23
    device: str = "auto"


def run_deformation_training(config: DeformationTrainingConfig | None = None) -> Dict[str, str]:
    cfg = config or DeformationTrainingConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir) if cfg.output_dir else data_dir / "result" / "deformation"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SurfaceDeformationDataset(
        DeformationDatasetConfig(
            data_dir=cfg.data_dir,
            sfm_dir=cfg.sfm_dir,
            surface_dataset_path=cfg.surface_dataset_path,
            image_dir=cfg.image_dir,
            reference_name=cfg.reference_name,
            current_name=cfg.current_name,
            device=cfg.device,
        )
    )
    field_cfg = DeformationFieldConfig(
        hidden_dim=cfg.hidden_dim,
        hidden_layers=cfg.hidden_layers,
        use_positional_encoding=cfg.use_positional_encoding,
        positional_encoding_frequencies=cfg.positional_encoding_frequencies,
        output_scale=resolve_displacement_scale(cfg, data_dir),
    )
    print(f"[Deformation] displacement_scale={field_cfg.output_scale:.8g}")
    world_scale = resolve_sfm2world_scale(cfg, data_dir)
    print(f"[Deformation] sfm2world_scale={world_scale:.8g}")
    model = NeuralDisplacementField(field_cfg, dataset.coord_center, dataset.coord_scale).to(dataset.device)
    lambda_smooth = float(cfg.lambda_smooth) if cfg.use_positional_encoding else 0.0

    if cfg.batch_size == "auto":
        batch_size = estimate_batch_size(model, dataset, cfg, lambda_smooth)
    else:
        batch_size = int(cfg.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = int(np.ceil(dataset.n_points / batch_size))
    if cfg.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, int(cfg.max_steps_per_epoch))

    history = []
    best_loss = float("inf")
    best_record = None
    best_state_dict = None
    for epoch in range(1, cfg.epochs + 1):
        for step in range(1, steps_per_epoch + 1):
            batch = dataset.batch(dataset.sample_indices(batch_size))
            metrics = deformation_photometric_mse(
                model=model,
                batch=batch,
                cameras=dataset.cameras,
                reference_images=dataset.reference_images,
                current_images=dataset.current_images,
                image_sizes=dataset.image_sizes,
                loss_type=cfg.photometric_loss,
                patch_radius=cfg.patch_radius,
                min_valid_patch_ratio=cfg.min_valid_patch_ratio,
                invalid_patch_penalty=cfg.invalid_patch_penalty,
                znssd_eps=cfg.znssd_eps,
            )
            l_smooth = smoothness_loss(model, batch["points"]) if lambda_smooth > 0.0 else metrics["loss"].new_zeros(())
            loss = metrics["loss"] + lambda_smooth * l_smooth
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            record = {
                "epoch": epoch,
                "step": step,
                "loss": float(loss.detach().cpu()),
                "photometric_loss": float(metrics["photometric_loss"].detach().cpu()),
                "photometric_mse": float(metrics["photometric_mse"].detach().cpu()),
                "smoothness": float(l_smooth.detach().cpu()),
                "valid_pairs": float(metrics["valid_pairs"].detach().cpu()),
                "supervised_pairs": float(metrics.get("supervised_pairs", metrics["valid_pairs"]).detach().cpu()),
                "displacement_rms": float(metrics["displacement_rms"].detach().cpu()),
            }
            history.append(record)
            if record["loss"] < best_loss:
                best_loss = record["loss"]
                best_record = dict(record)
                best_state_dict = copy.deepcopy(model.state_dict())
            if step == 1 or step % cfg.log_interval == 0:
                print(
                    f"[Deformation] epoch={epoch} step={step}/{steps_per_epoch} "
                    f"loss={record['loss']:.6g} photo={record['photometric_loss']:.6g} "
                    f"smooth={record['smoothness']:.6g} pairs={record['valid_pairs']:.0f}/"
                    f"{record['supervised_pairs']:.0f} "
                    f"disp_rms={record['displacement_rms']:.6g}"
                )

    checkpoint_path = output_dir / "deformation_field.pt"
    best_checkpoint_path = output_dir / "deformation_field_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "field_config": asdict(field_cfg),
            "training_config": asdict(cfg),
            "displacement_scale": field_cfg.output_scale,
            "sfm2world_scale": world_scale,
            "coord_center": dataset.coord_center.detach().cpu(),
            "coord_scale": dataset.coord_scale.detach().cpu(),
            "batch_size": batch_size,
            "lambda_smooth_effective": lambda_smooth,
            "cam_names": dataset.cam_names,
        },
        checkpoint_path,
    )
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "field_config": asdict(field_cfg),
            "training_config": asdict(cfg),
            "displacement_scale": field_cfg.output_scale,
            "sfm2world_scale": world_scale,
            "coord_center": dataset.coord_center.detach().cpu(),
            "coord_scale": dataset.coord_scale.detach().cpu(),
            "batch_size": batch_size,
            "lambda_smooth_effective": lambda_smooth,
            "cam_names": dataset.cam_names,
            "best_record": best_record,
        },
        best_checkpoint_path,
    )
    history_path = output_dir / "deformation_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    surface_result_path = output_dir / "deformation_surface_result.npz"
    displacement_figure_path = output_dir / "deformation_3d_components.png"
    loss_figure_path = output_dir / "deformation_loss_curve.png"
    displacement_result = export_surface_displacement(
        model=model,
        dataset=dataset,
        path=surface_result_path,
        batch_size=batch_size,
        world_scale=world_scale,
    )
    plot_displacement_components(
        points=displacement_result["points"],
        displacement=displacement_result["displacement"],
        path=displacement_figure_path,
        max_points=cfg.max_visualization_points,
        seed=cfg.seed,
    )
    plot_loss_curve(history, loss_figure_path)
    meta_path = output_dir / "deformation_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "n_surface_points": dataset.n_points,
                "n_cameras": dataset.n_cameras,
                "batch_size": batch_size,
                "lambda_smooth_effective": lambda_smooth,
                "displacement_scale": field_cfg.output_scale,
                "sfm2world_scale": world_scale,
                "best_record": best_record,
                "output_schema": {
                    "checkpoint": "deformation_field.pt",
                    "best_checkpoint": "deformation_field_best.pt",
                    "history": "deformation_history.json",
                    "surface_result": "deformation_surface_result.npz",
                    "displacement_figure": "deformation_3d_components.png",
                    "loss_figure": "deformation_loss_curve.png",
                },
            },
            f,
            indent=2,
        )
    return {
        "checkpoint": str(checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "history": str(history_path),
        "surface_result": str(surface_result_path),
        "displacement_figure": str(displacement_figure_path),
        "loss_figure": str(loss_figure_path),
        "meta": str(meta_path),
    }


def resolve_displacement_scale(cfg: DeformationTrainingConfig, data_dir: Path) -> float:
    if cfg.displacement_scale is not None:
        scale = float(cfg.displacement_scale)
        if scale <= 0:
            raise ValueError("displacement_scale must be positive.")
        return scale

    scale_path = (
        Path(cfg.displacement_scale_path)
        if cfg.displacement_scale_path
        else data_dir / "result" / "deformation" / "precalculation" / "patch_dic_sparse" / "displacement_scale.json"
    )
    if not scale_path.exists():
        print(f"[Deformation] displacement scale file not found; using 1.0: {scale_path}")
        return 1.0
    with open(scale_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    stats = payload.get("scale_stats", {})
    stat_name = str(cfg.displacement_scale_stat)
    value = stats.get(stat_name)
    if value is None:
        raise KeyError(f"Scale statistic '{stat_name}' not found in {scale_path}")
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid displacement scale {scale} from {scale_path}")
    print(f"[Deformation] loaded displacement_scale={scale:.8g} ({stat_name}) from {scale_path}")
    return scale


def resolve_sfm2world_scale(cfg: DeformationTrainingConfig, data_dir: Path) -> float:
    if cfg.sfm2world_scale is not None:
        scale = float(cfg.sfm2world_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("sfm2world_scale must be positive.")
        return scale

    scale_path = (
        Path(cfg.sfm2world_scale_path)
        if cfg.sfm2world_scale_path
        else data_dir / "result" / "sfm2world" / "sfm2world_scale.json"
    )
    if not scale_path.exists():
        print(f"[Deformation] sfm2world scale file not found; exporting SfM-scale result: {scale_path}")
        return 1.0

    with open(scale_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    scale_payload = payload.get("scale", {})
    value = scale_payload.get("sfm_to_world_scale")
    if value is None:
        raise KeyError(f"'scale.sfm_to_world_scale' not found in {scale_path}")
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid sfm2world scale {scale} from {scale_path}")
    print(f"[Deformation] loaded sfm2world_scale={scale:.8g} from {scale_path}")
    return scale


@torch.no_grad()
def export_surface_displacement(
    model: NeuralDisplacementField,
    dataset: SurfaceDeformationDataset,
    path: Path,
    batch_size: int,
    world_scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    model.eval()
    displacements = []
    for start in range(0, dataset.n_points, int(batch_size)):
        stop = min(start + int(batch_size), dataset.n_points)
        disp = model(dataset.points[start:stop]).detach().cpu().numpy()
        displacements.append(disp)
    world_scale = float(world_scale)
    displacement_sfm = np.concatenate(displacements, axis=0).astype(np.float32)
    points_sfm = dataset.points.detach().cpu().numpy().astype(np.float32)
    displacement = (displacement_sfm * world_scale).astype(np.float32)
    points = (points_sfm * world_scale).astype(np.float32)
    magnitude = np.linalg.norm(displacement, axis=1).astype(np.float32)
    np.savez_compressed(
        path,
        points=points,
        displacement=displacement,
        displacement_magnitude=magnitude,
        points_sfm=points_sfm,
        displacement_sfm=displacement_sfm,
        displacement_magnitude_sfm=np.linalg.norm(displacement_sfm, axis=1).astype(np.float32),
        sfm2world_scale=np.asarray(world_scale, dtype=np.float64),
        coordinate_unit=np.asarray("world"),
        cam_names=np.asarray(dataset.cam_names),
    )
    model.train()
    return {
        "points": points,
        "displacement": displacement,
        "displacement_magnitude": magnitude,
        "sfm2world_scale": np.asarray(world_scale, dtype=np.float64),
    }


def plot_displacement_components(
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
        scatter = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=value,
            s=1.4,
            cmap="viridis" if i == 1 else "coolwarm",
            linewidths=0.0,
        )
        ax.set_title(title)
        ax.set_xlabel("World X")
        ax.set_ylabel("World Y")
        ax.set_zlabel("World Z")
        _set_axes_equal(ax, pts)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.08)
        cbar.set_label(f"{label} (world units)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_loss_curve(history: list[Dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    if not history:
        return
    steps = np.arange(1, len(history) + 1)
    loss = np.asarray([row["loss"] for row in history], dtype=np.float64)
    photo = np.asarray([row.get("photometric_loss", row["photometric_mse"]) for row in history], dtype=np.float64)
    smooth = np.asarray([row["smoothness"] for row in history], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    ax.plot(steps, loss, linewidth=1.2, label="total loss", color="#1f77b4")
    ax.plot(steps, photo, linewidth=1.0, label="photometric loss", color="#d62728", alpha=0.85)
    if np.any(smooth > 0):
        ax.plot(steps, smooth, linewidth=1.0, label="Lsmooth", color="#2ca02c", alpha=0.75)
    if len(loss) >= 5:
        window = min(50, max(5, len(loss) // 10))
        kernel = np.ones(window, dtype=np.float64) / window
        loss_smooth = np.convolve(loss, kernel, mode="valid")
        ax.plot(
            np.arange(window, window + len(loss_smooth)),
            loss_smooth,
            linewidth=1.5,
            color="black",
            label=f"total moving avg ({window})",
        )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Neural deformation training loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
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


def estimate_batch_size(
    model: NeuralDisplacementField,
    dataset: SurfaceDeformationDataset,
    cfg: DeformationTrainingConfig,
    lambda_smooth: float,
) -> int:
    device = dataset.device
    if device.type != "cuda":
        print("[Deformation] Non-CUDA device; using auto_batch_start.")
        return min(int(cfg.auto_batch_start), dataset.n_points)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    target = free_bytes * float(cfg.memory_fraction)
    max_probe = int(cfg.auto_batch_max) if cfg.auto_batch_max is not None else dataset.n_points
    max_probe = max(1, min(max_probe, dataset.n_points))
    start = max(1, min(int(cfg.auto_batch_start), max_probe))
    print(
        f"[Deformation] auto batch target: {cfg.memory_fraction:.0%} of free CUDA memory "
        f"({target / 1024**3:.2f}GB); search upper={max_probe}"
    )

    def try_batch(batch_size: int) -> tuple[bool, int]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        batch = dataset.batch(dataset.sample_indices(batch_size))
        try:
            metrics = deformation_photometric_mse(
                model=model,
                batch=batch,
                cameras=dataset.cameras,
                reference_images=dataset.reference_images,
                current_images=dataset.current_images,
                image_sizes=dataset.image_sizes,
                loss_type=cfg.photometric_loss,
                patch_radius=cfg.patch_radius,
                min_valid_patch_ratio=cfg.min_valid_patch_ratio,
                invalid_patch_penalty=cfg.invalid_patch_penalty,
                znssd_eps=cfg.znssd_eps,
            )
            l_smooth = smoothness_loss(model, batch["points"]) if lambda_smooth > 0.0 else metrics["loss"].new_zeros(())
            loss = metrics["loss"] + lambda_smooth * l_smooth
            loss.backward()
            model.zero_grad(set_to_none=True)
            peak = int(torch.cuda.max_memory_allocated(device))
            return peak <= target, peak
        except RuntimeError as exc:
            model.zero_grad(set_to_none=True)
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                return False, int(torch.cuda.max_memory_allocated(device))
            raise

    high = start
    last_good = 0
    while high <= max_probe:
        ok, peak = try_batch(high)
        print(f"[Deformation] dry-run batch_size={high} peak={peak / 1024**3:.2f}GB ok={ok}")
        if not ok:
            break
        last_good = high
        high *= 2
    high = min(high, max_probe)
    low = last_good
    while high - low > 1:
        mid = (low + high) // 2
        ok, peak = try_batch(mid)
        print(f"[Deformation] dry-run batch_size={mid} peak={peak / 1024**3:.2f}GB ok={ok}")
        if ok:
            low = mid
            last_good = mid
        else:
            high = mid
    if last_good <= 0:
        raise RuntimeError("Even the minimum deformation batch exceeds the configured memory budget.")
    print(f"[Deformation] selected batch_size={last_good}")
    return int(last_good)
