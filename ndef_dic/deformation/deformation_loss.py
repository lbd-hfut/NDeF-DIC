"""Photometric and smoothness losses for neural deformation DIC."""

from __future__ import annotations

from typing import Dict

import torch

from .deformation_field import NeuralDisplacementField


def deformation_photometric_mse(
    model: NeuralDisplacementField,
    batch: Dict[str, torch.Tensor],
    cameras: Dict[str, torch.Tensor],
    reference_images: torch.Tensor,
    current_images: torch.Tensor,
    image_sizes: torch.Tensor,
    loss_type: str = "mse",
    patch_radius: int = 2,
    min_valid_patch_ratio: float = 1.0,
    invalid_patch_penalty: float = 0.05,
    znssd_eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    loss_type = loss_type.lower()
    if loss_type not in {"mse", "znssd"}:
        raise ValueError(f"Unsupported photometric loss: {loss_type}. Use 'mse' or 'znssd'.")

    points = batch["points"]
    visibility = batch["visibility_mask"]
    reference_uv_all = batch["projected_uv"]
    visible_counts = batch["visible_counts"]

    displacement = model(points)
    deformed_points = points + displacement

    pair_ids = torch.nonzero(visibility, as_tuple=False)
    if pair_ids.numel() == 0:
        zero = displacement.sum() * 0.0
        return {
            "loss": zero,
            "photometric_loss": zero,
            "photometric_mse": zero,
            "valid_pairs": zero.detach(),
            "displacement_rms": displacement.square().sum(dim=1).mean().sqrt().detach(),
        }

    point_ids = pair_ids[:, 0]
    cam_ids = pair_ids[:, 1]
    ref_uv = reference_uv_all[point_ids, cam_ids]
    cur_uv, depth = project_world_torch(deformed_points[point_ids], cam_ids, cameras)
    patch_offsets = make_patch_offsets(int(patch_radius), ref_uv.device, ref_uv.dtype)
    ref_patch_uv = ref_uv[:, None, :] + patch_offsets[None, :, :]
    cur_patch_uv = cur_uv[:, None, :] + patch_offsets[None, :, :]
    ref_patch_bounds = in_image_patch_bounds(ref_patch_uv, cam_ids, image_sizes)
    cur_patch_bounds = in_image_patch_bounds(cur_patch_uv, cam_ids, image_sizes)
    ref_valid_counts = ref_patch_bounds.float().sum(dim=1)
    min_valid = max(1.0, patch_offsets.shape[0] * float(min_valid_patch_ratio))
    supervised = ref_valid_counts >= min_valid
    if not torch.any(supervised):
        zero = displacement.sum() * 0.0
        return {
            "loss": zero,
            "photometric_loss": zero,
            "photometric_mse": zero,
            "valid_pairs": zero.detach(),
            "displacement_rms": displacement.square().sum(dim=1).mean().sqrt().detach(),
        }

    point_ids = point_ids[supervised]
    cam_ids = cam_ids[supervised]
    ref_patch_uv = ref_patch_uv[supervised]
    cur_patch_uv = cur_patch_uv[supervised]
    cur_patch_bounds = cur_patch_bounds[supervised]
    depth = depth[supervised]
    cur_center_bounds = in_image_bounds(cur_uv[supervised], cam_ids, image_sizes) & (depth > 1e-8)
    cur_valid_counts = cur_patch_bounds.float().sum(dim=1)
    valid_current_patch = cur_center_bounds & (cur_valid_counts >= min_valid)

    patch_mse = torch.full(
        (len(point_ids),),
        float(invalid_patch_penalty),
        device=points.device,
        dtype=points.dtype,
    )
    if torch.any(valid_current_patch):
        valid_ids = torch.where(valid_current_patch)[0]
        valid_ref_patch_uv = ref_patch_uv[valid_ids]
        valid_cur_patch_uv = cur_patch_uv[valid_ids]
        valid_cam_ids = cam_ids[valid_ids]
        ref_gray = sample_per_camera(reference_images, valid_ref_patch_uv, valid_cam_ids)
        cur_gray = sample_per_camera(current_images, valid_cur_patch_uv, valid_cam_ids)
        if loss_type == "mse":
            patch_mse[valid_ids] = (ref_gray - cur_gray).square().mean(dim=1)
        else:
            patch_mse[valid_ids] = znssd_patch_loss(ref_gray, cur_gray, eps=float(znssd_eps))

    weights = 1.0 / visible_counts[point_ids]
    photometric = (patch_mse * weights).sum() / weights.sum().clamp_min(1e-8)
    return {
        "loss": photometric,
        "photometric_loss": photometric.detach(),
        "photometric_mse": photometric.detach(),
        "valid_pairs": torch.as_tensor(float(valid_current_patch.sum().detach().cpu()), device=points.device),
        "supervised_pairs": torch.as_tensor(float(len(patch_mse)), device=points.device),
        "displacement_rms": displacement.square().sum(dim=1).mean().sqrt().detach(),
    }


def znssd_patch_loss(ref_patch: torch.Tensor, cur_patch: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    ref_mean = ref_patch.mean(dim=1, keepdim=True)
    cur_mean = cur_patch.mean(dim=1, keepdim=True)
    ref_centered = ref_patch - ref_mean
    cur_centered = cur_patch - cur_mean
    ref_std = torch.sqrt(ref_centered.square().mean(dim=1, keepdim=True) + eps)
    cur_std = torch.sqrt(cur_centered.square().mean(dim=1, keepdim=True) + eps)
    ref_norm = ref_centered / ref_std
    cur_norm = cur_centered / cur_std
    return (ref_norm - cur_norm).square().mean(dim=1)


def smoothness_loss(
    model: NeuralDisplacementField,
    points: torch.Tensor,
) -> torch.Tensor:
    x_norm = model.normalize(points.detach()).requires_grad_(True)
    displacement = model.forward_normalized(x_norm)
    grads = []
    for component in range(3):
        grad = torch.autograd.grad(
            displacement[:, component].sum(),
            x_norm,
            create_graph=True,
            retain_graph=True,
        )[0]
        grads.append(grad)
    jac = torch.stack(grads, dim=1)
    return jac.square().sum(dim=(1, 2)).mean()


def project_world_torch(
    points_world: torch.Tensor,
    cam: torch.Tensor,
    cameras: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    K = cameras["K"]
    dist = cameras["dist"]
    R = cameras["R"]
    t = cameras["t"]
    cam_xyz = torch.bmm(R[cam], points_world.unsqueeze(-1)).squeeze(-1) + t[cam]
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


def in_image_bounds(uv: torch.Tensor, cam: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
    sizes = image_sizes[cam]
    width = sizes[:, 0]
    height = sizes[:, 1]
    return (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)


def in_image_patch_bounds(uv: torch.Tensor, cam: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
    sizes = image_sizes[cam]
    width = sizes[:, 0][:, None]
    height = sizes[:, 1][:, None]
    return (uv[..., 0] >= 0) & (uv[..., 0] <= width - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= height - 1)


def make_patch_offsets(radius: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if radius < 0:
        raise ValueError("patch_radius must be non-negative")
    values = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    dv, du = torch.meshgrid(values, values, indexing="ij")
    return torch.stack([du.reshape(-1), dv.reshape(-1)], dim=1)


def sample_per_camera(images: torch.Tensor, uv: torch.Tensor, cam: torch.Tensor) -> torch.Tensor:
    out = uv.new_empty((uv.shape[0], uv.shape[1]))
    for cam_id in torch.unique(cam):
        mask = cam == cam_id
        out[mask] = bilinear_sample_single(images[int(cam_id.item())], uv[mask])
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
    q00 = gray[y0c, x0c]
    q10 = gray[y0c, x1c]
    q01 = gray[y1c, x0c]
    q11 = gray[y1c, x1c]
    dx = x - x0.float()
    dy = y - y0.float()
    return (
        q00 * (1.0 - dx) * (1.0 - dy)
        + q10 * dx * (1.0 - dy)
        + q01 * (1.0 - dx) * dy
        + q11 * dx * dy
    )
