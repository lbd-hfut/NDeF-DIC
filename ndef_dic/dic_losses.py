"""
Step 3 shared losses and utilities.

Provides standalone functions for:
  - extract_patches:  Extract w×w patches from images via grid_sample.
  - znssd:            Zero-mean Normalized Sum of Squared Differences.
  - deformation_smoothness_loss: Frobenius norm of displacement gradient.

All functions are parameter-free and work with arbitrary batch sizes.
Extracted/adapted from pinn_stereo.py and temp/losses.py for reuse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Patch extraction
# =========================================================================

def extract_patches(
    image: torch.Tensor,              # (H, W) grayscale float
    uv_centers: torch.Tensor,         # (N, 2) pixel coords (col, row)
    patch_size: int,                  # side length (odd recommended)
    H: int,                           # image height
    W: int,                           # image width
    sub_batch_size: int = 128,
) -> torch.Tensor:
    """Extract square patches centered at pixel coordinates.

    Uses F.grid_sample with bilinear interpolation for sub-pixel accuracy.
    Processes in sub-batches to limit GPU memory.

    Args:
        image:      (H, W) grayscale float tensor.
        uv_centers: (N, 2) center pixel coords (col, row).
        patch_size: Side length of square patch.
        H, W:       Image dimensions.
        sub_batch_size: Max samples per grid_sample call.

    Returns:
        patches: (N, 1, patch_size, patch_size) intensity values.
    """
    N = uv_centers.shape[0]
    device = uv_centers.device
    half = (patch_size - 1) / 2.0

    # Build offset grid in pixel coordinates (shared across batch)
    dy, dx = torch.meshgrid(
        torch.arange(-half, half + 1, device=device, dtype=torch.float32),
        torch.arange(-half, half + 1, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # Normalize offsets once: pixel → [-1, 1]
    dx_norm = 2.0 * dx / (W - 1)
    dy_norm = 2.0 * dy / (H - 1)
    offsets_norm = torch.stack([dx_norm, dy_norm], dim=-1)  # (P, P, 2)

    all_patches = []

    for start in range(0, N, sub_batch_size):
        end = min(start + sub_batch_size, N)
        sb_uv = uv_centers[start:end]  # (sb, 2)
        sb_size = end - start

        # Normalize center coords: pixel → [-1, 1]
        u_norm = 2.0 * sb_uv[:, 0:1] / (W - 1) - 1.0  # (sb, 1)
        v_norm = 2.0 * sb_uv[:, 1:2] / (H - 1) - 1.0  # (sb, 1)
        centers_norm = torch.cat([u_norm, v_norm], dim=-1)  # (sb, 2)

        # Build per-patch sampling grid: (sb, P, P, 2)
        grid = centers_norm.unsqueeze(1).unsqueeze(1) + offsets_norm.unsqueeze(0)

        # Expand image for batch processing
        img_batch = image.unsqueeze(0).unsqueeze(0).expand(sb_size, 1, H, W)
        patches = F.grid_sample(
            img_batch, grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        all_patches.append(patches)

    return torch.cat(all_patches, dim=0)  # (N, 1, P, P)


# =========================================================================
# ZNSSD loss
# =========================================================================

def znssd(
    P_ref: torch.Tensor,   # (N, 1, P, P) or (N, P*P)
    P_def: torch.Tensor,   # (N, 1, P, P) or (N, P*P)
    eps: float = 1e-6,
) -> torch.Tensor:
    """Zero-mean Normalized Sum of Squared Differences.

    ZNSSD(P, Q) = mean_i [ Σ_j ((P_ij - μ_Pi)/σ_Pi - (Q_ij - μ_Qi)/σ_Qi)² ]

    Invariant to affine illumination changes (scale + offset).
    Range [0, 4] theoretically — 0 for perfect match, 2 for uncorrelated.

    Args:
        P_ref, P_def: Patch tensors, same shape.
        eps:          Numerical stability for std division.

    Returns:
        scalar mean ZNSSD.
    """
    N = P_ref.shape[0]
    P_ref = P_ref.reshape(N, -1)  # (N, P²)
    P_def = P_def.reshape(N, -1)  # (N, P²)

    # Zero-mean normalization per patch
    mu_ref = P_ref.mean(dim=-1, keepdim=True)
    sigma_ref = P_ref.std(dim=-1, keepdim=True) + eps
    mu_def = P_def.mean(dim=-1, keepdim=True)
    sigma_def = P_def.std(dim=-1, keepdim=True) + eps

    P_ref_norm = (P_ref - mu_ref) / sigma_ref
    P_def_norm = (P_def - mu_def) / sigma_def

    # Per-patch SSD of normalized values
    znssd_per_patch = ((P_ref_norm - P_def_norm) ** 2).sum(dim=-1)  # (N,)
    return znssd_per_patch.mean()


# =========================================================================
# Deformation smoothness regularization
# =========================================================================

def deformation_smoothness_loss(
    deformation_net: nn.Module,  # DeformationNetwork Φ(x, t)
    x: torch.Tensor,             # (N, 3) surface points
    t: torch.Tensor,             # (N, 1) time
) -> torch.Tensor:
    """||∇_x Φ(x,t)||_F^2 — penalize high-frequency spatial variation.

    Computes the full 3×3 Jacobian of Φ w.r.t. x via autograd,
    then returns the mean squared Frobenius norm.

    Args:
        deformation_net: DeformationNetwork instance.
        x:               (N, 3) surface points (will set requires_grad).
        t:               (N, 1) time values.

    Returns:
        scalar smoothness loss.
    """
    x.requires_grad_(True)
    phi = deformation_net(x, t)  # (N, 3)

    grad_u = torch.autograd.grad(
        outputs=phi[:, 0].sum(), inputs=x,
        create_graph=True, retain_graph=True,
    )[0]  # (N, 3)
    grad_v = torch.autograd.grad(
        outputs=phi[:, 1].sum(), inputs=x,
        create_graph=True, retain_graph=True,
    )[0]  # (N, 3)
    grad_w = torch.autograd.grad(
        outputs=phi[:, 2].sum(), inputs=x,
        create_graph=True, retain_graph=True,
    )[0]  # (N, 3)

    # ||∇Φ||_F² = Σ_{i∈{u,v,w}} Σ_{j∈{x,y,z}} (∂Φ_i/∂x_j)²
    loss = (
        (grad_u ** 2).sum(dim=-1).mean() +
        (grad_v ** 2).sum(dim=-1).mean() +
        (grad_w ** 2).sum(dim=-1).mean()
    )
    return loss
