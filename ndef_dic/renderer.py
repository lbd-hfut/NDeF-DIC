"""
Differentiable surface renderer for NDeF-DIC.

Replaces NeRF's volumetric rendering with efficient surface rendering:
  1. Generate camera rays
  2. Sphere-trace along each ray to find the SDF zero-crossing
  3. Query intensity and deformation at the surface point
  4. Project deformed point back to camera
  5. Apply appearance correction

This is O(1) per ray instead of O(N_samples).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Ray generation
# ---------------------------------------------------------------------------

def generate_rays(
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    H: int,
    W: int,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate ray origins and directions for all pixels of a camera.

    Args:
        K: (3, 3) intrinsic matrix.
           [fx,  0, cx]
           [ 0, fy, cy]
           [ 0,  0,  1]
        R: (3, 3) world-to-camera rotation matrix.
        t: (3,) world-to-camera translation vector.
        H, W: image height and width.

    Returns:
        rays_o: (H*W, 3) ray origins in world coordinates.
        rays_d: (H*W, 3) normalized ray directions in world coordinates.
    """
    # Pixel grid
    i, j = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )  # i: (H, W), j: (H, W)

    # Camera coordinates of each pixel
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (j - cx) / fx  # (H, W)
    y_cam = (i - cy) / fy  # (H, W)
    z_cam = torch.ones_like(x_cam)  # looking down +Z in camera frame

    dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (H, W, 3)
    dirs_cam = F.normalize(dirs_cam, p=2, dim=-1)          # (H, W, 3)

    # Transform directions to world: d_world = R^T @ d_cam
    R_w2c = R  # world to camera
    R_c2w = R_w2c.T  # camera to world
    dirs_world = dirs_cam @ R_c2w  # (H, W, 3)
    dirs_world = F.normalize(dirs_world, p=2, dim=-1)

    # Camera center in world: C = -R^T @ t
    cam_center = -(R_c2w @ t)  # (3,)
    rays_o = cam_center.expand(H, W, 3)  # (H, W, 3)

    # Flatten
    rays_o = rays_o.reshape(-1, 3)  # (H*W, 3)
    rays_d = dirs_world.reshape(-1, 3)  # (H*W, 3)

    return rays_o, rays_d


def generate_rays_for_pixels(
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    pixels_uv: torch.Tensor,
    H: int,
    W: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays for specific pixel coordinates (for batched training).

    Args:
        K, R, t: Camera parameters.
        pixels_uv: (N, 2) pixel coordinates (col, row) — (u, v).
        H, W: Image dimensions (for bounds checking only).
    Returns:
        rays_o: (N, 3), rays_d: (N, 3).
    """
    u = pixels_uv[:, 0]  # (N,)
    v = pixels_uv[:, 1]  # (N,)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = torch.ones_like(x_cam)

    dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (N, 3)
    dirs_cam = F.normalize(dirs_cam, p=2, dim=-1)

    R_c2w = R.T
    dirs_world = dirs_cam @ R_c2w
    dirs_world = F.normalize(dirs_world, p=2, dim=-1)

    cam_center = -(R_c2w @ t)
    rays_o = cam_center.expand(len(pixels_uv), 3)

    return rays_o, dirs_world


# ---------------------------------------------------------------------------
# Sphere tracing
# ---------------------------------------------------------------------------

def sphere_trace(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    sdf_network: nn.Module,
    max_iters: int = 20,
    epsilon: float = 1e-4,
    t_min: float = 0.01,
    t_max: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable sphere tracing to find ray-SDF intersections.

    Algorithm:
      t = t_min
      for _ in range(max_iters):
          x = rays_o + t * rays_d
          s = SDF(x)
          t = t + |s|
      hit = |S(x_final)| < epsilon

    This is naturally differentiable — gradients flow through the loop.

    Args:
        rays_o: (N, 3) ray origins.
        rays_d: (N, 3) normalized ray directions.
        sdf_network: SDF network, S(x) -> s.
        max_iters: Maximum sphere tracing steps.
        epsilon: Convergence threshold.
        t_min: Minimum travel distance.
        t_max: Maximum travel distance (clamp).

    Returns:
        x_surface: (N, 3) intersection points.
        hit_mask: (N,) bool — True if converged to surface.
        t_values: (N, 1) distance along each ray.
    """
    N = rays_o.shape[0]
    device = rays_o.device

    t = torch.full((N, 1), t_min, device=device)
    x = rays_o + t * rays_d

    for _ in range(max_iters):
        s = sdf_network(x)  # (N, 1)
        t = t + s.abs()
        t = torch.clamp(t, min=t_min, max=t_max)
        x = rays_o + t * rays_d

    # Convergence check
    with torch.no_grad():
        s_final = sdf_network(x)
        hit_mask = (s_final.abs() < epsilon).squeeze(-1)  # (N,)
        # Also reject rays that hit t_max (likely missed)
        hit_mask = hit_mask & (t.squeeze(-1) < t_max - 0.1)

    return x, hit_mask, t


def sphere_trace_with_grad(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    sdf_network: nn.Module,
    max_iters: int = 20,
    epsilon: float = 1e-4,
    t_min: float = 0.01,
    t_max: float = 100.0,
    use_implicit_grad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sphere tracing with optional implicit differentiation at the surface.

    Standard sphere tracing: gradients flow through the entire iterative
    process. This can be memory-intensive.

    Implicit differentiation: Once the surface point is found, the gradient
    ∂x/∂θ is computed analytically using the SDF gradient at the surface:
      ∂x/∂θ = -n / (n·v) * ∂S/∂θ
    where n = ∇S(x) and v is the ray direction.

    Args:
        use_implicit_grad: If True, use the implicit gradient formulation
                          at the final step for memory efficiency.

    Returns:
        x_surface: (N, 3) intersection points.
        hit_mask: (N,) bool.
    """
    if not use_implicit_grad:
        x, hit_mask, _ = sphere_trace(rays_o, rays_d, sdf_network,
                                       max_iters, epsilon, t_min, t_max)
        return x, hit_mask

    # Standard tracing to find surface point
    with torch.no_grad():
        x_approx, hit_mask, t_vals = sphere_trace(
            rays_o, rays_d, sdf_network, max_iters, epsilon, t_min, t_max
        )

    # Implicit refinement: Newton step to improve surface hit
    # and get gradients through the implicit function theorem
    x_surface = x_approx.clone().requires_grad_(True)

    s = sdf_network(x_surface)  # (N, 1)

    # Gradient of SDF at surface
    grad_s = torch.autograd.grad(
        outputs=s,
        inputs=x_surface,
        grad_outputs=torch.ones_like(s),
        create_graph=True,
        retain_graph=True,
    )[0]  # (N, 3)

    # Normal direction (normalized SDF gradient)
    n = F.normalize(grad_s, p=2, dim=-1)  # (N, 3)

    # Implicit surface correction: project onto zero level-set
    correction = -s * n  # (N, 3)
    x_surface = x_surface + correction

    return x_surface, hit_mask


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_points(
    x_world: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D world points to camera pixel coordinates.

    Args:
        x_world: (N, 3) points in world coordinates.
        K: (3, 3) intrinsics.
        R: (3, 3) world-to-camera rotation.
        t: (3,) world-to-camera translation.

    Returns:
        uv: (N, 2) pixel coordinates (col, row).
        depth: (N,) depth in camera frame (positive if in front).
    """
    # World to camera: x_cam = R @ x_world + t
    x_cam = x_world @ R.T + t.unsqueeze(0)  # (N, 3)

    # Perspective projection
    uv_h = x_cam @ K.T  # (N, 3)
    depth = uv_h[:, 2]  # (N,)
    uv = uv_h[:, :2] / depth.unsqueeze(-1).clamp(min=1e-8)  # (N, 2)

    return uv, depth


# ---------------------------------------------------------------------------
# Full surface renderer
# ---------------------------------------------------------------------------

class SurfaceRenderer:
    """
    Differentiable surface rendering pipeline.

    Combines ray generation, sphere tracing, deformation query,
    intensity query, projection, and appearance correction.

    Usage:
        renderer = SurfaceRenderer(sdf_net, intensity_net, deform_net, appearance_net)
        rendered, valid = renderer.render(camera_params, t, pixels_uv)
    """

    def __init__(
        self,
        sdf_network: nn.Module,
        intensity_field: nn.Module,
        deformation_field: nn.Module,
        appearance_embedding: nn.Module,
        config: "NDeFDICConfig" = None,
    ):
        self.sdf = sdf_network
        self.intensity = intensity_field
        self.deformation = deformation_field
        self.appearance = appearance_embedding

        if config is not None:
            self.max_iters = config.sdf.sphere_trace_iters
            self.eps = config.sdf.sphere_trace_eps
        else:
            self.max_iters = 20
            self.eps = 1e-4

    def render(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t_vec: torch.Tensor,
        camera_id: int,
        load_step: float,
        pixels_uv: torch.Tensor,
    ) -> dict:
        """
        Render a batch of pixels for one camera at one load step.

        Args:
            K: (3, 3) intrinsics.
            R: (3, 3) world-to-camera rotation.
            t_vec: (3,) world-to-camera translation.
            camera_id: int, camera index for appearance embedding.
            load_step: float ∈ [0, 1], normalized load step.
            pixels_uv: (N, 2) pixel coordinates to render.

        Returns:
            dict with keys:
              'intensity': (N, 1) rendered grayscale values.
              'hit_mask': (N,) bool — rays that hit the surface.
              'proj_uv': (N, 2) where surface points project in this camera.
              'depth': (N,) depth in camera frame.
              'x_surface': (N, 3) reference surface points hit.
              'x_deformed': (N, 3) deformed surface points.
              'valid': (N,) bool — pixels valid for loss (hit + in front).
        """
        N = pixels_uv.shape[0]
        device = K.device

        # --- 1. Generate rays for requested pixels ---
        rays_o, rays_d = generate_rays_for_pixels(
            K, R, t_vec, pixels_uv, H=0, W=0  # H,W unused
        )

        # --- 2. Sphere trace to find reference surface ---
        x_surface, hit_mask, t_vals = sphere_trace(
            rays_o, rays_d, self.sdf,
            max_iters=self.max_iters,
            epsilon=self.eps,
        )
        # x_surface: (N, 3), hit_mask: (N,)

        # --- 3. Query deformation field ---
        t_tensor = torch.full((N, 1), load_step, device=device)
        uvw = self.deformation(x_surface, t_tensor)  # (N, 3)
        x_deformed = x_surface + uvw  # (N, 3)

        # --- 4. Project deformed points to this camera ---
        proj_uv, depth = project_points(x_deformed, K, R, t_vec)

        # --- 5. Query intensity at reference surface points ---
        # (brightness constancy: intensity follows material point)
        base_gray = self.intensity(x_surface)  # (N, 1)

        # --- 6. Apply appearance correction ---
        cam_ids = torch.full((N,), camera_id, device=device, dtype=torch.long)
        rendered_gray = self.appearance.correct(base_gray, cam_ids)

        # --- 7. Build validity mask ---
        in_front = depth > 0  # (N,)
        valid = hit_mask & in_front

        return {
            "intensity": rendered_gray,
            "hit_mask": hit_mask,
            "proj_uv": proj_uv,
            "depth": depth,
            "x_surface": x_surface,
            "x_deformed": x_deformed,
            "valid": valid,
        }

    def render_reference_only(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        t_vec: torch.Tensor,
        camera_id: int,
        pixels_uv: torch.Tensor,
    ) -> dict:
        """
        Render reference frame (no deformation).

        This is a fast path for Stage 2 (intensity field pre-training).
        Deformation field is bypassed; uvw = 0.
        """
        N = pixels_uv.shape[0]
        device = K.device

        rays_o, rays_d = generate_rays_for_pixels(K, R, t_vec, pixels_uv, 0, 0)

        x_surface, hit_mask, _ = sphere_trace(
            rays_o, rays_d, self.sdf,
            max_iters=self.max_iters,
            epsilon=self.eps,
        )

        # Reference: no deformation, projection is from reference surface
        proj_uv, depth = project_points(x_surface, K, R, t_vec)

        base_gray = self.intensity(x_surface)
        cam_ids = torch.full((N,), camera_id, device=device, dtype=torch.long)
        rendered_gray = self.appearance.correct(base_gray, cam_ids)

        in_front = depth > 0
        valid = hit_mask & in_front

        return {
            "intensity": rendered_gray,
            "hit_mask": hit_mask,
            "proj_uv": proj_uv,
            "depth": depth,
            "x_surface": x_surface,
            "valid": valid,
        }
