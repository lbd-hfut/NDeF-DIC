"""
Cylinder Multi-View DIC Simulation.

Generates synthetic multi-view speckle images of a cylindrical specimen
with ground-truth deformation, for validating the NDF-DIC pipeline.

Geometry:
  - Cylinder axis: world Y
  - Cameras uniformly distributed around the cylinder in the XZ plane
  - All cameras look at the cylinder center

Output layout (compatible with MultiCamDataset):
  case/CylinderDIC/
  ├── images/
  │   ├── cam_0/
  │   │   └── ref.bmp          # Reference image
  │   │   └── 001.bmp          # Deformed image (step 1)
  │   ├── cam_1/
  │   │   └── ...
  │   └── ...
  ├── calibration/
  │   ├── cameras.mat          # COLMAP-format camera parameters
  │   └── points3D.mat         # Sparse 3D surface points
  └── ground_truth/
      ├── camera_intrinsics.npy               # (C, 3, 3) theoretical K
      ├── camera_rotations.npy                # (C, 3, 3) theoretical world-to-camera R
      ├── camera_translations.npy             # (C, 3) theoretical world-to-camera t
      ├── theoretical_surface_points.npy      # (N, 3) reference morphology points
      ├── theoretical_deformation_field_*.npy # (N, 3) deformation field per step
      └── meta.json            # Simulation parameters
"""

import os
import sys
import json
import argparse
import numpy as np
from scipy.io import savemat
from scipy.ndimage import gaussian_filter
import imageio.v3 as iio
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, List


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class CylinderSimConfig:
    """Simulation configuration."""

    # ---- Output ----
    output_dir: str = "case/CylinderDIC"

    # ---- Cylinder geometry ----
    cylinder_radius: float = 80.0       # mm
    cylinder_height: float = 120.0      # mm
    surface_type: str = "cylinder"       # "cylinder" | "plane"

    # ---- Camera array ----
    num_cameras: int = 12
    working_distance: float = 300.0     # mm, from cylinder center to camera
    image_width: int = 1440
    image_height: int = 1080
    pixel_size: float = 3.45e-3         # mm/pixel
    focal_length: float = 8.0           # mm
    k1: float = 0.0                     # Radial distortion (0 = ideal pinhole)
    k2: float = 0.0

    # ---- Speckle pattern ----
    speckle_image: str = ""             # Path to speckle image; empty = procedural 3D grains
    num_surface_points: int = 15_000_000
    ground_truth_num_points: int = 100_000
    surface_coverage_angle: float = 360.0  # Degrees of circumference covered
    speckle_physical_size: float = 0.0      # mm — (texture mode only) set 0 = no tiling, >0 = tile
    num_speckle_grains: int = 80_000        # Number of Gaussian grains on surface (3D procedural mode)
    grain_sigma_mean: float = 0.25          # mm — mean grain radius on surface

    # ---- Rendering ----
    gaussian_sigma: float = 0.3        # px, lens blur
    noise_std: float = 0.0             # gray levels, sensor noise
    intensity_range: Tuple[float, float] = (30, 250)  # output gray range
    gamma: float = 0.55                 # Gamma correction: < 1 brightens midtones

    # ---- Deformation ----
    deformation_type: str = "expansion"  # "none" | "expansion" | "torsion" | "compression" | "combined"
    deformation_magnitude: float = 0.5  # expansion: radial Δr (mm); torsion: degrees; compression: % strain
    num_deformed_steps: int = 1

    # ---- Random seed ----
    seed: int = 42


# =========================================================================
# Camera geometry
# =========================================================================

def build_camera_array(config: CylinderSimConfig):
    """
    Build cameras uniformly distributed around the cylinder.

    Camera i is at angle θ_i = 2π * i / N in the XZ plane,
    at distance D = working_distance from the cylinder center,
    looking at the origin.

    Returns:
        K_list:      List of (3, 3) intrinsic matrices
        R_list:      List of (3, 3) world-to-camera rotation matrices
        t_list:      List of (3, 1) translation vectors (world origin in camera frame)
        dist_list:   List of (5,) distortion coefficient arrays
        cam_centers: (N, 3) camera center positions in world coords
    """
    N = config.num_cameras
    D = config.cylinder_radius + config.working_distance

    # Intrinsics
    fx = config.focal_length / config.pixel_size
    fy = fx
    cx = config.image_width / 2.0
    cy = config.image_height / 2.0

    K_base = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0,  0,  1]], dtype=np.float64)
    dist = np.array([config.k1, config.k2, 0.0, 0.0, 0.0], dtype=np.float64)

    K_list, R_list, t_list, dist_list = [], [], [], []
    cam_centers = []

    for i in range(N):
        theta = 2.0 * np.pi * i / N  # Camera azimuthal angle

        # Camera center in world coords (XZ plane)
        C = np.array([D * np.cos(theta), 0.0, D * np.sin(theta)])

        # ---- Build world-to-camera rotation ----
        # Camera forward (looking at origin): -C / |C|
        forward = -C / np.linalg.norm(C)

        # Camera right: forward × world_up
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)

        # Camera up: right × forward
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # R maps world → camera:
        # Row 0 = camera X axis in world  → right
        # Row 1 = camera Y axis in world  → -up (image Y goes down)
        # Row 2 = camera Z axis in world  → forward (looking at origin)
        R = np.stack([right, -up, forward], axis=0)

        # t = -R @ C  (world origin expressed in camera frame)
        t = -R @ C.reshape(3, 1)

        K_list.append(K_base.copy())
        R_list.append(R)
        t_list.append(t)
        dist_list.append(dist.copy())
        cam_centers.append(C)

    return K_list, R_list, t_list, dist_list, np.array(cam_centers)


# =========================================================================
# Speckle surface generation — procedural 3D grains
# =========================================================================

def generate_cylinder_surface(config: CylinderSimConfig, rng: np.random.Generator):
    """
    Generate speckle points directly on the cylinder surface in 3D.

    Strategy:
      1. Place N_grains random Gaussian grains on the cylinder surface.
         Each grain has a 3D center, isotropic sigma (mm), and amplitude.
      2. Build a spatial binning grid in (θ, y) space for efficient lookup.
      3. Sample M surface points uniformly on the cylinder.
      4. For each point, sum contributions from nearby grains (using bins).

    Returns:
        points:      (N, 3) world coordinates
        intensities: (N,)  [0, 1] float
        normals:     (N, 3) unit surface normals
        grains:      dict with grain parameters
    """
    N = config.num_surface_points
    R = config.cylinder_radius
    H = config.cylinder_height
    angle_range_rad = np.deg2rad(config.surface_coverage_angle)

    # ---- Grain parameters ----
    # Target grain size in image: ~4 pixels at 0.207 mm/px → ~0.83 mm physical.
    # Gaussian sigma ≈ grain_radius / 2 ≈ 0.2 mm.
    grain_sigma_mean = 0.25   # mm (geodesic distance on surface)
    grain_sigma_std = 0.08
    n_grains = config.num_speckle_grains

    # ---- Generate grain centers on cylinder surface ----
    print(f"  Placing {n_grains:,} speckle grains on cylinder surface...")
    grain_theta = rng.uniform(-np.pi, np.pi, n_grains)
    grain_y = rng.uniform(-H / 2, H / 2, n_grains)
    grain_sigma = np.abs(rng.normal(grain_sigma_mean, grain_sigma_std, n_grains))
    grain_sigma = np.clip(grain_sigma, 0.1, 0.6)  # clamp extreme sizes
    grain_amp = rng.uniform(0.3, 1.0, n_grains)

    # 3D positions of grain centers
    grain_x = R * np.cos(grain_theta)
    grain_z = R * np.sin(grain_theta)
    grain_centers_3d = np.stack([grain_x, grain_y, grain_z], axis=1)  # (n_grains, 3)

    # ---- Spatial binning in (θ, y) space ----
    # Bin size: 3× max grain sigma (covers 3σ support on each side)
    max_sigma = grain_sigma.max()
    bin_size_theta = 3.0 * max_sigma / R  # angular bin size
    bin_size_y = 3.0 * max_sigma          # vertical bin size (mm)

    # Handle case where bin is too small
    bin_size_theta = max(bin_size_theta, 0.01)
    bin_size_y = max(bin_size_y, 1.0)

    n_bins_theta = max(1, int(np.ceil(2 * np.pi / bin_size_theta)))
    n_bins_y = max(1, int(np.ceil(H / bin_size_y)))

    # Assign each grain to its bin
    grain_bin_theta = ((grain_theta + np.pi) / (2 * np.pi) * n_bins_theta).astype(int)
    grain_bin_theta = np.clip(grain_bin_theta, 0, n_bins_theta - 1)
    grain_bin_y = ((grain_y + H / 2) / H * n_bins_y).astype(int)
    grain_bin_y = np.clip(grain_bin_y, 0, n_bins_y - 1)

    # Build bin → grain index list
    bins = [[] for _ in range(n_bins_theta * n_bins_y)]
    for g in range(n_grains):
        idx = grain_bin_theta[g] + grain_bin_y[g] * n_bins_theta
        bins[idx].append(g)

    print(f"  Bins: {n_bins_theta}×{n_bins_y} = {len(bins)} bins, "
          f"avg {n_grains / len(bins):.1f} grains/bin")

    # ---- Sample surface points and compute intensities ----
    block_size = 500_000
    num_blocks = int(np.ceil(N / block_size))
    points_list, intensity_list = [], []

    for block in range(num_blocks):
        end = min((block + 1) * block_size, N)
        n_block = end - block * block_size

        theta = (rng.random(n_block) - 0.5) * angle_range_rad
        y_p = (rng.random(n_block) - 0.5) * H

        # 3D coordinates
        x_p = R * np.cos(theta)
        z_p = R * np.sin(theta)
        pts = np.stack([x_p, y_p, z_p], axis=1)

        # Find which bin each point is in
        p_bin_t = ((theta + np.pi) / (2 * np.pi) * n_bins_theta).astype(int)
        p_bin_t = np.clip(p_bin_t, 0, n_bins_theta - 1)
        p_bin_y = ((y_p + H / 2) / H * n_bins_y).astype(int)
        p_bin_y = np.clip(p_bin_y, 0, n_bins_y - 1)

        # Compute intensity for each point by summing nearby grains
        I = np.zeros(n_block, dtype=np.float64)

        # For each point, check grains in its bin + adjacent bins
        for dt in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                bt = (p_bin_t + dt) % n_bins_theta
                by = np.clip(p_bin_y + dy, 0, n_bins_y - 1)
                bin_idx = bt + by * n_bins_theta

                # For each unique bin, process the grains within it
                unique_bins = np.unique(bin_idx)
                for ub in unique_bins:
                    grain_ids = bins[ub]
                    if not grain_ids:
                        continue
                    mask = bin_idx == ub
                    p_idx = np.where(mask)[0]

                    g_theta = grain_theta[grain_ids]  # (n_grains_in_bin,)
                    g_y = grain_y[grain_ids]
                    g_sig = grain_sigma[grain_ids]
                    g_amp = grain_amp[grain_ids]

                    for j, gid in enumerate(grain_ids):
                        # Angular distance (handle wrap-around)
                        dtheta = np.abs(theta[p_idx] - g_theta[j])
                        dtheta = np.minimum(dtheta, 2 * np.pi - dtheta)
                        # Geodesic distance on cylinder
                        dx = dtheta * R
                        dy_dist = y_p[p_idx] - g_y[j]
                        dist_sq = dx ** 2 + dy_dist ** 2
                        sig_sq = g_sig[j] ** 2
                        I[p_idx] += g_amp[j] * np.exp(-dist_sq / (2 * sig_sq))

        intensity_list.append(I)
        points_list.append(pts)

        if (block + 1) % 10 == 0:
            print(f"  Surface generation: {end}/{N} points")

    points = np.concatenate(points_list, axis=0)
    intensities = np.concatenate(intensity_list, axis=0)

    # Normalize intensity to [0, 1]
    i_min, i_max = intensities.min(), intensities.max()
    if i_max > i_min:
        intensities = (intensities - i_min) / (i_max - i_min)

    # Compute normals for each point (points outward from cylinder axis)
    normals = points.copy()
    normals[:, 1] = 0  # zero out Y component → radial direction in XZ plane
    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(nrm, 1e-8)

    grains = {
        "centers_3d": grain_centers_3d,
        "theta": grain_theta, "y": grain_y,
        "sigma": grain_sigma, "amp": grain_amp,
    }

    return points, intensities, normals, grains


def _generate_speckle_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """(保留，用于自动生成 2D 散斑图的后备方案)"""
    n_blobs = 20_000
    tex = np.zeros((h, w), dtype=np.float32)
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    for i in range(n_blobs):
        cx = rng.integers(0, w)
        cy = rng.integers(0, h)
        sigma = rng.uniform(1.5, 6.0)
        amp = rng.uniform(0.3, 1.0)
        radius = int(np.ceil(3 * sigma))
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        local_x, local_y = x_grid[y0:y1, x0:x1], y_grid[y0:y1, x0:x1]
        tex[y0:y1, x0:x1] += amp * np.exp(-((local_x - cx) ** 2 + (local_y - cy) ** 2) / (2 * sigma ** 2))
    tex = tex - tex.min()
    tex = tex / tex.max() * 255.0
    return tex.astype(np.uint8)


# =========================================================================
# Image rendering
# =========================================================================

def render_images(
    points: np.ndarray,
    intensities: np.ndarray,
    normals: np.ndarray,
    K_list: List[np.ndarray],
    R_list: List[np.ndarray],
    t_list: List[np.ndarray],
    dist_list: List[np.ndarray],
    cam_centers: np.ndarray,
    config: CylinderSimConfig,
    rng_per_cam: List[np.random.Generator],
) -> List[np.ndarray]:
    """
    Render speckle images for all cameras via scatter-to-pixel accumulation.

    Pipeline:
      1. Back-face culling: discard points whose normal faces away from camera.
      2. Project to camera (with optional distortion).
      3. Accumulate intensities into pixel grid.
      4. Normalize, gamma-correct, blur, add noise.

    Returns:
        List of (H, W) uint8 images, one per camera.
    """
    N_cam = len(K_list)
    H, W = config.image_height, config.image_width
    images = []

    for cam_id in range(N_cam):
        print(f"  Rendering camera {cam_id + 1}/{N_cam}...")

        K = K_list[cam_id]
        R = R_list[cam_id]
        t = t_list[cam_id]
        C = cam_centers[cam_id]  # (3,) camera center in world coords

        # ---- Back-face culling ----
        # View direction from each point toward camera
        view_dir = C.reshape(1, 3) - points  # (N, 3)
        view_dir = view_dir / np.linalg.norm(view_dir, axis=1, keepdims=True)
        cos_angle = np.sum(normals * view_dir, axis=1)  # (N,)
        front_facing = cos_angle > 0.05  # ~87° max grazing angle

        # ---- Project ----
        P_cam = R @ points.T + t.reshape(3, 1)
        Z = P_cam[2, :]

        # Discard points behind camera or on back face
        valid_geo = (Z > 1e-6) & front_facing
        if valid_geo.sum() == 0:
            images.append(np.zeros((H, W), dtype=np.uint8))
            continue

        P_cam = P_cam[:, valid_geo]

        # Normalized image coordinates
        xn = P_cam[0, :] / P_cam[2, :]
        yn = P_cam[1, :] / P_cam[2, :]

        # ---- Apply radial distortion (optional) ----
        k1, k2 = dist_list[cam_id][0], dist_list[cam_id][1]
        if abs(k1) > 1e-12 or abs(k2) > 1e-12:
            r2 = xn ** 2 + yn ** 2
            radial = 1.0 + k1 * r2 + k2 * r2 ** 2
            xn_dist = xn * radial
            yn_dist = yn * radial
        else:
            xn_dist = xn
            yn_dist = yn

        # Pixel coordinates
        u = K[0, 0] * xn_dist + K[0, 1] * yn_dist + K[0, 2]
        v = K[1, 0] * xn_dist + K[1, 1] * yn_dist + K[1, 2]

        # Filter points within image bounds
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[valid]
        v = v[valid]
        I_in = intensities[valid_geo][valid]

        # ---- Bilinear splatting to pixel grid ----
        # Each point contributes to 4 neighboring pixels with bilinear weights.
        # This eliminates "dead pixel" artifacts from nearest-neighbor rounding.
        u0 = np.floor(u).astype(int)
        v0 = np.floor(v).astype(int)
        u1 = u0 + 1
        v1 = v0 + 1

        wu1 = u - u0.astype(float)   # weight for u1 (0..1)
        wv1 = v - v0.astype(float)   # weight for v1 (0..1)
        wu0 = 1.0 - wu1
        wv0 = 1.0 - wv1

        # 4 corners with bilinear weights
        weights = [
            (u0, v0, wu0 * wv0),  # top-left
            (u1, v0, wu1 * wv0),  # top-right
            (u0, v1, wu0 * wv1),  # bottom-left
            (u1, v1, wu1 * wv1),  # bottom-right
        ]

        img_flat = np.zeros(W * H, dtype=np.float64)
        cnt_flat = np.zeros(W * H, dtype=np.float64)

        for uc, vc, wt in weights:
            in_bounds = (uc >= 0) & (uc < W) & (vc >= 0) & (vc < H)
            if in_bounds.sum() == 0:
                continue
            idx = vc[in_bounds] * W + uc[in_bounds]
            img_flat += np.bincount(idx, weights=I_in[in_bounds] * wt[in_bounds],
                                    minlength=W * H)
            cnt_flat += np.bincount(idx, weights=wt[in_bounds], minlength=W * H)

        img = np.divide(img_flat, cnt_flat, out=np.zeros_like(img_flat),
                        where=cnt_flat > 1e-12)
        img = img.reshape(H, W)

        # ---- Post-process ----
        # Linear stretch from [img_min, img_max] → normalized [0, 1]
        covered = cnt_flat.reshape(H, W) > 1e-12
        img_min = img[covered].min() if covered.sum() > 0 else 0.0
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

        # Gamma correction: < 1 brightens midtones
        if config.gamma != 1.0:
            img = np.maximum(img, 0.0) ** config.gamma

        # Scale to target output range
        img = img * (config.intensity_range[1] - config.intensity_range[0])
        img = img + config.intensity_range[0]

        # Gaussian blur
        if config.gaussian_sigma > 0:
            img = gaussian_filter(img, sigma=config.gaussian_sigma)

        # Sensor noise
        if config.noise_std > 0:
            noise = rng_per_cam[cam_id].normal(0, config.noise_std, (H, W))
            img = img + noise

        img = np.clip(np.round(img), 0, 255).astype(np.uint8)
        images.append(img)

    return images


# =========================================================================
# Deformation
# =========================================================================

def apply_deformation(
    points_ref: np.ndarray,
    config: CylinderSimConfig,
) -> List[np.ndarray]:
    """
    Apply ground-truth deformation to surface points.

    Deformation types:
      "expansion":   r → r + Δr, uniform radial expansion outward.
      "torsion":     θ → θ + Δθ * (y / H_half),  shear along cylinder axis.
      "compression": y → y * (1 - ε),  r → r * (1 + ν·ε)
      "combined":    torsion + compression applied together.
      "none":        identity (useful for generating reference-only data).

    Returns:
        List of (N, 3) deformed point clouds, one per loading step.
    """
    R = config.cylinder_radius
    H = config.cylinder_height
    results = []

    for step in range(1, config.num_deformed_steps + 1):
        frac = step / config.num_deformed_steps
        pts = points_ref.copy()

        if config.deformation_type == "none":
            pass

        elif config.deformation_type == "expansion":
            # Uniform radial expansion: each point moves outward along its radial direction
            dr = config.deformation_magnitude * frac  # Δr in mm
            # Compute radial direction in XZ plane for each point
            r = np.sqrt(pts[:, 0] ** 2 + pts[:, 2] ** 2)  # current radius
            r_safe = np.maximum(r, 1e-6)
            # Scale (x, z) proportionally to push outward
            scale = 1.0 + dr / r_safe
            pts[:, 0] = pts[:, 0] * scale
            pts[:, 2] = pts[:, 2] * scale

        elif config.deformation_type == "torsion":
            # Rotation angle proportional to height
            angle_max = np.deg2rad(config.deformation_magnitude)
            theta_rot = angle_max * frac * pts[:, 1] / (H / 2)

            # Rotate (x, z) coordinates
            cos_t = np.cos(theta_rot)
            sin_t = np.sin(theta_rot)
            x_new = pts[:, 0] * cos_t - pts[:, 2] * sin_t
            z_new = pts[:, 0] * sin_t + pts[:, 2] * cos_t
            pts[:, 0] = x_new
            pts[:, 2] = z_new

        elif config.deformation_type == "compression":
            strain = config.deformation_magnitude / 100.0  # % → fraction
            eps = strain * frac
            nu = 0.35  # Poisson ratio for rubber-like materials

            pts[:, 1] = pts[:, 1] * (1.0 - eps)

            # Radial expansion via Poisson effect
            scale = 1.0 + nu * eps
            # Scale x and z (in-plane coordinates)
            pts[:, 0] = pts[:, 0] * scale
            pts[:, 2] = pts[:, 2] * scale

        elif config.deformation_type == "combined":
            strain = config.deformation_magnitude / 100.0
            eps = strain * frac
            nu = 0.35
            angle_max = np.deg2rad(config.deformation_magnitude * 0.5)

            # Compression
            pts[:, 1] = pts[:, 1] * (1.0 - eps)
            scale = 1.0 + nu * eps
            pts[:, 0] = pts[:, 0] * scale
            pts[:, 2] = pts[:, 2] * scale

            # Torsion
            theta_rot = angle_max * frac * pts[:, 1] / (H / 2)
            cos_t = np.cos(theta_rot)
            sin_t = np.sin(theta_rot)
            x_new = pts[:, 0] * cos_t - pts[:, 2] * sin_t
            z_new = pts[:, 0] * sin_t + pts[:, 2] * cos_t
            pts[:, 0] = x_new
            pts[:, 2] = z_new

        else:
            raise ValueError(f"Unknown deformation type: {config.deformation_type}")

        results.append(pts)

    return results


def _compute_cylinder_normals(points: np.ndarray) -> np.ndarray:
    """Compute outward normals for points on a cylinder (radial direction in XZ)."""
    n = points.copy()
    n[:, 1] = 0.0
    nrm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(nrm, 1e-8)


# =========================================================================
# Output
# =========================================================================

def save_ground_truth_outputs(
    config: CylinderSimConfig,
    K_list: List[np.ndarray],
    R_list: List[np.ndarray],
    t_list: List[np.ndarray],
    dist_list: List[np.ndarray],
    cam_centers: np.ndarray,
    points_ref: np.ndarray,
    points_def_list: List[np.ndarray],
):
    """Save theoretical camera parameters, surface morphology, and deformation fields."""
    gt_dir = os.path.join(config.output_dir, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)

    N_cam = len(K_list)
    K_arr = np.stack(K_list, axis=0).astype(np.float64)
    R_arr = np.stack(R_list, axis=0).astype(np.float64)
    t_arr = np.stack([t.reshape(3) for t in t_list], axis=0).astype(np.float64)
    dist_arr = np.stack(dist_list, axis=0).astype(np.float64)
    cam_centers_arr = cam_centers.astype(np.float64)
    cam_names = np.array([f"cam_{i}" for i in range(N_cam)])

    n_available = len(points_ref)
    n_save = config.ground_truth_num_points
    if n_save <= 0 or n_save >= n_available:
        gt_indices = np.arange(n_available, dtype=np.int64)
    else:
        gt_indices = np.linspace(0, n_available - 1, n_save, dtype=np.int64)

    points_ref_gt = points_ref[gt_indices]

    # Theoretical camera intrinsics/extrinsics.
    np.save(os.path.join(gt_dir, "camera_intrinsics.npy"), K_arr)
    np.save(os.path.join(gt_dir, "camera_rotations.npy"), R_arr)
    np.save(os.path.join(gt_dir, "camera_translations.npy"), t_arr)
    np.save(os.path.join(gt_dir, "camera_centers.npy"), cam_centers_arr)
    np.save(os.path.join(gt_dir, "camera_distortion.npy"), dist_arr)
    np.savez_compressed(
        os.path.join(gt_dir, "theoretical_camera_parameters.npz"),
        K=K_arr,
        R=R_arr,
        t=t_arr,
        camera_centers=cam_centers_arr,
        distortion=dist_arr,
        cam_names=cam_names,
        image_width=config.image_width,
        image_height=config.image_height,
        coordinate_convention="world_to_camera: X_cam = R @ X_world + t",
    )

    # Theoretical reference morphology/surface points.
    points_ref_f32 = points_ref_gt.astype(np.float32)
    np.save(os.path.join(gt_dir, "ground_truth_sample_indices.npy"), gt_indices)
    np.save(os.path.join(gt_dir, "theoretical_surface_points.npy"), points_ref_f32)
    np.savez_compressed(
        os.path.join(gt_dir, "theoretical_surface_points.npz"),
        points=points_ref_f32,
        surface_type=config.surface_type,
        cylinder_radius=config.cylinder_radius,
        cylinder_height=config.cylinder_height,
        units="mm",
    )

    # Backward-compatible name used by existing scripts.
    np.save(os.path.join(gt_dir, "points_ref.npy"), points_ref_f32)

    for step_idx, pts_def in enumerate(points_def_list):
        step = step_idx + 1
        pts_def_gt = pts_def[gt_indices]
        pts_def_f32 = pts_def_gt.astype(np.float32)
        disp_f32 = (pts_def_gt - points_ref_gt).astype(np.float32)

        # Theoretical deformed morphology and deformation field per step.
        np.save(os.path.join(gt_dir, f"theoretical_deformed_surface_points_step{step:03d}.npy"),
                pts_def_f32)
        np.save(os.path.join(gt_dir, f"theoretical_deformation_field_step{step:03d}.npy"),
                disp_f32)
        np.savez_compressed(
            os.path.join(gt_dir, f"theoretical_deformation_step{step:03d}.npz"),
            points_ref=points_ref_f32,
            points_def=pts_def_f32,
            displacement=disp_f32,
            deformation_type=config.deformation_type,
            deformation_magnitude=config.deformation_magnitude,
            step=step,
            num_deformed_steps=config.num_deformed_steps,
            units="mm",
        )

        # Backward-compatible names used by existing scripts.
        np.save(os.path.join(gt_dir, f"points_def_step{step_idx + 1}.npy"),
                pts_def_f32)
        np.save(os.path.join(gt_dir, f"displacement_step{step_idx + 1}.npy"),
                disp_f32)

    # Meta
    meta = {
        "cylinder_radius": config.cylinder_radius,
        "cylinder_height": config.cylinder_height,
        "num_cameras": config.num_cameras,
        "working_distance": config.working_distance,
        "deformation_type": config.deformation_type,
        "deformation_magnitude": config.deformation_magnitude,
        "speckle_physical_size": config.speckle_physical_size,
        "num_surface_points": config.num_surface_points,
        "ground_truth_num_points_requested": config.ground_truth_num_points,
        "ground_truth_num_points_saved": int(len(gt_indices)),
        "num_deformed_steps": config.num_deformed_steps,
        "ground_truth_files": {
            "camera_intrinsics": "camera_intrinsics.npy",
            "camera_rotations": "camera_rotations.npy",
            "camera_translations": "camera_translations.npy",
            "camera_centers": "camera_centers.npy",
            "camera_distortion": "camera_distortion.npy",
            "camera_parameters_bundle": "theoretical_camera_parameters.npz",
            "sample_indices": "ground_truth_sample_indices.npy",
            "surface_points": "theoretical_surface_points.npy",
            "surface_points_bundle": "theoretical_surface_points.npz",
            "deformed_surface_pattern": "theoretical_deformed_surface_points_step{step:03d}.npy",
            "deformation_field_pattern": "theoretical_deformation_field_step{step:03d}.npy",
            "deformation_bundle_pattern": "theoretical_deformation_step{step:03d}.npz",
        },
    }
    with open(os.path.join(gt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Ground truth saved to {gt_dir}/")


def save_outputs(
    config: CylinderSimConfig,
    images_ref: List[np.ndarray],
    images_def_list: List[List[np.ndarray]],
    K_list: List[np.ndarray],
    R_list: List[np.ndarray],
    t_list: List[np.ndarray],
    dist_list: List[np.ndarray],
    cam_centers: np.ndarray,
    points_ref: np.ndarray,
    points_def_list: List[np.ndarray],
):
    """Save all outputs in the format expected by NDF-DIC."""
    out = config.output_dir
    N_cam = len(images_ref)

    # ---- Images ----
    img_dir = os.path.join(out, "images")
    for cam_id in range(N_cam):
        cam_dir = os.path.join(img_dir, f"cam_{cam_id}")
        os.makedirs(cam_dir, exist_ok=True)

        # Clean old images
        for old in os.listdir(cam_dir):
            if old.endswith('.bmp'):
                os.remove(os.path.join(cam_dir, old))
        # Reference → 001.bmp
        iio.imwrite(os.path.join(cam_dir, "001.bmp"), images_ref[cam_id])

        # Deformed → 002.bmp, 003.bmp, ...
        for step_idx, images_def in enumerate(images_def_list):
            fname = f"{step_idx + 2:03d}.bmp"
            iio.imwrite(os.path.join(cam_dir, fname), images_def[cam_id])

    print(f"  Images saved to {img_dir}/")

    # ---- Camera parameters (COLMAT format) ----
    calib_dir = os.path.join(out, "calibration")
    os.makedirs(calib_dir, exist_ok=True)

    cameras_mat = {
        "num_cameras": N_cam,
        "K_list": np.array(K_list, dtype=object),
        "dist_list": np.array(dist_list, dtype=object),
        "cam_from_world_R": np.array(R_list, dtype=object),
        "cam_from_world_t": np.array([t.reshape(3, 1) for t in t_list], dtype=object),
        "camera_models": np.array(["PINHOLE"] * N_cam, dtype=object),
        "cam_names": np.array([f"cam_{i}" for i in range(N_cam)], dtype=object),
    }
    savemat(os.path.join(calib_dir, "cameras.mat"), cameras_mat)

    # Sparse points: subsample for COLMAP-like output
    n_sparse = min(5000, len(points_ref))
    idx_sparse = np.linspace(0, len(points_ref) - 1, n_sparse, dtype=int)
    savemat(os.path.join(calib_dir, "points3D.mat"),
            {"points3D": points_ref[idx_sparse].astype(np.float64)})

    print(f"  Calibration saved to {calib_dir}/")

    # ---- Ground truth ----
    save_ground_truth_outputs(
        config,
        K_list, R_list, t_list, dist_list, cam_centers,
        points_ref, points_def_list,
    )


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cylinder Multi-View DIC Simulation"
    )
    parser.add_argument("--output_dir", type=str, default="case/CylinderDIC")
    parser.add_argument("--num_cameras", type=int, default=12)
    parser.add_argument("--working_distance", type=float, default=400.0)
    parser.add_argument("--cylinder_radius", type=float, default=80.0)
    parser.add_argument("--cylinder_height", type=float, default=120.0)
    parser.add_argument("--focal_length", type=float, default=8.0)
    parser.add_argument("--image_width", type=int, default=1440)
    parser.add_argument("--image_height", type=int, default=1080)
    parser.add_argument("--num_points", type=int, default=15_000_000)
    parser.add_argument("--ground_truth_points", type=int, default=100_000,
                        help="Number of theoretical surface/deformation points saved to ground_truth.")
    parser.add_argument("--deformation", type=str, default="expansion",
                        choices=["none", "expansion", "torsion", "compression", "combined"])
    parser.add_argument("--deformation_magnitude", type=float, default=0.5)
    parser.add_argument("--speckle_image", type=str, default="",
                        help="Path to speckle pattern image; auto-generate if empty")
    parser.add_argument("--speckle_physical_size", type=float, default=0.0,
                        help="(texture mode) Set 0 = no tiling; >0 for tiling")
    parser.add_argument("--num_speckle_grains", type=int, default=80_000,
                        help="Number of Gaussian grains on surface (3D procedural mode)")
    parser.add_argument("--grain_sigma_mean", type=float, default=0.25,
                        help="Mean grain radius on surface in mm")
    parser.add_argument("--gamma", type=float, default=0.55)
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--gaussian_sigma", type=float, default=0.3)
    parser.add_argument("--k1", type=float, default=0.0)
    parser.add_argument("--k2", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_deformed_steps", type=int, default=1)
    parser.add_argument("--ground_truth_only", action="store_true",
                        help="Only save theoretical ground truth; do not render or overwrite images.")

    args = parser.parse_args()

    config = CylinderSimConfig(
        output_dir=args.output_dir,
        num_cameras=args.num_cameras,
        working_distance=args.working_distance,
        cylinder_radius=args.cylinder_radius,
        cylinder_height=args.cylinder_height,
        focal_length=args.focal_length,
        image_width=args.image_width,
        image_height=args.image_height,
        num_surface_points=args.num_points,
        ground_truth_num_points=args.ground_truth_points,
        deformation_type=args.deformation,
        deformation_magnitude=args.deformation_magnitude,
        speckle_image=args.speckle_image,
        speckle_physical_size=args.speckle_physical_size,
        num_speckle_grains=args.num_speckle_grains,
        grain_sigma_mean=args.grain_sigma_mean,
        gamma=args.gamma,
        noise_std=args.noise_std,
        gaussian_sigma=args.gaussian_sigma,
        k1=args.k1,
        k2=args.k2,
        num_deformed_steps=args.num_deformed_steps,
        seed=args.seed,
    )

    # Resolve speckle image path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if config.speckle_image:
        if not os.path.isabs(config.speckle_image):
            config.speckle_image = os.path.join(script_dir, config.speckle_image)
    else:
        # Auto-detect: look for 002.bmp in SimulationExperiment directory
        default_speckle = os.path.join(script_dir, "..", "SimulationExperiment", "002.bmp")
        if os.path.exists(default_speckle):
            config.speckle_image = os.path.abspath(default_speckle)
            print(f"[INFO] Auto-detected speckle image: {config.speckle_image}")

    print("=" * 60)
    print("Cylinder Multi-View DIC Simulation")
    print("=" * 60)
    print(f"  Cameras:        {config.num_cameras}")
    print(f"  Working dist:   {config.working_distance} mm")
    print(f"  Cylinder:       R={config.cylinder_radius}, H={config.cylinder_height} mm")
    print(f"  Image:          {config.image_width}×{config.image_height}")
    print(f"  Render points:  {config.num_surface_points:,}")
    print(f"  GT save points: {config.ground_truth_num_points:,}")
    print(f"  Deformation:    {config.deformation_type} ({config.deformation_magnitude})")
    print(f"  Output:         {config.output_dir}")
    print()

    # ---- Random number generators ----
    rng = np.random.default_rng(config.seed)

    # Per-camera noise RNGs (different seeds per camera, for realistic variation)
    rng_per_cam = [np.random.default_rng(config.seed + 100 + i * 10)
                   for i in range(config.num_cameras)]

    # ---- Build camera array ----
    print("[1/5] Building camera array...")
    K_list, R_list, t_list, dist_list, cam_centers = build_camera_array(config)
    for i in range(config.num_cameras):
        print(f"  Camera {i}: center=({cam_centers[i,0]:.1f}, {cam_centers[i,1]:.1f}, "
              f"{cam_centers[i,2]:.1f}) mm, f={K_list[i][0,0]:.1f} px")

    surface_config = config
    if args.ground_truth_only:
        surface_config = replace(config, num_surface_points=config.ground_truth_num_points)

    # ---- Generate surface ----
    print(f"\n[2/5] Generating {surface_config.num_surface_points:,} surface points...")
    points_ref, intensities, normals_ref, grains = generate_cylinder_surface(surface_config, rng)
    print(f"  Generated {len(points_ref):,} points on cylinder surface")

    if args.ground_truth_only:
        print(f"\n[3/3] Applying deformation and saving ground truth only...")
        points_def_list = apply_deformation(points_ref, config)
        save_ground_truth_outputs(
            config,
            K_list, R_list, t_list, dist_list, cam_centers,
            points_ref, points_def_list,
        )
        print("\nDone. Ground truth output:")
        print(f"  {os.path.join(config.output_dir, 'ground_truth')}/")
        return

    # ---- Render reference images ----
    print(f"\n[3/5] Rendering reference images ({config.num_cameras} cameras)...")
    images_ref = render_images(
        points_ref, intensities, normals_ref,
        K_list, R_list, t_list, dist_list, cam_centers,
        config, rng_per_cam,
    )

    # ---- Apply deformation ----
    print(f"\n[4/5] Applying deformation: {config.deformation_type}...")
    points_def_list = apply_deformation(points_ref, config)
    images_def_list = []
    for step_idx, pts_def in enumerate(points_def_list):
        print(f"  Step {step_idx + 1}: rendering deformed images...")
        # Deformation changes positions but intensities stay the same
        # Recompute normals for deformed surface, then render with back-face culling
        normals_def = _compute_cylinder_normals(pts_def)
        images_def = render_images(
            pts_def, intensities, normals_def,
            K_list, R_list, t_list, dist_list, cam_centers,
            config, rng_per_cam,
        )
        images_def_list.append(images_def)

    # ---- Save outputs ----
    print(f"\n[5/5] Saving outputs...")
    save_outputs(
        config,
        images_ref, images_def_list,
        K_list, R_list, t_list, dist_list,
        cam_centers,
        points_ref, points_def_list,
    )

    print("\nDone. Output structure:")
    print(f"  {config.output_dir}/")
    print(f"    images/cam_*/001.bmp    — reference images")
    print(f"    images/cam_*/002.bmp    — deformed images")
    print(f"    calibration/cameras.mat — camera parameters")
    print(f"    calibration/points3D.mat— sparse surface points")
    print(f"    ground_truth/*.npy      — true positions & displacements")


if __name__ == "__main__":
    main()
