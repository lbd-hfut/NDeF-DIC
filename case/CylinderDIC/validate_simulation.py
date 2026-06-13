"""
Validation report generator for Cylinder DIC simulation.

Reads the output of simulate_cylinder.py and produces:
  1. Speckle quality metrics (MIG, autocorrelation, grain statistics)
  2. Per-camera coverage & intensity analysis
  3. Multi-view geometry diagnostics
  4. Deformation field verification
  5. Automated markdown report with diagnostic plots

Usage:
  python validate_simulation.py [--output_dir case/CylinderDIC]
"""

import os
import sys
import json
import argparse
import numpy as np
import imageio.v3 as iio
from scipy.io import loadmat
from scipy.ndimage import sobel, gaussian_filter
from scipy.signal import correlate2d
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

# matplotlib may not be available in headless mode
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class ValidateConfig:
    output_dir: str = "case/CylinderDIC"
    report_dir: str = ""           # Default: output_dir/report/
    speckle_subset_size: int = 31  # px, subset size for subset-level MIG
    autocorrelation_crop: int = 64  # px, central crop for autocorrelation


# =========================================================================
# Speckle quality analysis
# =========================================================================

def compute_mig(image: np.ndarray) -> Dict:
    """
    Mean Intensity Gradient (MIG) — the primary DIC speckle quality metric.

    MIG = mean(|∇I|) over the image.
    Good DIC speckle typically has MIG > 15 gray levels/pixel.

    Returns:
        mig_map:   (H, W) gradient magnitude
        mig_mean:  scalar mean MIG
        mig_std:   scalar std of MIG
        mig_x:     mean of |dI/dx|
        mig_y:     mean of |dI/dy|
    """
    if image.dtype == np.uint8:
        img = image.astype(np.float64)
    else:
        img = image

    # Sobel gradients
    gx = sobel(img, axis=1, mode='reflect')
    gy = sobel(img, axis=0, mode='reflect')
    gm = np.sqrt(gx ** 2 + gy ** 2)

    return {
        "mig_map": gm,
        "mig_mean": float(np.mean(gm)),
        "mig_std": float(np.std(gm)),
        "mig_x": float(np.mean(np.abs(gx))),
        "mig_y": float(np.mean(np.abs(gy))),
    }


def compute_autocorrelation(image: np.ndarray, crop: int = 64) -> Dict:
    """
    Speckle size estimation via normalized autocorrelation.

    Speckle size ≈ FWHM of the autocorrelation peak.

    Returns:
        ac_map:        (2*crop+1, 2*crop+1) normalized autocorrelation
        speckle_size:  estimated speckle diameter in pixels
        peak_sharpness: ratio of peak to first sidelobe (higher is better for DIC)
        anisotropy:    ratio of FWHM_x / FWHM_y
    """
    if image.dtype == np.uint8:
        img = image.astype(np.float64)
    else:
        img = image.copy()

    # Use central crop to estimate speckle statistics
    h, w = img.shape
    cy, cx = h // 2, w // 2
    half = crop
    patch = img[max(0, cy - half):min(h, cy + half),
                max(0, cx - half):min(w, cx + half)]

    # Remove mean for normalized correlation
    patch = patch - patch.mean()
    norm = np.std(patch)
    if norm > 1e-8:
        patch = patch / norm

    # 2D autocorrelation via FFT (much faster than correlate2d for large patches)
    fft = np.fft.fft2(patch, s=(2 * patch.shape[0] - 1, 2 * patch.shape[1] - 1))
    ac = np.fft.fftshift(np.fft.ifft2(fft * fft.conj()).real)
    # Normalize
    ac = ac / ac.max()

    # Extract central line profiles
    cy_ac, cx_ac = ac.shape[0] // 2, ac.shape[1] // 2
    profile_x = ac[cy_ac, :]
    profile_y = ac[:, cx_ac]

    # FWHM estimation (interpolated)
    fwhm_x = _compute_fwhm(profile_x)
    fwhm_y = _compute_fwhm(profile_y)
    speckle_size = (fwhm_x + fwhm_y) / 2.0

    # Peak sharpness: ratio of central peak to mean of outer region
    center_val = ac[cy_ac, cx_ac]
    outer_region = ac.copy()
    r = min(cy_ac, cx_ac) // 3
    outer_region[cy_ac - r:cy_ac + r, cx_ac - r:cx_ac + r] = np.nan
    sidelobe_mean = np.nanmean(np.abs(outer_region))
    peak_sharpness = center_val / max(sidelobe_mean, 1e-8)

    # Anisotropy
    anisotropy = fwhm_x / max(fwhm_y, 1e-8)

    return {
        "ac_map": ac,
        "speckle_size_px": float(speckle_size),
        "fwhm_x": float(fwhm_x),
        "fwhm_y": float(fwhm_y),
        "peak_sharpness": float(peak_sharpness),
        "anisotropy": float(anisotropy),
    }


def _compute_fwhm(profile: np.ndarray) -> float:
    """Compute FWHM of a 1D profile with sub-sample interpolation."""
    half_max = (profile.max() + profile.min()) / 2.0
    above = profile >= half_max
    transitions = np.diff(above.astype(int))

    rise = np.where(transitions == 1)[0]
    fall = np.where(transitions == -1)[0]

    if len(rise) == 0 or len(fall) == 0:
        return 0.0

    # Use the pair closest to the center
    center = len(profile) // 2
    left_idx = rise[np.argmin(np.abs(rise - center))]
    right_idx = fall[np.argmin(np.abs(fall - center))]

    # Linear interpolation
    if left_idx + 1 < len(profile):
        frac_left = (half_max - profile[left_idx]) / max(profile[left_idx + 1] - profile[left_idx], 1e-8)
    else:
        frac_left = 0.0

    if right_idx + 1 < len(profile):
        frac_right = (profile[right_idx] - half_max) / max(profile[right_idx] - profile[right_idx + 1], 1e-8)
    else:
        frac_right = 0.0

    return float((right_idx + frac_right) - (left_idx + frac_left))


def analyze_speckle_quality(
    images_ref: List[np.ndarray],
    images_def: List[np.ndarray],
    config: ValidateConfig,
) -> Dict:
    """Aggregate speckle quality metrics across all cameras."""
    n_cam = len(images_ref)
    results = {
        "per_camera": [],
        "summary": {},
    }

    mig_vals, size_vals, sharpness_vals = [], [], []
    mig_def_vals = []

    for cam_id in range(n_cam):
        img_ref = images_ref[cam_id]

        # MIG on reference image
        mig = compute_mig(img_ref)
        mig_vals.append(mig["mig_mean"])

        # Autocorrelation on reference
        ac = compute_autocorrelation(img_ref, config.autocorrelation_crop)
        size_vals.append(ac["speckle_size_px"])
        sharpness_vals.append(ac["peak_sharpness"])

        # MIG on deformed image
        mig_def = compute_mig(images_def[cam_id])
        mig_def_vals.append(mig_def["mig_mean"])

        results["per_camera"].append({
            "cam_id": cam_id,
            "mig_mean": mig["mig_mean"],
            "mig_std": mig["mig_std"],
            "mig_x": mig["mig_x"],
            "mig_y": mig["mig_y"],
            "mig_def_mean": mig_def["mig_mean"],
            "speckle_size_px": ac["speckle_size_px"],
            "fwhm_x": ac["fwhm_x"],
            "fwhm_y": ac["fwhm_y"],
            "peak_sharpness": ac["peak_sharpness"],
            "anisotropy": ac["anisotropy"],
        })

    results["summary"] = {
        "mig_mean_of_means": float(np.mean(mig_vals)),
        "mig_std_of_means": float(np.std(mig_vals)),
        "mig_min": float(np.min(mig_vals)),
        "mig_max": float(np.max(mig_vals)),
        "mig_def_mean": float(np.mean(mig_def_vals)),
        "mig_stability": float(np.mean(np.array(mig_vals) - np.array(mig_def_vals))),
        "speckle_size_mean": float(np.mean(size_vals)),
        "speckle_size_std": float(np.std(size_vals)),
        "peak_sharpness_mean": float(np.mean(sharpness_vals)),
    }

    return results


# =========================================================================
# Coverage & intensity analysis
# =========================================================================

def analyze_coverage(images: List[np.ndarray]) -> Dict:
    """
    Per-camera coverage and intensity statistics.

    Coverage = fraction of pixels above background (not pure black).
    """
    results = {"per_camera": [], "summary": {}}

    coverage_vals, mean_vals, std_vals = [], [], []
    min_vals, max_vals = [], []

    for cam_id, img in enumerate(images):
        if img.dtype == np.uint8:
            img_f = img.astype(np.float64)
        else:
            img_f = img

        # Coverage: pixels significantly above background
        # Background is either 0 (pure black) or intensity_range[0] (offset black)
        bg_thresh = max(img_f.min() + 5.0, 10.0)
        covered = img_f > bg_thresh
        coverage = float(covered.sum() / img_f.size)
        coverage_vals.append(coverage)

        # Intensity statistics (covered pixels only)
        if covered.sum() > 0:
            mean_val = float(img_f[covered].mean())
            std_val = float(img_f[covered].std())
            min_val = float(img_f[covered].min())
            max_val = float(img_f[covered].max())
        else:
            mean_val = std_val = min_val = max_val = 0.0

        mean_vals.append(mean_val)
        std_vals.append(std_val)
        min_vals.append(min_val)
        max_vals.append(max_val)

        results["per_camera"].append({
            "cam_id": cam_id,
            "coverage_ratio": coverage,
            "intensity_mean": mean_val,
            "intensity_std": std_val,
            "intensity_min": min_val,
            "intensity_max": max_val,
            "dynamic_range": max_val - min_val,
        })

    results["summary"] = {
        "coverage_mean": float(np.mean(coverage_vals)),
        "coverage_std": float(np.std(coverage_vals)),
        "coverage_min": float(np.min(coverage_vals)),
        "coverage_max": float(np.max(coverage_vals)),
        "intensity_mean_of_means": float(np.mean(mean_vals)),
        "intensity_mean_of_stds": float(np.mean(std_vals)),
        "global_dynamic_range": float(max(max_vals) - min(min_vals)),
    }

    return results


# =========================================================================
# Multi-view geometry analysis
# =========================================================================

def analyze_multiview_geometry(
    K_list: List[np.ndarray],
    R_list: List[np.ndarray],
    t_list: List[np.ndarray],
    cylinder_radius: float,
    cylinder_height: float,
    working_distance: float,
) -> Dict:
    """
    Analyze multi-view configuration:
      - Camera baseline distances
      - Triangulation angles between adjacent cameras
      - FOV overlap estimation
    """
    n_cam = len(K_list)

    # Compute camera centers in world coords
    centers = []
    for R, t in zip(R_list, t_list):
        C = -R.T @ t.reshape(3, 1)
        centers.append(C.flatten())
    centers = np.array(centers)  # (N, 3)

    # Baselines between adjacent cameras
    baselines = []
    triangulation_angles = []
    for i in range(n_cam):
        j = (i + 1) % n_cam
        b = np.linalg.norm(centers[i] - centers[j])
        baselines.append(b)
        # Approximate triangulation angle: angle between camera rays at object center
        d = np.linalg.norm(centers[i])
        angle = 2.0 * np.arctan(b / (2.0 * d))
        triangulation_angles.append(float(np.degrees(angle)))

    # All-pair baselines
    all_baselines = []
    for i in range(n_cam):
        for j in range(i + 1, n_cam):
            all_baselines.append(float(np.linalg.norm(centers[i] - centers[j])))

    # Horizontal FOV (from camera intrinsics)
    fx = K_list[0][0, 0]
    W = 1440  # typical image width
    fov_horizontal = 2.0 * np.degrees(np.arctan(W / (2.0 * fx)))

    # Cylinder angular extent per camera (what matters for surface overlap)
    camera_distance = float(np.mean(np.linalg.norm(centers, axis=1)))
    cylinder_angular_extent = 2.0 * np.degrees(np.arcsin(cylinder_radius / camera_distance))

    # Overlap of cylinder surface between adjacent cameras
    angular_step = 360.0 / n_cam
    cylinder_overlap_angle = cylinder_angular_extent - angular_step
    cylinder_overlap_ratio = max(0.0, cylinder_overlap_angle / cylinder_angular_extent)
    # Overlap width on surface (mm) along circumference
    overlap_arc_mm = max(0.0, cylinder_overlap_angle * np.pi / 180.0 * cylinder_radius)

    return {
        "num_cameras": n_cam,
        "camera_centers": centers,
        "camera_distance_mean": camera_distance,
        "cylinder_radius_mm": cylinder_radius,
        "adjacent_baselines": [float(b) for b in baselines],
        "baseline_mean": float(np.mean(baselines)),
        "baseline_std": float(np.std(baselines)),
        "baseline_min": float(np.min(all_baselines)),
        "baseline_max": float(np.max(all_baselines)),
        "triangulation_angle_mean": float(np.mean(triangulation_angles)),
        "triangulation_angle_std": float(np.std(triangulation_angles)),
        "fov_horizontal_deg": float(fov_horizontal),
        "cylinder_angular_extent_deg": float(cylinder_angular_extent),
        "angular_step_deg": float(angular_step),
        "cylinder_overlap_angle_deg": float(cylinder_overlap_angle),
        "overlap_ratio": float(cylinder_overlap_ratio),
        "overlap_arc_mm": float(overlap_arc_mm),
    }


# =========================================================================
# Deformation field analysis
# =========================================================================

def analyze_deformation(
    points_ref: np.ndarray,
    points_def: np.ndarray,
    config_dict: Dict,
) -> Dict:
    """Analyze ground-truth deformation field."""
    # Displacement vectors
    disp = points_def - points_ref  # (N, 3)
    disp_mag = np.linalg.norm(disp, axis=1)

    # Per-component statistics
    results = {
        "num_points": len(points_ref),
        "displacement": {
            "u_mean": float(np.mean(disp[:, 0])),
            "u_std": float(np.std(disp[:, 0])),
            "u_min": float(np.min(disp[:, 0])),
            "u_max": float(np.max(disp[:, 0])),
            "v_mean": float(np.mean(disp[:, 1])),
            "v_std": float(np.std(disp[:, 1])),
            "v_min": float(np.min(disp[:, 1])),
            "v_max": float(np.max(disp[:, 1])),
            "w_mean": float(np.mean(disp[:, 2])),
            "w_std": float(np.std(disp[:, 2])),
            "w_min": float(np.min(disp[:, 2])),
            "w_max": float(np.max(disp[:, 2])),
            "magnitude_mean": float(np.mean(disp_mag)),
            "magnitude_std": float(np.std(disp_mag)),
            "magnitude_max": float(np.max(disp_mag)),
            "magnitude_min": float(np.min(disp_mag)),
        },
    }

    # Cylindrical decomposition (for cylinder geometry)
    r_ref = np.sqrt(points_ref[:, 0] ** 2 + points_ref[:, 2] ** 2)
    theta_ref = np.arctan2(points_ref[:, 2], points_ref[:, 0])
    r_def = np.sqrt(points_def[:, 0] ** 2 + points_def[:, 2] ** 2)
    theta_def = np.arctan2(points_def[:, 2], points_def[:, 0])

    dr = r_def - r_ref
    dtheta = theta_def - theta_ref
    # Handle angle wrap
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    dy = points_def[:, 1] - points_ref[:, 1]

    results["cylindrical_displacement"] = {
        "dr_mean": float(np.mean(dr)),
        "dr_std": float(np.std(dr)),
        "dr_max": float(np.max(np.abs(dr))),
        "dtheta_deg_mean": float(np.degrees(np.mean(np.abs(dtheta)))),
        "dtheta_deg_max": float(np.degrees(np.max(np.abs(dtheta)))),
        "dy_mean": float(np.mean(dy)),
        "dy_std": float(np.std(dy)),
        "dy_max": float(np.max(np.abs(dy))),
    }

    # Surface coverage
    R = config_dict.get("cylinder_radius", 80.0)
    H = config_dict.get("cylinder_height", 120.0)
    surface_area = 2 * np.pi * R * H
    point_density = len(points_ref) / surface_area  # points/mm²

    results["surface"] = {
        "surface_area_mm2": float(surface_area),
        "point_density_per_mm2": float(point_density),
        "avg_spacing_mm": float(1.0 / np.sqrt(point_density)) if point_density > 0 else 0.0,
    }

    results["deformation_type"] = config_dict.get("deformation_type", "unknown")
    results["deformation_magnitude"] = config_dict.get("deformation_magnitude", 0.0)

    return results


# =========================================================================
# Diagnostic plots
# =========================================================================

def generate_plots(
    images_ref: List[np.ndarray],
    images_def: List[np.ndarray],
    points_ref: np.ndarray,
    points_def: np.ndarray,
    K_list: List[np.ndarray],
    R_list: List[np.ndarray],
    t_list: List[np.ndarray],
    speckle_quality: Dict,
    multiview: Dict,
    deformation: Dict,
    config: ValidateConfig,
    report_dir: str,
) -> List[str]:
    """Generate all diagnostic plots. Returns list of saved filenames."""
    if not HAS_MPL:
        print("[WARNING] matplotlib not available, skipping plot generation.")
        return []

    saved = []

    # ---- Figure 1: Camera geometry (top-down view) ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    centers = multiview["camera_centers"]
    ax.scatter(centers[:, 0], centers[:, 2], c='red', s=80, marker='s', label='Cameras')
    # Cylinder outline
    R = 80.0
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(R * np.cos(theta), R * np.sin(theta), 'b-', linewidth=2, label='Cylinder')
    # Camera labels and viewing directions
    for i, (cx, cy, cz) in enumerate(centers):
        ax.annotate(f"{i}", (cx, cz), textcoords="offset points", xytext=(8, 8), fontsize=9)
        # View direction to origin
        ax.plot([cx, 0], [cz, 0], 'gray', linewidth=0.5, alpha=0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Z [mm]')
    ax.set_title(f'Camera Geometry (Top-Down View)\n{len(centers)} cameras, D={multiview["camera_distance_mean"]:.0f} mm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = os.path.join(report_dir, "camera_geometry.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    saved.append(fname)

    # ---- Figure 2: Per-camera thumbnails with MIG overlay ----
    n_cam = len(images_ref)
    n_cols = min(4, n_cam)
    n_rows = int(np.ceil(n_cam / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for cam_id in range(n_cam):
        ax = axes[cam_id]
        ax.imshow(images_ref[cam_id], cmap='gray', vmin=0, vmax=255)
        mig_val = speckle_quality["per_camera"][cam_id]["mig_mean"]
        ax.set_title(f"Cam {cam_id}\nMIG={mig_val:.1f}", fontsize=8)
        ax.axis('off')
    for i in range(n_cam, len(axes)):
        axes[i].axis('off')
    fig.suptitle('Reference Images — Coverage & MIG Overview', fontsize=12, y=1.01)
    fig.tight_layout()
    fname = os.path.join(report_dir, "camera_thumbnails.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    saved.append(fname)

    # ---- Figure 3: Intensity histograms (4 representative cameras) ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    step = max(1, n_cam // 4)
    for idx, cam_id in enumerate([0, step, 2 * step, 3 * step]):
        ax = axes[idx // 2, idx % 2]
        cam_id = min(cam_id, n_cam - 1)
        img = images_ref[cam_id]
        ax.hist(img.ravel(), bins=128, range=(0, 255), color='steelblue', alpha=0.7, density=True)
        ax.axvline(img.mean(), color='red', linestyle='--', label=f'Mean={img.mean():.0f}')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Density')
        ax.set_title(f'Camera {cam_id} Intensity Distribution')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 255)
    fig.suptitle('Speckle Intensity Histograms', fontsize=13)
    fig.tight_layout()
    fname = os.path.join(report_dir, "intensity_histograms.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    saved.append(fname)

    # ---- Figure 4: Displacement field visualization ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    disp = points_def - points_ref
    disp_mag = np.linalg.norm(disp, axis=1)

    # Subsample for scatter plot
    n_sample = min(5000, len(points_ref))
    idx = np.random.default_rng(42).choice(len(points_ref), n_sample, replace=False)

    pts_sample = points_ref[idx]
    disp_sample = disp[idx]
    mag_sample = disp_mag[idx]

    # Map displacement magnitude on cylinder unwrapped
    theta = np.arctan2(pts_sample[:, 2], pts_sample[:, 0])
    y = pts_sample[:, 1]

    sc = axes[0].scatter(np.degrees(theta), y, c=mag_sample, s=2, cmap='hot')
    axes[0].set_xlabel('θ [deg]')
    axes[0].set_ylabel('Y [mm]')
    axes[0].set_title(f'Displacement Magnitude\nMean={disp_mag.mean():.4f} mm, Max={disp_mag.max():.4f} mm')
    plt.colorbar(sc, ax=axes[0], label='|disp| [mm]')

    # Histogram of displacement magnitude
    axes[1].hist(disp_mag, bins=100, color='steelblue', alpha=0.7, density=True)
    axes[1].axvline(disp_mag.mean(), color='red', linestyle='--', label=f'Mean={disp_mag.mean():.4f}')
    axes[1].axvline(disp_mag.max(), color='orange', linestyle=':', label=f'Max={disp_mag.max():.4f}')
    axes[1].set_xlabel('Displacement magnitude [mm]')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Displacement Distribution')
    axes[1].legend()

    # Radial component
    r = np.sqrt(pts_sample[:, 0] ** 2 + pts_sample[:, 2] ** 2)
    dr = disp_sample[:, 0] * pts_sample[:, 0] / np.maximum(r, 1e-6) + \
         disp_sample[:, 2] * pts_sample[:, 2] / np.maximum(r, 1e-6)
    sc = axes[2].scatter(np.degrees(theta), y, c=dr, s=2, cmap='RdBu_r', vmin=-np.abs(dr).max(), vmax=np.abs(dr).max())
    axes[2].set_xlabel('θ [deg]')
    axes[2].set_ylabel('Y [mm]')
    axes[2].set_title(f'Radial Displacement\nMean radial={dr.mean():.4f} mm')
    plt.colorbar(sc, ax=axes[2], label='dr [mm]')

    fig.suptitle('Ground Truth Deformation Field', fontsize=13)
    fig.tight_layout()
    fname = os.path.join(report_dir, "deformation_field.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    saved.append(fname)

    # ---- Figure 5: MIG comparison bar chart ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    cam_ids = list(range(n_cam))
    mig_vals = [speckle_quality["per_camera"][i]["mig_mean"] for i in cam_ids]
    mig_def_vals = [speckle_quality["per_camera"][i].get("mig_def_mean", 0) for i in cam_ids]
    x = np.arange(len(cam_ids))
    w = 0.35
    bars1 = ax.bar(x - w / 2, mig_vals, w, label='Reference (001.bmp)', color='steelblue')
    bars2 = ax.bar(x + w / 2, mig_def_vals, w, label='Deformed (002.bmp)', color='coral')
    ax.axhline(15, color='green', linestyle='--', linewidth=1, label='DIC minimum (15)')
    ax.set_xlabel('Camera ID')
    ax.set_ylabel('MIG [gray levels/px]')
    ax.set_title('Per-Camera MIG Comparison')
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fname = os.path.join(report_dir, "mig_comparison.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    saved.append(fname)

    # ---- Figure 6: Speckle autocorrelation (camera 0) ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ac = speckle_quality["per_camera"][0]
    ac_map_data = compute_autocorrelation(images_ref[0], config.autocorrelation_crop)["ac_map"]

    ax = axes[0]
    im = ax.imshow(ac_map_data, cmap='hot', extent=[-ac_map_data.shape[1]//2, ac_map_data.shape[1]//2,
                                                      -ac_map_data.shape[0]//2, ac_map_data.shape[0]//2])
    ax.set_xlabel('Δx [px]')
    ax.set_ylabel('Δy [px]')
    ax.set_title(f'Autocorrelation (Cam 0)\nSpeckle Size={ac["speckle_size_px"]:.1f} px')
    plt.colorbar(im, ax=ax)

    # Central profiles
    ax = axes[1]
    cy = ac_map_data.shape[0] // 2
    cx = ac_map_data.shape[1] // 2
    profile_x = ac_map_data[cy, :]
    profile_y = ac_map_data[:, cx]
    x_axis = np.arange(len(profile_x)) - cx
    y_axis = np.arange(len(profile_y)) - cy
    ax.plot(x_axis, profile_x, label=f'X profile (FWHM={ac["fwhm_x"]:.1f})')
    ax.plot(y_axis, profile_y, label=f'Y profile (FWHM={ac["fwhm_y"]:.1f})')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Lag [px]')
    ax.set_ylabel('Correlation')
    ax.set_title('Autocorrelation Profiles')
    ax.legend()
    ax.set_xlim(-30, 30)
    ax.grid(alpha=0.3)

    fig.suptitle('Speckle Pattern Analysis', fontsize=13)
    fig.tight_layout()
    fname = os.path.join(report_dir, "speckle_analysis.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    saved.append(fname)

    # ---- Figure 7: Deformed image comparison (cam 0: ref vs def vs difference) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cam_id = 0
    ax = axes[0]
    ax.imshow(images_ref[cam_id], cmap='gray', vmin=0, vmax=255)
    ax.set_title('Reference (001.bmp)')
    ax.axis('off')

    ax = axes[1]
    ax.imshow(images_def[cam_id], cmap='gray', vmin=0, vmax=255)
    ax.set_title('Deformed (002.bmp)')
    ax.axis('off')

    ax = axes[2]
    diff = images_def[cam_id].astype(np.float64) - images_ref[cam_id].astype(np.float64)
    vmax = max(abs(diff.min()), abs(diff.max()))
    ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'Difference (Def − Ref)\nRange: [{diff.min():.0f}, {diff.max():.0f}]')
    ax.axis('off')

    fig.suptitle(f'Reference vs Deformed — Camera {cam_id}', fontsize=13)
    fig.tight_layout()
    fname = os.path.join(report_dir, "ref_vs_def.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    saved.append(fname)

    return saved


# =========================================================================
# Report generation
# =========================================================================

def generate_report(
    speckle_quality: Dict,
    coverage: Dict,
    multiview: Dict,
    deformation: Dict,
    plot_files: List[str],
    config: ValidateConfig,
    report_dir: str,
) -> str:
    """Generate a comprehensive markdown validation report."""

    sq = speckle_quality["summary"]
    cov = coverage["summary"]
    mv = multiview
    df = deformation

    lines = []
    def w(s=""):
        lines.append(s)

    w("# Cylinder DIC Simulation — Validation Report")
    w()
    w(f"**Output directory:** `{config.output_dir}`")
    w()
    w("---")
    w()
    w("## 1. Executive Summary")
    w()
    # Overall verdict
    mig_ok = sq["mig_mean_of_means"] > 15
    cov_ok = cov["coverage_mean"] > 0.1
    size_ok = 2.0 < sq["speckle_size_mean"] < 8.0
    overlap_ok = mv["overlap_ratio"] > 0.4

    all_ok = mig_ok and cov_ok and size_ok and overlap_ok
    status = "✅ **PASS** — All key metrics within acceptable ranges." if all_ok else \
             "⚠️ **REVIEW** — Some metrics need attention (see details below)."

    w(f"**Overall Assessment:** {status}")
    w()
    w("| Category | Key Metric | Value | Status |")
    w("|----------|-----------|-------|--------|")
    w(f"| Speckle Quality | MIG (mean) | {sq['mig_mean_of_means']:.1f} | {'✅' if mig_ok else '⚠️'} |")
    w(f"| Coverage | Mean ratio | {cov['coverage_mean']:.2%} | {'✅' if cov_ok else '⚠️'} |")
    w(f"| Speckle Size | Mean FWHM | {sq['speckle_size_mean']:.1f} px | {'✅' if size_ok else '⚠️'} |")
    w(f"| View Overlap | Adjacent overlap | {mv['overlap_ratio']:.1%} | {'✅' if overlap_ok else '⚠️'} |")
    w(f"| Peak Sharpness | Correlation peak | {sq['peak_sharpness_mean']:.1f} | — |")
    w()
    w("---")
    w()
    w("## 2. Camera Configuration")
    w()
    w(f"| Parameter | Value |")
    w(f"|-----------|-------|")
    w(f"| Number of cameras | {mv['num_cameras']} |")
    w(f"| Mean camera distance | {mv['camera_distance_mean']:.1f} mm |")
    w(f"| Horizontal FOV per camera | {mv['fov_horizontal_deg']:.1f}° |")
    w(f"| Cylinder angular extent per camera | {mv['cylinder_angular_extent_deg']:.1f}° |")
    w(f"| Angular step between cameras | {mv['angular_step_deg']:.1f}° |")
    w(f"| Cylinder surface overlap angle | {mv['cylinder_overlap_angle_deg']:.1f}° |")
    w(f"| Overlap ratio (relative to cylinder) | {mv['overlap_ratio']:.1%} |")
    w(f"| Overlap arc length on surface | {mv['overlap_arc_mm']:.1f} mm |")
    w(f"| Mean adjacent baseline | {mv['baseline_mean']:.1f} mm |")
    w(f"| Mean triangulation angle | {mv['triangulation_angle_mean']:.1f}° |")
    w(f"| Min / Max baseline | {mv['baseline_min']:.1f} / {mv['baseline_max']:.1f} mm |")
    w()
    w("### Baseline Distances (adjacent cameras)")
    w()
    w("| Camera Pair | Baseline [mm] | Triangulation Angle [°] |")
    w("|-------------|---------------|------------------------|")
    for i, (b, a) in enumerate(zip(mv["adjacent_baselines"],
                                     [mv["triangulation_angle_mean"]] * len(mv["adjacent_baselines"]))):
        angle = mv.get("triangulation_angles", [mv["triangulation_angle_mean"]] * len(mv["adjacent_baselines"]))
        w(f"| {i} → {(i+1) % mv['num_cameras']} | {b:.1f} | {angle[i]:.1f} |")
    w()

    w("![Camera Geometry](camera_geometry.png)")
    w()

    w("---")
    w()
    w("## 3. Speckle Quality Analysis")
    w()
    w("### 3.1 Mean Intensity Gradient (MIG)")
    w()
    w("MIG is the primary DIC quality metric. Higher values enable more precise sub-pixel matching.")
    w("A value above 15 gray levels/pixel is considered adequate for DIC.")
    w()
    w(f"| Metric | Reference (001.bmp) | Deformed (002.bmp) |")
    w(f"|--------|---------------------|---------------------|")
    w(f"| Mean MIG | **{sq['mig_mean_of_means']:.1f}** | {sq['mig_def_mean']:.1f} |")
    w(f"| Std of means | {sq['mig_std_of_means']:.1f} | — |")
    w(f"| Min MIG | {sq['mig_min']:.1f} | — |")
    w(f"| Max MIG | {sq['mig_max']:.1f} | — |")
    w(f"| Ref→Def stability | {sq['mig_stability']:.2f} | — |")
    w()

    w("### 3.2 Per-Camera MIG")
    w()
    w("| Camera | MIG (ref) | MIG (def) | Δ |")
    w("|--------|-----------|-----------|----|")
    for cam in speckle_quality["per_camera"]:
        delta = cam["mig_mean"] - cam.get("mig_def_mean", 0)
        w(f"| {cam['cam_id']} | {cam['mig_mean']:.1f} | {cam.get('mig_def_mean', 0):.1f} | {delta:+.1f} |")
    w()

    w("### 3.3 Speckle Size (Autocorrelation FWHM)")
    w()
    w(f"| Metric | Value | Ideal Range |")
    w(f"|--------|-------|-------------|")
    w(f"| Mean speckle size | **{sq['speckle_size_mean']:.1f} px** | 3–5 px |")
    w(f"| Std of sizes | {sq['speckle_size_std']:.1f} px | — |")
    w(f"| Mean peak sharpness | {sq['peak_sharpness_mean']:.1f} | > 5 |")
    w()

    w("| Camera | Speckle Size [px] | FWHM X | FWHM Y | Anisotropy | Peak Sharpness |")
    w("|--------|-------------------|--------|--------|------------|----------------|")
    for cam in speckle_quality["per_camera"]:
        w(f"| {cam['cam_id']} | {cam['speckle_size_px']:.1f} | {cam['fwhm_x']:.1f} | {cam['fwhm_y']:.1f} | {cam['anisotropy']:.2f} | {cam['peak_sharpness']:.1f} |")
    w()

    w("![MIG Comparison](mig_comparison.png)")
    w()
    w("![Speckle Analysis](speckle_analysis.png)")
    w()

    w("---")
    w()
    w("## 4. Coverage & Intensity")
    w()
    w(f"| Metric | Value |")
    w(f"|--------|-------|")
    w(f"| Mean coverage | **{cov['coverage_mean']:.1%}** |")
    w(f"| Coverage std | {cov['coverage_std']:.1%} |")
    w(f"| Coverage range | [{cov['coverage_min']:.1%}, {cov['coverage_max']:.1%}] |")
    w(f"| Mean intensity | {cov['intensity_mean_of_means']:.1f} |")
    w(f"| Mean within-image std | {cov['intensity_mean_of_stds']:.1f} |")
    w(f"| Global dynamic range | {cov['global_dynamic_range']:.1f} |")
    w()

    w("| Camera | Coverage | Mean Int. | Std Int. | Range |")
    w("|--------|----------|-----------|----------|-------|")
    for cam in coverage["per_camera"]:
        c = cam
        w(f"| {c['cam_id']} | {c['coverage_ratio']:.2%} | {c['intensity_mean']:.1f} | {c['intensity_std']:.1f} | [{c['intensity_min']:.0f}, {c['intensity_max']:.0f}] |")
    w()

    w("![Coverage Overview](camera_thumbnails.png)")
    w()
    w("![Intensity Histograms](intensity_histograms.png)")
    w()

    w("---")
    w()
    w("## 5. Ground Truth Deformation")
    w()
    w(f"**Deformation type:** {deformation.get('deformation_type', 'N/A')}")
    w(f"**Deformation magnitude:** {deformation.get('deformation_magnitude', 'N/A')}")
    w()
    w("### 5.1 Displacement Statistics")
    w()
    d = deformation["displacement"]
    w("| Component | Mean [mm] | Std [mm] | Min [mm] | Max [mm] |")
    w("|-----------|-----------|----------|----------|----------|")
    w(f"| U (X) | {d['u_mean']:.4f} | {d['u_std']:.4f} | {d['u_min']:.4f} | {d['u_max']:.4f} |")
    w(f"| V (Y) | {d['v_mean']:.4f} | {d['v_std']:.4f} | {d['v_min']:.4f} | {d['v_max']:.4f} |")
    w(f"| W (Z) | {d['w_mean']:.4f} | {d['w_std']:.4f} | {d['w_min']:.4f} | {d['w_max']:.4f} |")
    w(f"| Magnitude | **{d['magnitude_mean']:.4f}** | {d['magnitude_std']:.4f} | {d['magnitude_min']:.4f} | {d['magnitude_max']:.4f} |")
    w()

    if "cylindrical_displacement" in deformation:
        cd = deformation["cylindrical_displacement"]
        w("### 5.2 Cylindrical Decomposition")
        w()
        w(f"| Component | Mean | Max |")
        w(f"|-----------|------|-----|")
        w(f"| Radial dr | {cd['dr_mean']:.4f} mm | {cd['dr_max']:.4f} mm |")
        w(f"| Tangential dθ | {cd['dtheta_deg_mean']:.4f}° | {cd['dtheta_deg_max']:.4f}° |")
        w(f"| Axial dy | {cd['dy_mean']:.4f} mm | {cd['dy_max']:.4f} mm |")
        w()

    if "surface" in deformation:
        s = deformation["surface"]
        w("### 5.3 Surface Sampling")
        w()
        w(f"| Metric | Value |")
        w(f"|--------|-------|")
        w(f"| Surface area | {s['surface_area_mm2']:.1f} mm² |")
        w(f"| Number of points | {deformation['num_points']:,} |")
        w(f"| Point density | {s['point_density_per_mm2']:.2f} pts/mm² |")
        w(f"| Avg point spacing | {s['avg_spacing_mm']:.3f} mm |")
        w()

    w("![Deformation Field](deformation_field.png)")
    w()
    w("![Reference vs Deformed](ref_vs_def.png)")
    w()

    w("---")
    w()
    w("## 6. Quality Checklist for Step 1 Readiness")
    w()
    # Build checklist
    checks = []
    # Speckle MIG
    checks.append((mig_ok, "MIG > 15 gray levels/px",
                   f"All cameras have sufficient texture for DIC matching (mean MIG = {sq['mig_mean_of_means']:.1f})",
                   f"Mean MIG ({sq['mig_mean_of_means']:.1f}) below 15 — DIC matching may be unreliable"))
    # Coverage
    checks.append((cov_ok, "Coverage > 10%",
                   f"All cameras have adequate surface coverage (mean = {cov['coverage_mean']:.1%})",
                   f"Mean coverage ({cov['coverage_mean']:.1%}) below 10% — too few surface pixels"))
    # Speckle size
    checks.append((size_ok, "Speckle size 3–5 px",
                   f"Speckle grains are properly sized for DIC (mean = {sq['speckle_size_mean']:.1f} px)",
                   f"Speckle size ({sq['speckle_size_mean']:.1f} px) outside ideal 3–5 px range"))
    # Overlap
    checks.append((overlap_ok, "View overlap > 40%",
                   f"Adjacent cameras have sufficient overlap for multi-view reconstruction ({mv['overlap_ratio']:.1%})",
                   f"View overlap ({mv['overlap_ratio']:.1%}) below 40% — may cause gaps in 3D reconstruction"))
    # Intensity
    intensity_ok = cov['intensity_mean_of_means'] > 30
    checks.append((intensity_ok, "Mean intensity > 30",
                   f"Adequate brightness for DIC (mean = {cov['intensity_mean_of_means']:.1f})",
                   f"Images may be too dark (mean = {cov['intensity_mean_of_means']:.1f})"))
    # Coverage uniformity
    uniformity_ok = cov['coverage_std'] < 0.15
    checks.append((uniformity_ok, "Coverage uniformity (std < 15%)",
                   f"Coverage is uniform across cameras (std = {cov['coverage_std']:.1%})",
                   f"Coverage varies significantly across cameras (std = {cov['coverage_std']:.1%})"))

    for ok, criterion, pass_msg, fail_msg in checks:
        icon = "✅" if ok else "⚠️"
        msg = pass_msg if ok else fail_msg
        w(f"- {icon} **{criterion}** — {msg}")

    w()
    w("---")
    w()
    w("## 7. Output Files")
    w()
    w(f"| Path | Description |")
    w(f"|------|-------------|")
    w(f"| `images/cam_*/001.bmp` | Reference images ({mv['num_cameras']} cameras) |")
    w(f"| `images/cam_*/002.bmp` | Deformed images ({mv['num_cameras']} cameras) |")
    w(f"| `calibration/cameras.mat` | Camera intrinsics & extrinsics |")
    w(f"| `calibration/points3D.mat` | Sparse surface points |")
    w(f"| `ground_truth/points_ref.npy` | Ground truth reference points |")
    w(f"| `ground_truth/points_def_step1.npy` | Ground truth deformed points |")
    w(f"| `ground_truth/displacement_step1.npy` | Ground truth displacement field |")
    w(f"| `ground_truth/meta.json` | Simulation parameters |")
    w()

    report_path = os.path.join(report_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate Cylinder DIC simulation outputs"
    )
    parser.add_argument("--output_dir", type=str, default="case/CylinderDIC",
                        help="Path to simulation output directory")
    parser.add_argument("--subset_size", type=int, default=31,
                        help="Subset size for subset-level MIG (px)")
    parser.add_argument("--skip_plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    config = ValidateConfig(
        output_dir=args.output_dir,
        speckle_subset_size=args.subset_size,
    )

    out_dir = config.output_dir
    report_dir = os.path.join(out_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    config.report_dir = report_dir

    print("=" * 60)
    print("Cylinder DIC Simulation — Validation Report Generator")
    print("=" * 60)
    print(f"  Output dir:  {out_dir}")
    print(f"  Report dir:  {report_dir}")
    print()

    # ---- Load data ----
    print("[1/6] Loading data...")

    img_dir = os.path.join(out_dir, "images")
    n_cam = 0
    # Detect number of cameras
    while os.path.isdir(os.path.join(img_dir, f"cam_{n_cam}")):
        n_cam += 1
    if n_cam == 0:
        print("ERROR: No camera directories found!")
        sys.exit(1)
    print(f"  Found {n_cam} cameras")

    # Load reference images
    images_ref = []
    for cam_id in range(n_cam):
        fname = os.path.join(img_dir, f"cam_{cam_id}", "001.bmp")
        if not os.path.exists(fname):
            print(f"ERROR: Missing reference image: {fname}")
            sys.exit(1)
        images_ref.append(iio.imread(fname))
    print(f"  Loaded {len(images_ref)} reference images")
    print(f"    Image size: {images_ref[0].shape}")

    # Load deformed images
    images_def = []
    for cam_id in range(n_cam):
        fname = os.path.join(img_dir, f"cam_{cam_id}", "002.bmp")
        if not os.path.exists(fname):
            print(f"WARNING: Missing deformed image for cam_{cam_id}, using reference")
            images_def.append(images_ref[cam_id])
        else:
            images_def.append(iio.imread(fname))
    print(f"  Loaded {len(images_def)} deformed images")

    # Load calibration
    calib_path = os.path.join(out_dir, "calibration", "cameras.mat")
    if os.path.exists(calib_path):
        calib = loadmat(calib_path)

        # Helper to extract a clean float array from nested object arrays
        def _extract_mat(obj, shape):
            """Recursively extract floats from nested object array."""
            flat = np.array([float(obj.flat[i].item()) for i in range(obj.size)])
            return flat.reshape(shape)

        n_cams_mat = int(calib["num_cameras"][0, 0])
        K_raw = calib["K_list"]
        R_raw = calib["cam_from_world_R"]
        t_raw = calib["cam_from_world_t"]

        K_list = [_extract_mat(K_raw[i], (3, 3)) for i in range(n_cams_mat)]
        R_list = [_extract_mat(R_raw[i], (3, 3)) for i in range(n_cams_mat)]
        t_list = [_extract_mat(t_raw[i], (3, 1)) for i in range(n_cams_mat)]

        print(f"  Loaded calibration: {len(K_list)} cameras")
    else:
        print("ERROR: Missing calibration file!")
        sys.exit(1)

    # Load ground truth
    gt_dir = os.path.join(out_dir, "ground_truth")
    points_ref_path = os.path.join(gt_dir, "points_ref.npy")
    points_def_path = os.path.join(gt_dir, "points_def_step1.npy")
    if os.path.exists(points_ref_path) and os.path.exists(points_def_path):
        points_ref = np.load(points_ref_path)
        points_def = np.load(points_def_path)
        print(f"  Loaded ground truth: {len(points_ref)} points")
    else:
        print("WARNING: Ground truth files missing, skipping deformation analysis")
        points_ref = np.zeros((1, 3))
        points_def = np.zeros((1, 3))

    meta_path = os.path.join(gt_dir, "meta.json")
    config_dict = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            config_dict = json.load(f)

    # ---- Analyze ----
    print("\n[2/6] Analyzing speckle quality...")
    speckle_quality = analyze_speckle_quality(images_ref, images_def, config)
    print(f"  Mean MIG: {speckle_quality['summary']['mig_mean_of_means']:.1f}")
    print(f"  Mean speckle size: {speckle_quality['summary']['speckle_size_mean']:.1f} px")

    print("\n[3/6] Analyzing coverage & intensity...")
    coverage = analyze_coverage(images_ref)
    print(f"  Mean coverage: {coverage['summary']['coverage_mean']:.1%}")
    print(f"  Mean intensity: {coverage['summary']['intensity_mean_of_means']:.1f}")

    print("\n[4/6] Analyzing multi-view geometry...")
    multiview = analyze_multiview_geometry(
        K_list, R_list, t_list,
        config_dict.get("cylinder_radius", 80.0),
        config_dict.get("cylinder_height", 120.0),
        config_dict.get("working_distance", 400.0),
    )
    print(f"  Mean baseline: {multiview['baseline_mean']:.1f} mm")
    print(f"  Overlap ratio: {multiview['overlap_ratio']:.1%}")

    print("\n[5/6] Analyzing deformation field...")
    deformation = analyze_deformation(points_ref, points_def, config_dict)
    print(f"  Displacement magnitude: mean={deformation['displacement']['magnitude_mean']:.4f} mm")

    # Store coverage in speckle_quality for plot access
    speckle_quality["coverage"] = coverage

    # ---- Generate plots ----
    plot_files = []
    if not args.skip_plots:
        print("\n[6/6] Generating diagnostic plots...")
        plot_files = generate_plots(
            images_ref, images_def,
            points_ref, points_def,
            K_list, R_list, t_list,
            speckle_quality, multiview, deformation,
            config, report_dir,
        )
        print(f"  Generated {len(plot_files)} plots")
    else:
        print("\n[6/6] Skipping plot generation")

    # ---- Generate report ----
    print("\nGenerating markdown report...")
    report_path = generate_report(
        speckle_quality, coverage, multiview, deformation,
        plot_files, config, report_dir,
    )
    print(f"\n  Report saved to: {report_path}")

    # Save metrics as JSON for programmatic access
    metrics_json = {
        "speckle_quality": speckle_quality["summary"],
        "coverage": coverage["summary"],
        "multiview": {k: v for k, v in multiview.items() if k != "camera_centers"},
        "deformation": {
            "displacement": deformation.get("displacement", {}),
            "cylindrical": deformation.get("cylindrical_displacement", {}),
            "surface": deformation.get("surface", {}),
        },
    }
    # Convert numpy values
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    metrics_json = convert(metrics_json)
    metrics_path = os.path.join(report_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Metrics JSON saved to: {metrics_path}")

    print("\n" + "=" * 60)
    print("Validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
