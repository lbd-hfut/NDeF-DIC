"""
Dense Multi-View Stereo via COLMAP PatchMatch.

Wraps pycolmap's stereo pipeline:
  1. Image undistortion (creates dense workspace)
  2. PatchMatch Stereo (requires CUDA)
  3. Stereo Fusion → dense point cloud

If CUDA is unavailable, the dense steps are skipped gracefully.
"""

import os
import shutil
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class DenseMVSConfig:
    """Dense MVS parameters, tuned for speckle patterns."""

    # ---- Undistortion ----
    max_image_size: int = -1         # -1 = full resolution
    min_scale: float = 1.0           # 1.0 = no downscaling
    max_scale: float = 1.0

    # ---- PatchMatch Stereo ----
    window_radius: int = 7           # Larger for speckle (default 5)
    window_step: int = 1
    num_iterations: int = 7          # More iterations for speckle
    num_samples: int = 15
    geom_consistency: bool = True    # Essential for speckle
    geom_consistency_regularizer: float = 0.4
    geom_consistency_max_cost: float = 3.0
    filter_min_ncc: float = 0.1
    filter_min_num_consistent: int = 2
    filter_min_triangulation_angle: float = 3.0
    filter_geom_consistency_max_cost: float = 1.0
    min_triangulation_angle: float = 1.0
    ncc_sigma: float = 0.6
    depth_min: float = -1.0          # -1 = auto
    depth_max: float = -1.0          # -1 = auto
    gpu_index: str = "-1"            # "-1" = auto-select

    # ---- Stereo Fusion ----
    min_num_pixels: int = 5
    max_num_pixels: int = 10000
    max_depth_error: float = 0.01
    max_normal_error: float = 10.0
    max_reproj_error: float = 2.0
    max_traversal_depth: int = 100
    check_num_images: int = 50


# =========================================================================
# CUDA check
# =========================================================================

def has_cuda() -> bool:
    """Check if pycolmap was built with CUDA support."""
    try:
        import pycolmap
        return getattr(pycolmap, "has_cuda", False)
    except ImportError:
        return False


# =========================================================================
# Dense workspace
# =========================================================================

def _prepare_colmap_input(
    image_dir: str,
    sfm_path: str,
    workspace_dir: str,
) -> str:
    """
    Prepare a COLMAP-compatible input directory for undistortion.

    COLMAP undistorter expects:
      - An image_path with the original images
      - An input_path with the sparse SfM result (cameras.bin, images.bin, points3D.bin)

    For pycolmap.undistort_images(), the workflow is:
      output_path = <workspace>/dense
      input_path  = <sfm_path>        (sparse reconstruction)
      image_path  = <image_dir>       (original images)

    Returns output_path.
    """
    os.makedirs(workspace_dir, exist_ok=True)
    output_path = os.path.join(workspace_dir, "dense")
    return output_path


def _build_patchmatch_options(cfg: DenseMVSConfig) -> "pycolmap.PatchMatchOptions":
    """Build PatchMatchOptions from config."""
    import pycolmap
    opts = pycolmap.PatchMatchOptions()
    opts.max_image_size = cfg.max_image_size
    opts.gpu_index = cfg.gpu_index
    opts.depth_min = cfg.depth_min
    opts.depth_max = cfg.depth_max
    opts.window_radius = cfg.window_radius
    opts.window_step = cfg.window_step
    opts.num_iterations = cfg.num_iterations
    opts.num_samples = cfg.num_samples
    opts.geom_consistency = cfg.geom_consistency
    opts.geom_consistency_regularizer = cfg.geom_consistency_regularizer
    opts.geom_consistency_max_cost = cfg.geom_consistency_max_cost
    opts.filter = True
    opts.filter_min_ncc = cfg.filter_min_ncc
    opts.filter_min_num_consistent = cfg.filter_min_num_consistent
    opts.filter_min_triangulation_angle = cfg.filter_min_triangulation_angle
    opts.filter_geom_consistency_max_cost = cfg.filter_geom_consistency_max_cost
    opts.min_triangulation_angle = cfg.min_triangulation_angle
    opts.ncc_sigma = cfg.ncc_sigma
    return opts


def _build_stereo_fusion_options(cfg: DenseMVSConfig) -> "pycolmap.StereoFusionOptions":
    """Build StereoFusionOptions from config."""
    import pycolmap
    opts = pycolmap.StereoFusionOptions()
    opts.min_num_pixels = cfg.min_num_pixels
    opts.max_num_pixels = cfg.max_num_pixels
    opts.max_depth_error = cfg.max_depth_error
    opts.max_normal_error = cfg.max_normal_error
    opts.max_reproj_error = cfg.max_reproj_error
    opts.max_traversal_depth = cfg.max_traversal_depth
    opts.check_num_images = cfg.check_num_images
    return opts


# =========================================================================
# Main pipeline
# =========================================================================

def run_dense_mvs(
    image_dir: str,
    sfm_path: str,
    workspace_dir: str,
    config: Optional[DenseMVSConfig] = None,
    clean: bool = False,
) -> Dict:
    """
    Run COLMAP dense MVS pipeline.

    Args:
        image_dir:     Directory containing original images.
        sfm_path:      Directory containing sparse SfM output (from colmap_calib).
        workspace_dir: Working directory for dense output.
        config:        Dense MVS parameters.
        clean:         Remove existing workspace before running.

    Returns:
        dict with keys:
          - "status":         "ok" | "skipped_cuda" | "error"
          - "dense_points":   (N, 3) np.ndarray (None if skipped)
          - "dense_normals":  (N, 3) np.ndarray (None if skipped)
          - "depth_map_dir":  path to depth maps
          - "reconstruction": pycolmap Reconstruction (None if skipped)
          - "message":        human-readable status
    """
    import pycolmap

    cfg = config or DenseMVSConfig()

    if clean and os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir, exist_ok=True)

    # ---- Check CUDA ----
    if not has_cuda():
        msg = (
            "CUDA not available — pycolmap patch_match_stereo requires CUDA.\n"
            "  Dense MVS skipped. Use sparse points or load ground truth.\n"
            "  To enable: install COLMAP with CUDA support."
        )
        print(f"[Step 1 Dense] WARNING: {msg}")
        return {
            "status": "skipped_cuda",
            "dense_points": None,
            "dense_normals": None,
            "depth_map_dir": None,
            "reconstruction": None,
            "message": msg,
        }

    # ---- 1. Undistort images ----
    print("[Step 1 Dense] Undistorting images...")
    output_path = os.path.join(workspace_dir, "dense")

    undistort_opts = pycolmap.UndistortCameraOptions()
    undistort_opts.max_image_size = cfg.max_image_size
    undistort_opts.min_scale = cfg.min_scale
    undistort_opts.max_scale = cfg.max_scale

    pycolmap.undistort_images(
        output_path=output_path,
        input_path=sfm_path,
        image_path=image_dir,
        undistort_options=undistort_opts,
    )
    print(f"  Undistorted images saved to {output_path}")

    # ---- 2. PatchMatch Stereo ----
    print("[Step 1 Dense] Running PatchMatch Stereo...")
    pm_opts = _build_patchmatch_options(cfg)
    print(f"  window_radius={cfg.window_radius}, "
          f"num_iterations={cfg.num_iterations}, "
          f"geom_consistency={cfg.geom_consistency}")

    pycolmap.patch_match_stereo(
        workspace_path=output_path,
        options=pm_opts,
    )
    print(f"  PatchMatch complete. Depth maps saved.")

    # ---- 3. Stereo Fusion ----
    print("[Step 1 Dense] Running Stereo Fusion...")
    sf_opts = _build_stereo_fusion_options(cfg)

    reconstruction = pycolmap.stereo_fusion(
        output_path=os.path.join(workspace_dir, "fused.ply"),
        workspace_path=output_path,
        options=sf_opts,
    )

    # Extract points and normals
    points3D_list = []
    normals_list = []
    for pid in reconstruction.point3D_ids:
        pt = reconstruction.point3D(pid)
        if pt.error < 4.0:
            points3D_list.append(pt.xyz)
            normals_list.append(pt.normal if hasattr(pt, 'normal') else np.zeros(3))

    dense_points = np.array(points3D_list, dtype=np.float32) if points3D_list else np.zeros((0, 3))
    dense_normals = np.array(normals_list, dtype=np.float32) if normals_list else np.zeros((0, 3))

    print(f"  Fusion complete: {len(dense_points)} points")

    # Save fused point cloud
    ply_path = os.path.join(workspace_dir, "dense_points.ply")
    _save_ply(ply_path, dense_points, dense_normals)
    np.save(os.path.join(workspace_dir, "dense_normals.npy"), dense_normals)

    print(f"  Dense points saved to {ply_path}")

    return {
        "status": "ok",
        "dense_points": dense_points,
        "dense_normals": dense_normals,
        "depth_map_dir": output_path,
        "reconstruction": reconstruction,
        "message": f"Dense MVS complete: {len(dense_points)} points",
    }


# =========================================================================
# PLY I/O
# =========================================================================

def _save_ply(path: str, points: np.ndarray, normals: Optional[np.ndarray] = None):
    """Save point cloud as PLY (ASCII format)."""
    n = len(points)
    has_n = normals is not None and len(normals) == n
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_n:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            if has_n:
                nx, ny, nz = normals[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {nx:.6f} {ny:.6f} {nz:.6f}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def load_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a PLY point cloud. Returns (points, normals)."""
    with open(path, "r") as f:
        lines = f.readlines()

    # Parse header
    n_verts = 0
    has_normal = False
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
        if "nx" in line:
            has_normal = True
        if line.startswith("end_header"):
            header_end = i + 1
            break

    points = np.zeros((n_verts, 3), dtype=np.float32)
    normals = np.zeros((n_verts, 3), dtype=np.float32) if has_normal else None

    for i, line in enumerate(lines[header_end:header_end + n_verts]):
        parts = line.strip().split()
        points[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        if has_normal:
            normals[i] = [float(parts[3]), float(parts[4]), float(parts[5])]

    return points, normals
