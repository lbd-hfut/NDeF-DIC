"""
Scale Correction for COLMAP SfM output.

COLMAP SfM recovers camera poses and 3D points only up to a similarity
transform — the absolute scale is arbitrary. This module corrects that:

  X_physical = s * X_colmap

For uniform scaling by factor s:
  - Camera translations:  t_new = s * t_old
  - 3D points:            pts_new = s * pts_old
  - R, K, dist:           unchanged

Supported methods:
  - manual:         user provides scale factor directly
  - auto_cylinder:  fit a cylinder to sparse points, compare radius to known value
  - checkerboard:   detect checkerboard in images, triangulate, measure (TODO)
"""

import os
import numpy as np
from typing import Optional, Dict
from scipy.io import loadmat, savemat
from scipy.optimize import least_squares

from ..common.mat_io import unwrap_mat_cell, unwrap_mat_batch


# =========================================================================
# Scale computation
# =========================================================================

def compute_scale_auto_cylinder(
    points3D: np.ndarray,
    known_radius_mm: float,
    axis_hint: Optional[str] = None,
) -> float:
    """
    Fit a cylinder to sparse SfM points and compute the scale factor.

    Algorithm:
      1. If axis_hint is given, use it directly. Otherwise, use PCA to find
         the dominant axis — the axis with largest spread (cylinder height).
      2. Project points onto plane perpendicular to the axis.
      3. Fit a circle (cx, cy, r) to the projected 2D points via least squares.
      4. scale = known_radius_mm / fitted_radius.

    Args:
        points3D: (N, 3) sparse 3D points in COLMAP (arbitrary-scale) coordinates.
        known_radius_mm: Known physical radius of the cylinder in mm.
        axis_hint: Optional cylinder axis direction. If not provided, PCA is used.
                   E.g., axis_hint="y" → np.array([0, 1, 0]).

    Returns:
        Scale factor s such that X_physical = s * X_colmap.
    """
    if len(points3D) < 10:
        raise ValueError(
            f"Need at least 10 points for cylinder fit, got {len(points3D)}"
        )

    pts = np.asarray(points3D, dtype=np.float64)
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid

    # ---- 1. Find cylinder axis ----
    if axis_hint is not None:
        axis_map = {"x": 0, "y": 1, "z": 2}
        idx = axis_map.get(axis_hint.lower(), 1)
        axis = np.zeros(3)
        axis[idx] = 1.0
    else:
        # PCA: the axis with largest variance is the cylinder height direction
        cov = np.cov(pts_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigenvector with largest eigenvalue (last after eigh sort)
        axis = eigenvectors[:, -1]
        # Normalize
        axis = axis / np.linalg.norm(axis)
        print(f"  [ScaleCorr] PCA cylinder axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")

    # ---- 2. Project points to plane perpendicular to axis ----
    # Build orthonormal basis for the plane
    # Find any vector not parallel to axis
    if abs(axis[0]) < 0.9:
        u = np.cross(axis, [1, 0, 0])
    else:
        u = np.cross(axis, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    v = v / np.linalg.norm(v)

    # Project: (u·p, v·p) for each point
    proj_u = pts_centered @ u
    proj_v = pts_centered @ v

    # ---- 3. Fit circle: minimize sum_i (||p_i - c|| - r)^2 ----
    def circle_residuals(params):
        cx, cy, r = params
        dx = proj_u - cx
        dy = proj_v - cy
        dist = np.sqrt(dx**2 + dy**2)
        return dist - r

    # Initial guess: centroid of projections, mean distance as radius
    cx0 = np.median(proj_u)
    cy0 = np.median(proj_v)
    r0 = np.median(np.sqrt((proj_u - cx0)**2 + (proj_v - cy0)**2))

    result = least_squares(circle_residuals, [cx0, cy0, r0], method="lm")
    _, _, fitted_radius = result.x
    fitted_radius = abs(fitted_radius)  # radius is positive

    scale = known_radius_mm / fitted_radius
    print(f"  [ScaleCorr] Fitted cylinder radius: {fitted_radius:.4f} (colmap units)")
    print(f"  [ScaleCorr] Known physical radius: {known_radius_mm:.2f} mm")
    print(f"  [ScaleCorr] Scale factor: {scale:.4f}")

    # Quick sanity check
    if scale < 0.1 or scale > 1000:
        print(f"  [ScaleCorr] WARNING: scale factor {scale:.2f} seems extreme. "
              f"Check known_radius_mm ({known_radius_mm}) and point cloud.")

    return float(scale)


def compute_scale_checkerboard(
    image_paths,
    K_list,
    dist_list,
    square_size_mm: float,
    pattern_size,
):
    """
    [STUB] Compute scale from a checkerboard of known physical size.

    This method detects checkerboard corners in the reference images,
    triangulates them using COLMAP camera parameters, and computes the
    scale factor by comparing measured square sizes with the known size.

    Not yet implemented — requires OpenCV checkerboard detection +
    multi-view triangulation.

    Args:
        image_paths: List of paths to reference images.
        K_list: Camera intrinsic matrices.
        dist_list: Distortion coefficients.
        square_size_mm: Physical size of one checkerboard square in mm.
        pattern_size: (rows, cols) number of inner corners.

    Returns:
        Scale factor, or None if detection failed.
    """
    raise NotImplementedError(
        "Checkerboard scale correction is not yet implemented. "
        "Use method='manual' or method='auto_cylinder' instead."
    )


def compute_scale_from_camera_distance(
    calib_dir: str,
    known_working_distance_mm: float,
) -> float:
    """
    Compute scale factor from known working distance (camera-to-object).

    COLMAP anchors the world coordinate frame on the first registered
    camera (R₀ ≈ I, t₀ ≈ 0).  The world origin is therefore near the
    optical centre of camera 0, and the centroid of the sparse point
    cloud is a natural, geometry-agnostic estimate of the object centre.

    This method:
      1. Loads cameras.mat and extracts camera centres C_i = -R_iᵀ · t_i.
      2. Computes the centroid of the sparse 3-D points as the object centre.
      3. Averages the Euclidean distances ‖C_i − centroid‖ across cameras.
      4. Returns scale = known_distance / mean_colmap_distance.

    Because it uses simple Euclidean distance to the centroid, it works
    for **arbitrary object geometry** — cylinders, plates, beams, etc.

    .. note::
       The user must measure ``known_working_distance_mm`` consistently:
       as the mean camera-to-object-centre distance.  For a multi-camera
       rig this is typically the radius of the camera ring.

    Args:
        calib_dir: Path to calibration directory.
        known_working_distance_mm: Known physical working distance in mm
            (mean camera-to-object-centre distance).

    Returns:
        Scale factor s such that X_physical = s * X_colmap.
    """
    cameras_path = os.path.join(calib_dir, "cameras.mat")
    if not os.path.exists(cameras_path):
        raise FileNotFoundError(f"cameras.mat not found: {cameras_path}")

    pts_path = os.path.join(calib_dir, "points3D.mat")

    calib = loadmat(cameras_path)
    n_cam = int(calib["num_cameras"][0, 0])

    # ---- Camera centres: C = -Rᵀ · t ----
    R_batch = unwrap_mat_batch(calib["cam_from_world_R"], (3, 3))
    t_batch = unwrap_mat_batch(calib["cam_from_world_t"], (3, 1))

    centers = np.array([
        (-R_batch[i].T @ t_batch[i]).ravel() for i in range(n_cam)
    ])  # (N_cam, 3)
    print(f"  [ScaleCorr] Camera centres: mean dist from origin = "
          f"{np.linalg.norm(centers, axis=1).mean():.1f} (colmap units)")

    # ---- Object centre = centroid of sparse points ----
    if os.path.exists(pts_path):
        pts_data = loadmat(pts_path)
        pts_arr = unwrap_mat_batch(pts_data["points3D"], (3,))
        centroid = pts_arr.mean(axis=0)
        print(f"  [ScaleCorr] Sparse-point centroid: "
              f"[{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}]")
    else:
        # Fallback: assume origin (camera 0 is typically [I | 0])
        centroid = np.array([0.0, 0.0, 0.0])
        print(f"  [ScaleCorr] No sparse points — using origin as object centre")

    # ---- Mean Euclidean distance (no axis projection — works for any shape) ----
    cam_dists = np.linalg.norm(centers - centroid, axis=1)
    mean_dist = float(cam_dists.mean())
    print(f"  [ScaleCorr] Mean camera-to-centroid distance: {mean_dist:.1f} (colmap units)")

    scale = known_working_distance_mm / mean_dist
    print(f"  [ScaleCorr] Known working distance: {known_working_distance_mm:.1f} mm")
    print(f"  [ScaleCorr] Scale factor: {scale:.4f}")

    if scale < 0.1 or scale > 10:
        print(f"  [ScaleCorr] WARNING: scale factor {scale:.4f} seems extreme.")

    return float(scale)


# =========================================================================
# Apply correction to calibration files
# =========================================================================

def apply_scale_to_calibration(
    calib_dir: str,
    scale: float,
    verbose: bool = True,
) -> Dict:
    """
    Apply uniform scale correction to cameras.mat and points3D.mat.

    Modifies the .mat files in-place:
      - cameras.mat: scales cam_from_world_t by `scale`
      - points3D.mat: scales points3D by `scale`

    Also updates P_list (projection matrices) to reflect new t vectors.

    Args:
        calib_dir: Path to calibration directory containing .mat files.
        scale: Scale factor s such that X_physical = s * X_colmap.
        verbose: Print summary statistics.

    Returns:
        dict with keys: scale, old_bbox, new_bbox.
    """
    if scale == 1.0:
        if verbose:
            print("  [ScaleCorr] scale=1.0, no correction needed.")
        return {"scale": 1.0, "old_bbox": None, "new_bbox": None}

    cameras_path = os.path.join(calib_dir, "cameras.mat")
    points_path = os.path.join(calib_dir, "points3D.mat")

    if not os.path.exists(cameras_path):
        raise FileNotFoundError(f"cameras.mat not found: {cameras_path}")

    # ---- Load cameras ----
    calib = loadmat(cameras_path)
    n_cam = int(calib["num_cameras"][0, 0])

    # Unwrap once — handles both old (object) and new (numeric) .mat formats
    K_batch = unwrap_mat_batch(calib["K_list"], (3, 3))           # (N, 3, 3)
    R_batch = unwrap_mat_batch(calib["cam_from_world_R"], (3, 3)) # (N, 3, 3)
    t_batch = unwrap_mat_batch(calib["cam_from_world_t"], (3, 1)) # (N, 3, 1)

    # ---- Scale t vectors ----
    t_scaled = t_batch * scale

    # Write back clean float64 arrays (migrates old object-format files)
    calib["cam_from_world_t"] = t_scaled.astype(np.float64)
    calib["K_list"] = K_batch.astype(np.float64)
    calib["cam_from_world_R"] = R_batch.astype(np.float64)

    # ---- Update P_list ----
    P_list = []
    for i in range(n_cam):
        K = K_batch[i]
        R = R_batch[i]
        t_i = t_scaled[i].reshape(3, 1)
        P_list.append(K @ np.hstack((R, t_i)))

    P_new = np.stack(P_list, axis=0).astype(np.float64)
    calib["P_list"] = P_new

    # ---- Save cameras ----
    savemat(cameras_path, calib)

    # ---- Load and scale points3D ----
    old_bbox = None
    new_bbox = None

    if os.path.exists(points_path):
        pts_data = loadmat(points_path)
        if "points3D" in pts_data:
            pts_arr = unwrap_mat_batch(pts_data["points3D"], (3,))  # (N, 3)

            if len(pts_arr) > 0:
                old_bbox = (pts_arr.min(axis=0).tolist(), pts_arr.max(axis=0).tolist())
                pts_arr *= scale
                new_bbox = (pts_arr.min(axis=0).tolist(), pts_arr.max(axis=0).tolist())

            pts_data["points3D"] = pts_arr.astype(np.float64)
            savemat(points_path, pts_data)

    if verbose:
        print(f"  [ScaleCorr] Applied scale={scale:.4f} to {calib_dir}")
        if old_bbox and new_bbox:
            print(f"  [ScaleCorr] Points bbox (min): "
                  f"[{old_bbox[0][0]:.1f}, {old_bbox[0][1]:.1f}, {old_bbox[0][2]:.1f}] → "
                  f"[{new_bbox[0][0]:.1f}, {new_bbox[0][1]:.1f}, {new_bbox[0][2]:.1f}]")
            print(f"  [ScaleCorr] Points bbox (max): "
                  f"[{old_bbox[1][0]:.1f}, {old_bbox[1][1]:.1f}, {old_bbox[1][2]:.1f}] → "
                  f"[{new_bbox[1][0]:.1f}, {new_bbox[1][1]:.1f}, {new_bbox[1][2]:.1f}]")

    return {"scale": scale, "old_bbox": old_bbox, "new_bbox": new_bbox}


# =========================================================================
# Main entry point
# =========================================================================

def run_scale_correction(
    calib_dir: str,
    method: str = "manual",
    scale: float = 1.0,
    cylinder_radius: float = 80.0,
    working_distance: float = 300.0,
    image_paths=None,
    K_list=None,
    dist_list=None,
    checkerboard_square_size: float = 10.0,
    checkerboard_rows: int = 8,
    checkerboard_cols: int = 11,
    verbose: bool = True,
) -> Dict:
    """
    Compute and apply scale correction to COLMAP calibration.

    This is the main entry point called from the Step 1 pipeline.

    Args:
        calib_dir: Path to calibration directory.
        method: "manual", "auto_cylinder", "from_camera_distance", or "checkerboard".
        scale: Scale factor for "manual" method.
        cylinder_radius: Known physical radius in mm (for "auto_cylinder").
        working_distance: Known camera-to-object distance in mm
                          (for "from_camera_distance").
        image_paths: List of reference image paths (for "checkerboard").
        K_list: Camera intrinsics (for "checkerboard").
        dist_list: Distortion coefficients (for "checkerboard").
        checkerboard_square_size: Square size in mm (for "checkerboard").
        checkerboard_rows: Number of inner corner rows (for "checkerboard").
        checkerboard_cols: Number of inner corner cols (for "checkerboard").
        verbose: Print progress.

    Returns:
        dict with keys: method, scale, old_bbox, new_bbox.
    """
    if verbose:
        print(f"\n[ScaleCorr] Computing scale correction (method={method})")

    if method == "manual":
        scale_factor = float(scale)

    elif method == "auto_cylinder":
        # Load sparse points from COLMAP output
        pts_path = os.path.join(calib_dir, "points3D.mat")
        if not os.path.exists(pts_path):
            raise FileNotFoundError(
                f"points3D.mat not found at {pts_path}. "
                f"Cannot auto-compute cylinder scale without sparse points."
            )
        pts_data = loadmat(pts_path)
        pts_arr = unwrap_mat_batch(pts_data["points3D"], (3,))

        scale_factor = compute_scale_auto_cylinder(pts_arr, cylinder_radius)

    elif method == "from_camera_distance":
        scale_factor = compute_scale_from_camera_distance(
            calib_dir=calib_dir,
            known_working_distance_mm=working_distance,
        )

    elif method == "checkerboard":
        scale_factor = compute_scale_checkerboard(
            image_paths=image_paths,
            K_list=K_list,
            dist_list=dist_list,
            square_size_mm=checkerboard_square_size,
            pattern_size=(checkerboard_rows, checkerboard_cols),
        )
        # returned None if detection failed; handle gracefully
        if scale_factor is None:
            print("  [ScaleCorr] Checkerboard detection failed, no scale applied.")
            return {"method": method, "scale": 1.0, "old_bbox": None, "new_bbox": None}

    else:
        raise ValueError(
            f"Unknown scale_correction method: {method}. "
            f"Valid options: manual, auto_cylinder, checkerboard."
        )

    # Apply
    result = apply_scale_to_calibration(calib_dir, scale_factor, verbose=verbose)
    result["method"] = method
    return result
