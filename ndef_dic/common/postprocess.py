"""
Post-processing for dense point cloud.

Operations:
  1. Statistical Outlier Removal (SOR)
  2. Normal estimation (PCA on local neighborhood)
  3. Visibility matrix computation (vis_mask)
  4. ROI cropping (optional, cylinder-aware)
  5. Save dense/ outputs for Step 2 consumption
"""

import os
import json
import numpy as np
from typing import Optional, Tuple, List, Dict


# =========================================================================
# PLY I/O (self-contained, mirrors dense_mvs.py)
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


# =========================================================================
# Statistical Outlier Removal (memory-efficient)
# =========================================================================

def statistical_outlier_removal(
    points: np.ndarray,
    k: int = 20,
    std_ratio: float = 2.0,
    chunk_size: int = 500,
    ref_size: int = 2000,
) -> np.ndarray:
    """
    Remove points whose mean distance to k nearest neighbors exceeds
    mean + std_ratio * std.

    Uses approximate nearest-neighbor via random subset sampling,
    with small chunk sizes to limit memory.

    Returns boolean mask: True = inlier, False = outlier.
    """
    n = len(points)
    if n < k:
        return np.ones(n, dtype=bool)

    rng = np.random.default_rng(42)
    ref_size = min(ref_size, n)
    chunk_size = min(chunk_size, n)

    all_distances = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = points[start:end]  # (chunk, 3)

        # Random reference subset (new subset each chunk for better coverage)
        ref_idx = rng.choice(n, ref_size, replace=False)
        ref = points[ref_idx]  # (ref_size, 3)

        # Pairwise distances: (chunk, ref_size)
        diff = chunk[:, None, :] - ref[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        dist = np.sqrt(dist_sq, out=dist_sq)  # reuse memory
        del diff, dist_sq

        k_eff = min(k, ref_size)
        k_dist = np.partition(dist, k_eff, axis=1)[:, k_eff]
        all_distances.append(k_dist)
        del dist, k_dist

    k_dist_all = np.concatenate(all_distances)

    mean_dist = float(np.mean(k_dist_all))
    std_dist = float(np.std(k_dist_all))
    threshold = mean_dist + std_ratio * std_dist

    inlier_mask = k_dist_all <= threshold
    del k_dist_all

    return inlier_mask


def _subsample_points(
    points: np.ndarray,
    normals: Optional[np.ndarray],
    max_points: int,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Randomly subsample points to max_points, returning indices."""
    n = len(points)
    if n <= max_points:
        return points, normals, np.arange(n)
    idx = np.random.default_rng(rng_seed).choice(n, max_points, replace=False)
    idx.sort()
    n_out = normals[idx] if normals is not None and len(normals) == n else None
    return points[idx], n_out, idx


def filter_points(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    intensities: Optional[np.ndarray] = None,
    k: int = 20,
    std_ratio: float = 2.0,
    max_points: int = 500_000,
) -> Dict:
    """
    Apply SOR to a point cloud and return filtered results.

    For clouds larger than max_points, automatically subsamples before SOR
    (the subsample is temporary — the full cloud is returned with outliers
    interpolated from the subsample mask).

    Returns dict with "points", "normals" (if provided),
    "intensities" (if provided), "mask", "num_removed".
    """
    n = len(points)

    if n > max_points:
        # Subsample for SOR, then broadcast mask via nearest-neighbor lookup
        sub_pts, _, sub_idx = _subsample_points(points, None, max_points)
        sub_mask = statistical_outlier_removal(sub_pts, k=k, std_ratio=std_ratio)

        # For each full point, check if it's close to any subsample inlier
        # Simple approach: assign each full point the mask of its nearest subsample
        # Use chunked nearest-neighbor
        full_mask = np.ones(n, dtype=bool)
        chunk_sz = 5000
        for start in range(0, n, chunk_sz):
            end = min(start + chunk_sz, n)
            chunk = points[start:end]
            diff = chunk[:, None, :] - sub_pts[None, :, :]
            nn_idx = np.argmin(np.sum(diff ** 2, axis=-1), axis=1)
            full_mask[start:end] = sub_mask[nn_idx]
            del diff

        mask = full_mask
        del sub_pts, sub_mask, full_mask, sub_idx
    else:
        mask = statistical_outlier_removal(points, k=k, std_ratio=std_ratio)

    n_removed = (~mask).sum()

    result = {
        "points": points[mask],
        "mask": mask,
        "num_removed": int(n_removed),
    }
    if normals is not None and len(normals) == len(points):
        result["normals"] = normals[mask]
    if intensities is not None and len(intensities) == len(points):
        result["intensities"] = intensities[mask]

    return result


# =========================================================================
# Normal estimation
# =========================================================================

def estimate_normals(
    points: np.ndarray,
    k: int = 30,
    orient_toward: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Estimate normals via local PCA on k nearest neighbors.

    For a cylinder, orient_toward can be set to [0, 0, 0] (the cylinder axis)
    so all normals point outward.

    Args:
        points: (N, 3) input points
        k:      Number of neighbors for local PCA
        orient_toward: (3,) reference point to orient normals toward.
                       For cylinder, use [0, 0, 0] (axis center).

    Returns:
        normals: (N, 3) unit normal vectors
    """
    n = len(points)
    if n < k:
        # Not enough points — return cylinder radial normals as fallback
        nrm = points.copy()
        nrm[:, 1] = 0.0
        nrm_norm = np.linalg.norm(nrm, axis=1, keepdims=True)
        return nrm / np.maximum(nrm_norm, 1e-8)

    # Build approximate neighborhood via spatial grid
    # Compute bounding box
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    extent = pmax - pmin

    # Grid cell size based on point density
    vol = np.prod(np.maximum(extent, 1e-6))
    density = n / vol
    cell_size = (k / density) ** (1.0 / 3.0)
    cell_size = max(cell_size, np.mean(extent) * 0.02)

    grid_dims = np.maximum(np.ceil(extent / cell_size).astype(int), 1)
    grid_dims = np.minimum(grid_dims, 100)  # cap at 100 per dimension

    # Assign points to grid cells
    cell_indices = np.floor((points - pmin) / cell_size).astype(int)
    cell_indices = np.clip(cell_indices, 0, grid_dims - 1)

    # Build cell → point index mapping
    n_cells = grid_dims[0] * grid_dims[1] * grid_dims[2]
    cell_to_pts = [[] for _ in range(min(n_cells, 1000000))]
    for i, (cx, cy, cz) in enumerate(cell_indices):
        idx = cx + cy * grid_dims[0] + cz * grid_dims[0] * grid_dims[1]
        if idx < len(cell_to_pts):
            cell_to_pts[idx].append(i)

    # For each point, collect neighbors from adjacent cells
    normals = np.zeros((n, 3), dtype=np.float32)
    search_radius = 1

    chunk_size = 10000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        for pi in range(start, end):
            cx, cy, cz = cell_indices[pi]
            neighbors = []

            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    for dz in range(-search_radius, search_radius + 1):
                        nx, ny, nz = cx + dx, cy + dy, cz + dz
                        if 0 <= nx < grid_dims[0] and 0 <= ny < grid_dims[1] and 0 <= nz < grid_dims[2]:
                            idx = nx + ny * grid_dims[0] + nz * grid_dims[0] * grid_dims[1]
                            if idx < len(cell_to_pts):
                                neighbors.extend(cell_to_pts[idx])

            if len(neighbors) < 3:
                # Fallback: use radial direction for cylinder
                nrm = points[pi].copy()
                nrm[1] = 0.0
                nrm_nrm = np.linalg.norm(nrm)
                normals[pi] = nrm / max(nrm_nrm, 1e-8) if nrm_nrm > 1e-8 else np.array([0.0, 0.0, 1.0])
                continue

            # Local PCA
            neigh_pts = points[neighbors]
            centroid = neigh_pts.mean(axis=0)
            centered = neigh_pts - centroid
            cov = centered.T @ centered

            # Eigen decomposition — smallest eigenvector = normal
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normal = eigenvectors[:, 0]  # smallest eigenvalue
            normal = normal / np.linalg.norm(normal)

            # Orient normal toward reference point
            if orient_toward is not None:
                to_ref = np.array(orient_toward) - points[pi]
                if np.dot(normal, to_ref) < 0:
                    normal = -normal

            normals[pi] = normal

    return normals


# =========================================================================
# Visibility matrix
# =========================================================================

def compute_visibility_matrix(
    points: np.ndarray,
    normals: np.ndarray,
    K_list: List[np.ndarray],
    R_list: List[np.ndarray],
    t_list: List[np.ndarray],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """
    Compute visibility matrix: vis_mask[i, c] = True if point i is visible in camera c.

    Checks:
      1. FOV: projected pixel (u,v) within [0, W) × [0, H)
      2. Front-facing: normal points toward camera (n · view_dir > 0)
      3. Depth: point is in front of camera (Z_cam > 0)

    Args:
        points:       (N, 3) world coordinates
        normals:      (N, 3) unit normals
        K_list:       List of (3, 3) intrinsic matrices
        R_list:       List of (3, 3) rotation matrices (world → camera)
        t_list:       List of (3, 1) translation vectors
        image_width:  Image width in pixels
        image_height: Image height in pixels

    Returns:
        vis_mask: (N, N_cam) bool array
    """
    N = len(points)
    n_cam = len(K_list)
    vis_mask = np.zeros((N, n_cam), dtype=bool)

    for cam_id in range(n_cam):
        K = K_list[cam_id]
        R = R_list[cam_id]
        t = t_list[cam_id].reshape(3, 1)

        # Transform to camera frame
        P_cam = R @ points.T + t  # (3, N)
        Z = P_cam[2, :]

        # Depth check
        in_front = Z > 1e-6

        # Front-facing check: camera is at origin in camera frame,
        # so view direction is -P_cam (from point to camera)
        # In world coords, camera center C = -R.T @ t
        C = -R.T @ t  # (3, 1)

        # Normal in world coords dotted with view direction
        view_dir = C.T - points  # (N, 3)
        view_dir_norm = np.linalg.norm(view_dir, axis=1, keepdims=True)
        view_dir = view_dir / np.maximum(view_dir_norm, 1e-8)
        cos_angle = np.sum(normals * view_dir, axis=1)
        front_facing = cos_angle > 0.05  # ~87° max grazing

        # Project
        xn = P_cam[0, :] / Z
        yn = P_cam[1, :] / Z

        u = K[0, 0] * xn + K[0, 1] * yn + K[0, 2]
        v = K[1, 0] * xn + K[1, 1] * yn + K[1, 2]

        in_fov = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)

        vis_mask[:, cam_id] = in_front & front_facing & in_fov

    return vis_mask


# =========================================================================
# ROI cropping (cylinder-specific)
# =========================================================================

def crop_to_cylinder_roi(
    points: np.ndarray,
    cylinder_radius: float,
    cylinder_height: float,
    margin_mm: float = 5.0,
) -> np.ndarray:
    """
    Crop points to a cylindrical ROI.

    Keeps points within [R - margin, R + margin] radially
    and within [-H/2 - margin, H/2 + margin] vertically.

    Returns boolean mask.
    """
    r = np.sqrt(points[:, 0] ** 2 + points[:, 2] ** 2)
    y = points[:, 1]

    r_ok = (r >= cylinder_radius - margin_mm) & (r <= cylinder_radius + margin_mm)
    y_ok = (y >= -cylinder_height / 2 - margin_mm) & (y <= cylinder_height / 2 + margin_mm)

    return r_ok & y_ok


# =========================================================================
# Main post-processing entry point
# =========================================================================

def run_postprocess(
    points: np.ndarray,
    normals: Optional[np.ndarray],
    K_list: List[np.ndarray],
    R_list: List[np.ndarray],
    t_list: List[np.ndarray],
    image_width: int,
    image_height: int,
    output_dir: str,
    config: Optional[Dict] = None,
) -> Dict:
    """
    Run full post-processing pipeline on a point cloud.

    For large point clouds (> subsample_target), automatically subsamples
    before SOR and visibility computation to keep processing tractable.

    Args:
        points:       (N, 3) input point cloud
        normals:      (N, 3) or None — if None, will be estimated
        K_list, etc:  Camera parameters for visibility
        image_width, image_height: Image dimensions
        output_dir:   Output directory for dense/ files
        config:       Optional dict with keys:
                        - "subsample_target" (default 200_000): max points to process
                        - "skip_sor" (default False for synthetic data)
                        - "sor_k", "sor_std_ratio"
                        - "normal_k"
                        - "cylinder_radius", "cylinder_height"
                        - "skip_roi"

    Returns:
        dict with keys: points, normals, vis_mask, stats
    """
    cfg = config or {}
    os.makedirs(output_dir, exist_ok=True)

    n_original = len(points)
    print(f"[Step 1 Post] Input: {n_original} points")

    # ---- 0. Subsample if needed ----
    subsample_target = cfg.get("subsample_target", 200_000)
    if n_original > subsample_target:
        idx = np.random.default_rng(42).choice(n_original, subsample_target, replace=False)
        idx.sort()
        points_w = points[idx]
        normals_w = normals[idx] if normals is not None and len(normals) == n_original else None
        print(f"[Step 1 Post] Subsampled: {n_original} → {len(points_w)} points")
        subsample_idx = idx
    else:
        points_w = points
        normals_w = normals
        subsample_idx = np.arange(n_original)

    # ---- 1. SOR filtering (optional — skip for synthetic data) ----
    if cfg.get("skip_sor", False):
        points_f = points_w
        normals_f = normals_w
        n_removed_sor = 0
        print(f"[Step 1 Post] SOR skipped (ground truth data)")
    else:
        sor_k = cfg.get("sor_k", 20)
        sor_std = cfg.get("sor_std_ratio", 2.0)
        result = filter_points(points_w, normals_w, k=sor_k, std_ratio=sor_std,
                               max_points=subsample_target)
        points_f = result["points"]
        normals_f = result.get("normals")
        n_removed_sor = result["num_removed"]
        print(f"[Step 1 Post] SOR removed {n_removed_sor} outliers, "
              f"{len(points_f)} points remaining")

    # ---- 2. ROI cropping (if cylinder params provided) ----
    if not cfg.get("skip_roi", False) and "cylinder_radius" in cfg:
        roi_mask = crop_to_cylinder_roi(
            points_f,
            cfg["cylinder_radius"],
            cfg.get("cylinder_height", 120.0),
            cfg.get("roi_margin", 5.0),
        )
        points_f = points_f[roi_mask]
        if normals_f is not None:
            normals_f = normals_f[roi_mask]
        print(f"[Step 1 Post] ROI crop: {len(points_f)} points remaining")

    # ---- 3. Normal estimation (if not provided) ----
    if normals_f is None or len(normals_f) != len(points_f):
        # PCA gives normals up to a sign ambiguity.
        # Fix: align with radial direction (cylinder surface normals point outward).
        normals_f = estimate_normals(points_f, k=cfg.get("normal_k", 30),
                                     orient_toward=None)
        # Post-correct: flip any normal pointing toward the cylinder axis (center)
        # For cylinder, the radial (outward) direction at point (x,y,z) is (x,0,z).
        pts_radial = points_f.copy()
        pts_radial[:, 1] = 0.0  # zero out Y component
        dot_radial = np.sum(normals_f * pts_radial, axis=1)
        # Flip normals that point inward (negative dot with radial direction)
        flip_mask = dot_radial < 0
        normals_f[flip_mask] = -normals_f[flip_mask]
        n_flipped = flip_mask.sum()
        print(f"[Step 1 Post] Normals estimated via PCA (k={cfg.get('normal_k', 30)}), "
              f"{n_flipped}/{len(normals_f)} flipped to outward")
        print(f"[Step 1 Post] Normals estimated via PCA (k={cfg.get('normal_k', 30)})")

    # ---- 4. Visibility matrix ----
    n_cam = len(K_list)
    print(f"[Step 1 Post] Computing visibility matrix ({len(points_f)} × {n_cam})...")
    vis_mask = compute_visibility_matrix(
        points_f, normals_f,
        K_list, R_list, t_list,
        image_width, image_height,
    )
    n_visible = vis_mask.sum(axis=1)
    mean_vis = n_visible.mean()
    print(f"[Step 1 Post] Mean visible cameras per point: {mean_vis:.1f}")

    # ---- 5. Save outputs ----
    _save_ply(os.path.join(output_dir, "dense_points.ply"), points_f, normals_f)
    np.save(os.path.join(output_dir, "dense_normals.npy"), normals_f)
    np.save(os.path.join(output_dir, "vis_mask.npy"), vis_mask)

    # Meta
    meta = {
        "num_points_original": n_original,
        "num_points_filtered": len(points_f),
        "num_removed_sor": n_removed_sor,
        "subsample_target": subsample_target,
        "mean_visible_cameras": float(mean_vis),
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Step 1 Post] Outputs saved to {output_dir}/")
    print(f"  dense_points.ply, dense_normals.npy, vis_mask.npy, meta.json")

    return {
        "points": points_f,
        "normals": normals_f,
        "vis_mask": vis_mask,
        "stats": meta,
    }
