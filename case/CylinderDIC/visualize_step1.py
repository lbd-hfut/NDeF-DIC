#!/usr/bin/env python
"""
Visualize Step 1 dense output: point cloud, normals, visibility, camera poses.

Generates:
  1. 3D point cloud colored by per-camera visibility count
  2. Camera pose visualization (frustum wireframes)
  3. Per-camera visibility heatmap (projected onto cylinder)
  4. Visibility histogram + coverage statistics
  5. Orthogonal projections (XY, XZ, YZ) with visibility

Usage:
  cd case/CylinderDIC
  python visualize_step1.py

  # Or from project root:
  python case/CylinderDIC/visualize_step1.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat

# --- Resolve paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DENSE_DIR = os.path.join(SCRIPT_DIR, "calibration", "dense")
CALIB_DIR = os.path.join(SCRIPT_DIR, "calibration")
REPORT_DIR = os.path.join(SCRIPT_DIR, "report")

os.makedirs(REPORT_DIR, exist_ok=True)

DPI = 150
np.random.seed(42)  # deterministic subsampling

# =========================================================================
# Data loading
# =========================================================================

def load_dense_data(dense_dir: str):
    """Load all Step 1 dense outputs."""
    print(f"[LOAD] dense_dir = {dense_dir}")

    # Points from PLY
    ply_path = os.path.join(dense_dir, "dense_points.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Not found: {ply_path}")

    pts = _read_ply(ply_path)
    print(f"  Points:  {pts.shape}")

    # Normals
    nrm_path = os.path.join(dense_dir, "dense_normals.npy")
    nrm = np.load(nrm_path) if os.path.exists(nrm_path) else None
    if nrm is not None:
        print(f"  Normals: {nrm.shape}")

    # Visibility
    vis_path = os.path.join(dense_dir, "vis_mask.npy")
    vis = np.load(vis_path) if os.path.exists(vis_path) else None
    if vis is not None:
        print(f"  Vis mask: {vis.shape}")

    # Meta
    meta_path = os.path.join(dense_dir, "meta.json")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    return pts, nrm, vis, meta


def _read_ply(path: str) -> np.ndarray:
    """Read PLY point cloud, return (N, 3)."""
    with open(path, "r") as f:
        lines = f.readlines()

    n_verts = 0
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
        if line.startswith("end_header"):
            header_end = i + 1
            break

    pts = np.zeros((n_verts, 3), dtype=np.float64)
    for i, line in enumerate(lines[header_end:header_end + n_verts]):
        parts = line.strip().split()
        pts[i] = [float(parts[0]), float(parts[1]), float(parts[2])]

    return pts


def load_cameras(calib_dir: str):
    """Load camera parameters from cameras.mat."""
    cpath = os.path.join(calib_dir, "cameras.mat")
    if not os.path.exists(cpath):
        print(f"[WARN] cameras.mat not found at {cpath}")
        return None

    data = loadmat(cpath)
    n_cam = int(data["num_cameras"][0, 0])

    def _extract(arr, idx, shape):
        item = arr[idx]
        if arr.dtype == object:
            flat = np.array([float(item.flat[k].item()) for k in range(item.size)])
            return flat.reshape(shape)
        return np.array(item).reshape(shape)

    K_list = [_extract(data["K_list"], i, (3, 3)) for i in range(n_cam)]
    R_list = [_extract(data["cam_from_world_R"], i, (3, 3)) for i in range(n_cam)]
    t_raw = data["cam_from_world_t"]

    t_list = []
    for i in range(n_cam):
        if t_raw.dtype == object:
            item = t_raw[i]
            if hasattr(item, 'flat'):
                flat = [float(item.flat[k].item()) for k in range(item.size)]
            else:
                flat = np.array(item).ravel().tolist()
            t_list.append(np.array(flat).reshape(3,))
        else:
            ti = np.array(t_raw[i]).ravel()
            t_list.append(ti[:3])

    print(f"  Cameras: {n_cam}")

    # Compute world-space camera centers
    centers = np.array([
        (-R.T @ t) for R, t in zip(R_list, t_list)
    ])
    return {
        "K_list": K_list,
        "R_list": R_list,
        "t_list": [t.reshape(3, 1) for t in t_list],
        "centers": centers,
        "num_cameras": n_cam,
    }


def subsample(pts, vis, nrm, target=15000):
    """Randomly subsample to target points."""
    if len(pts) <= target:
        return pts, vis, nrm
    idx = np.random.choice(len(pts), target, replace=False)
    return pts[idx], (vis[idx] if vis is not None else None), (nrm[idx] if nrm is not None else None)


# =========================================================================
# Figure 1: 3D point cloud colored by visibility count
# =========================================================================

def fig1_point_cloud_visibility(pts, vis, calib, report_dir):
    """3D scatter: points colored by n_visible_cameras."""
    print("[FIG 1] Point cloud colored by visibility...")

    n_vis = vis.sum(axis=1)  # (N,) per-point visible camera count
    pts_s, n_vis_s, _ = subsample(pts, n_vis, None, target=15000)

    fig = plt.figure(figsize=(16, 7))

    # --- Subplot 1: 3D view ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    scatter = ax1.scatter(
        pts_s[:, 0], pts_s[:, 1], pts_s[:, 2],
        c=n_vis_s, cmap='plasma', s=1.5, alpha=0.85, vmin=0, vmax=vis.shape[1]
    )
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.6, pad=0.08)
    cbar.set_label('Visible Cameras', fontsize=10)

    # Plot camera centers
    if calib:
        centers = calib["centers"]
        ax1.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                    c='red', s=60, marker='^', edgecolors='black', linewidth=0.3,
                    label=f'{calib["num_cameras"]} Cameras')
        ax1.legend(fontsize=8)

    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'Point Cloud Visibility ({len(pts_s)} / {len(pts)} points sampled)',
                  fontsize=10, fontweight='bold')
    ax1.view_init(elev=25, azim=-55)

    # --- Subplot 2: Visibility histogram ---
    ax2 = fig.add_subplot(1, 2, 2)
    counts, bins, patches = ax2.hist(
        n_vis, bins=np.arange(-0.5, vis.shape[1] + 1.5, 1),
        color='steelblue', edgecolor='white', alpha=0.85, density=True
    )
    ax2.axvline(n_vis.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean = {n_vis.mean():.1f}')
    ax2.axvline(np.median(n_vis), color='orange', linestyle='--', linewidth=1.5,
                label=f'Median = {np.median(n_vis):.0f}')
    ax2.set_xlabel('Visible Cameras')
    ax2.set_ylabel('Density')
    ax2.set_title('Per-Point Visibility Distribution', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(vis.shape[1] + 1))

    fig.suptitle('Step 1 Dense Output — Visibility Analysis', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    path = os.path.join(report_dir, "step1_visibility.png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → {path}")


# =========================================================================
# Figure 2: Camera pose visualization
# =========================================================================

def fig2_camera_poses(pts, calib, report_dir):
    """Show cameras as oriented frustum wireframes around the point cloud."""
    print("[FIG 2] Camera poses...")
    if calib is None:
        print("  [SKIP] No calibration data")
        return

    pts_s, _, _ = subsample(pts, None, None, target=5000)
    centers = calib["centers"]
    R_list = calib["R_list"]
    t_list = calib["t_list"]
    K_list = calib["K_list"]
    n_cam = calib["num_cameras"]

    # Estimate a reasonable frustum size
    pts_center = pts.mean(axis=0)
    scale = np.linalg.norm(pts.std(axis=0)) * 0.15  # frustum size proportional to point spread

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Point cloud (faint)
    ax.scatter(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2],
               c='gray', s=0.5, alpha=0.4, label='Point Cloud')

    # Draw each camera
    for i in range(n_cam):
        R_c2w = R_list[i].T           # world_from_cam rotation
        center = centers[i]            # camera center in world

        # Camera axes in world frame
        axis_len = scale
        x_axis = R_c2w[:, 0] * axis_len
        y_axis = R_c2w[:, 1] * axis_len
        z_axis = R_c2w[:, 2] * axis_len  # camera forward (z-forward = OpenCV convention)

        ax.quiver(center[0], center[1], center[2],
                  x_axis[0], x_axis[1], x_axis[2],
                  color='red', linewidth=0.8, alpha=0.8, arrow_length_ratio=0.15)
        ax.quiver(center[0], center[1], center[2],
                  y_axis[0], y_axis[1], y_axis[2],
                  color='green', linewidth=0.8, alpha=0.8, arrow_length_ratio=0.15)
        ax.quiver(center[0], center[1], center[2],
                  z_axis[0], z_axis[1], z_axis[2],
                  color='blue', linewidth=0.8, alpha=0.8, arrow_length_ratio=0.15)

        # Label camera
        ax.text(center[0], center[1], center[2], f' {i}',
                fontsize=7, color='darkred', fontweight='bold')

        # Draw frustum quad (approximate image plane corners)
        fov_scale = scale * 1.8
        img_w, img_h = 1440, 1080
        f = (K_list[i][0, 0] + K_list[i][1, 1]) / 2.0
        aspect = img_w / img_h
        hw = fov_scale * 0.5
        hh = hw / aspect
        corners_local = np.array([
            [-hw, -hh, fov_scale],
            [ hw, -hh, fov_scale],
            [ hw,  hh, fov_scale],
            [-hw,  hh, fov_scale],
        ])
        corners_world = (R_c2w @ corners_local.T).T + center

        # Wireframe
        indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for a, b in indices:
            ax.plot(
                [center[0], corners_world[a, 0]], [center[1], corners_world[a, 1]], [center[2], corners_world[a, 2]],
                color='gray', linewidth=0.4, alpha=0.5
            )
            ax.plot(
                [corners_world[a, 0], corners_world[b, 0]],
                [corners_world[a, 1], corners_world[b, 1]],
                [corners_world[a, 2], corners_world[b, 2]],
                color='gray', linewidth=0.4, alpha=0.5
            )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=1, label='Cam X (right)'),
        Line2D([0], [0], color='green', lw=1, label='Cam Y (down)'),
        Line2D([0], [0], color='blue', lw=1, label='Cam Z (forward)'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
    ax.set_title(f'Camera Poses ({n_cam} cameras) + Point Cloud', fontweight='bold')
    ax.view_init(elev=30, azim=-50)

    plt.tight_layout()
    path = os.path.join(report_dir, "step1_camera_poses.png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → {path}")


# =========================================================================
# Figure 3: Per-camera visibility
# =========================================================================

def fig3_per_camera_visibility(pts, vis, calib, report_dir):
    """Show which cameras see which regions via per-camera subplots."""
    print("[FIG 3] Per-camera visibility...")

    n_cam = vis.shape[1]
    n_cols = min(4, n_cam)
    n_rows = int(np.ceil(n_cam / n_cols))
    target_pts = 5000

    pts_s, vis_s, _ = subsample(pts, vis, None, target=target_pts)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes_flat = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for cam_id in range(n_cam):
        ax = axes_flat[cam_id]
        visible = vis_s[:, cam_id]
        n_visible = visible.sum()

        # Visible points in blue, invisible in light gray
        ax.scatter(pts_s[~visible, 0], pts_s[~visible, 2],
                   c='lightgray', s=0.5, alpha=0.25, rasterized=True)
        ax.scatter(pts_s[visible, 0], pts_s[visible, 2],
                   c='steelblue', s=1.5, alpha=0.6, rasterized=True)

        coverage = 100.0 * n_visible / len(pts_s)
        ax.set_title(f'Cam {cam_id}:  {n_visible}/{len(pts_s)}  ({coverage:.1f}%)',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('X (mm)', fontsize=7)
        ax.set_ylabel('Z (mm)', fontsize=7)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for j in range(n_cam, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle('Per-Camera Visibility (X-Z Projection)', fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()
    path = os.path.join(report_dir, "step1_per_camera_vis.png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → {path}")


# =========================================================================
# Figure 4: Orthogonal projections + normals
# =========================================================================

def fig4_projections(pts, vis, nrm, report_dir):
    """XY, XZ, YZ projections colored by visibility. Normals if available."""
    print("[FIG 4] Orthogonal projections...")

    n_vis = vis.sum(axis=1)
    pts_s, n_vis_s, nrm_s = subsample(pts, n_vis, nrm, target=15000)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    cmap = 'plasma'

    views = [
        (0, 1, 'X-Y (Top View)', axes[0, 0]),
        (0, 2, 'X-Z (Front View)', axes[0, 1]),
        (1, 2, 'Y-Z (Side View)', axes[0, 2]),
    ]

    for idx_x, idx_y, title, ax in views:
        sc = ax.scatter(pts_s[:, idx_x], pts_s[:, idx_y],
                        c=n_vis_s, cmap=cmap, s=2, alpha=0.75, vmin=0, vmax=vis.shape[1])
        ax.set_xlabel(['X (mm)', 'Y (mm)', 'Z (mm)'][idx_x])
        ax.set_ylabel(['X (mm)', 'Y (mm)', 'Z (mm)'][idx_y])
        ax.set_title(title, fontweight='bold')
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, label='Visible Cameras', shrink=0.8)

    # Histogram of visibility
    ax = axes[1, 0]
    ax.hist(n_vis, bins=np.arange(-0.5, vis.shape[1] + 1.5, 1),
            color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(n_vis.mean(), color='red', linestyle='--', label=f'μ={n_vis.mean():.1f}')
    ax.set_xlabel('Visible Cameras'); ax.set_ylabel('Count')
    ax.set_title('Visibility Histogram', fontweight='bold')
    ax.legend(fontsize=8)

    # Box plot per camera
    ax = axes[1, 1]
    per_cam_vis = vis.sum(axis=0) / len(vis) * 100  # coverage %
    bars = ax.bar(range(len(per_cam_vis)), per_cam_vis, color='steelblue', edgecolor='white')
    ax.axhline(np.mean(per_cam_vis), color='red', linestyle='--',
               label=f'Mean coverage: {np.mean(per_cam_vis):.1f}%')
    ax.set_xlabel('Camera ID'); ax.set_ylabel('Coverage (%)')
    ax.set_title('Per-Camera Coverage', fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    # Normals if available
    ax = axes[1, 2]
    if nrm_s is not None:
        # Show normal directions on a subset
        n_q = min(500, len(pts_s))
        idx = np.random.choice(len(pts_s), n_q, replace=False)
        q = ax.quiver(pts_s[idx, 0], pts_s[idx, 2],
                      nrm_s[idx, 0], nrm_s[idx, 2],
                      scale=15, width=0.003, alpha=0.6, color='steelblue')
        ax.scatter(pts_s[idx, 0], pts_s[idx, 2],
                   c=n_vis_s[idx], cmap=cmap, s=3, alpha=0.5)
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Z (mm)')
        ax.set_title(f'Surface Normals (X-Z, {n_q} pts)', fontweight='bold')
        ax.set_aspect('equal')
    else:
        ax.text(0.5, 0.5, 'No normals available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_title('Normals', fontweight='bold')

    fig.suptitle('Step 1 Dense Output — Orthogonal Projections', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    path = os.path.join(report_dir, "step1_projections.png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → {path}")


# =========================================================================
# Figure 5: Summary statistics card
# =========================================================================

def fig5_summary_card(pts, vis, nrm, meta, calib, report_dir):
    """Generate a clean summary stats figure."""
    print("[FIG 5] Summary statistics...")

    n_vis = vis.sum(axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.axis('off')

    # Compute stats
    stats = {
        "Total Points": f"{len(pts):,}",
        "Bounding Box X": f"[{pts[:,0].min():.1f}, {pts[:,0].max():.1f}] mm",
        "Bounding Box Y": f"[{pts[:,1].min():.1f}, {pts[:,1].max():.1f}] mm",
        "Bounding Box Z": f"[{pts[:,2].min():.1f}, {pts[:,2].max():.1f}] mm",
        "Mean Visible Cameras": f"{n_vis.mean():.2f} / {vis.shape[1]}",
        "Median Visible Cameras": f"{np.median(n_vis):.0f} / {vis.shape[1]}",
        "Points seen by ≥3 cams": f"{(n_vis >= 3).sum():,} ({100*(n_vis>=3).mean():.1f}%)",
        "Points seen by ≥6 cams": f"{(n_vis >= 6).sum():,} ({100*(n_vis>=6).mean():.1f}%)",
        "Points seen by 0 cams": f"{(n_vis == 0).sum():,} ({100*(n_vis==0).mean():.1f}%)",
        "Has Normals": "Yes" if nrm is not None else "No",
        "Point Cloud Span (X)": f"{pts[:,0].max() - pts[:,0].min():.1f} mm",
        "Point Cloud Span (Y)": f"{pts[:,1].max() - pts[:,1].min():.1f} mm",
        "Point Cloud Span (Z)": f"{pts[:,2].max() - pts[:,2].min():.1f} mm",
        "Num Cameras": str(vis.shape[1]),
    }

    y = 0.95
    ax.text(0.5, y + 0.02, "Step 1 Dense Output — Summary Statistics",
            transform=ax.transAxes, fontsize=16, fontweight='bold',
            ha='center', va='center', fontfamily='monospace')

    for i, (key, val) in enumerate(stats.items()):
        y = 0.88 - i * 0.06
        ax.text(0.1, y, key, transform=ax.transAxes, fontsize=11, fontfamily='monospace',
                fontweight='bold', va='center')
        ax.text(0.55, y, val, transform=ax.transAxes, fontsize=11, fontfamily='monospace',
                va='center', color='darkblue')

    # Add a note about the output
    ax.text(0.5, 0.03, "Output directory: calibration/dense/",
            transform=ax.transAxes, fontsize=9, ha='center', va='center',
            fontfamily='monospace', color='gray')

    path = os.path.join(report_dir, "step1_summary.png")
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → {path}")


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 60)
    print("  Step 1 Dense Output Visualization")
    print("=" * 60)

    pts, nrm, vis, meta = load_dense_data(DENSE_DIR)
    calib = load_cameras(CALIB_DIR)

    fig1_point_cloud_visibility(pts, vis, calib, REPORT_DIR)
    fig2_camera_poses(pts, calib, REPORT_DIR)
    fig3_per_camera_visibility(pts, vis, calib, REPORT_DIR)
    fig4_projections(pts, vis, nrm, REPORT_DIR)
    fig5_summary_card(pts, vis, nrm, meta, calib, REPORT_DIR)

    print(f"\n[DONE] All figures saved to {REPORT_DIR}/")
    print(f"  → step1_visibility.png       Point cloud colored by visibility")
    print(f"  → step1_camera_poses.png     12-camera geometry + point cloud")
    print(f"  → step1_per_camera_vis.png   Per-camera visibility heatmaps")
    print(f"  → step1_projections.png      Orthogonal projections + stats")
    print(f"  → step1_summary.png          Summary statistics card")


if __name__ == "__main__":
    main()
