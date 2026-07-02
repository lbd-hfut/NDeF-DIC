"""
Visualize Stage A dense depth field initialization and compare with COLMAP sparse points.
Outputs to result/dense/stage_a/vis/
"""
import os, sys, numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from ndef_dic.common.mat_io import unwrap_mat_batch
from ndef_dic.dense.camera_model import (
    build_geometries, compute_mask_bounds, compute_world_bbox,
    PerCameraNorm, project, backproject,
)
from ndef_dic.dense.roi_builder import load_masks
from ndef_dic.dense.stage_a_init import StageATrainer

calib_dir = 'case/CylinderDIC/calibration'
dense_dir = 'case/CylinderDIC/result/dense'
stage_a_dir = os.path.join(dense_dir, 'stage_a')
vis_dir = os.path.join(stage_a_dir, 'vis')
os.makedirs(vis_dir, exist_ok=True)

# --- Load calibration ---
cameras = loadmat(os.path.join(calib_dir, 'cameras.mat'))
n_cam = int(cameras['num_cameras'][0, 0])
K_raw = cameras['K_list']
R_raw = cameras['cam_from_world_R']
t_raw = cameras['cam_from_world_t']

K_batch = unwrap_mat_batch(K_raw, (3, 3))
R_batch = unwrap_mat_batch(R_raw, (3, 3))
t_batch = unwrap_mat_batch(t_raw, (3, 1))

K_list = [K_batch[i] for i in range(n_cam)]
R_list = [R_batch[i] for i in range(n_cam)]
t_list = [t_batch[i].ravel() for i in range(n_cam)]

geometries = build_geometries(K_list, R_list, t_list, [np.zeros(5)] * n_cam, 1440, 1080)

# --- Sparse points ---
pts = loadmat(os.path.join(calib_dir, 'points3D.mat'))
pts3d = pts['points3D']
pts3d = unwrap_mat_batch(pts3d, (3,))

# --- Masks ---
masks = load_masks(dense_dir, n_cam)
masks_bool = [m.astype(bool) for m in masks]

# --- Load Stage A ---
bbox_centre, bbox_scale = compute_world_bbox(pts3d, margin=0.1)
per_cam_norms = [
    PerCameraNorm(
        u_min=b[0], u_max=b[1], v_min=b[2], v_max=b[3],
        bbox_centre=bbox_centre, bbox_scale=bbox_scale,
    )
    for b in [compute_mask_bounds(m) for m in masks_bool]
]
trainer = StageATrainer.load(
    stage_a_dir, geometries, per_cam_norms, bbox_centre, bbox_scale,
    masks=masks_bool, image_dims=(1440, 1080), device='cuda',
)
print(f"Loaded Stage A model from {stage_a_dir}")

# ============================================================
# 1. Depth map comparison (4 representative cameras)
# ============================================================
cam_show = [0, 3, 6, 9]
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
fig.suptitle('Stage A: Dense Depth vs Sparse COLMAP Projection', fontsize=14, fontweight='bold')

for row, cam_id in enumerate(cam_show):
    mask = masks_bool[cam_id]
    geo = geometries[cam_id]

    # Stage A dense depth
    dmap_dense = trainer.predict_dense_depth_map(cam_id, mask)
    dmap_dense_masked = np.where(mask, dmap_dense, np.nan)

    # Sparse depth
    uv_sp, depth_sp = project(pts3d, geo.K, geo.R, geo.t)
    valid_sp = (
        (depth_sp > 0) &
        (uv_sp[:, 0] >= 0) & (uv_sp[:, 0] < 1440) &
        (uv_sp[:, 1] >= 0) & (uv_sp[:, 1] < 1080)
    )
    u_sp = uv_sp[valid_sp, 0].astype(int)
    v_sp = uv_sp[valid_sp, 1].astype(int)
    d_sp = depth_sp[valid_sp]
    dmap_sparse = np.full((1080, 1440), np.nan, dtype=np.float32)
    dmap_sparse[v_sp, u_sp] = d_sp

    vmin = min(np.nanmin(dmap_dense_masked), np.nanmin(dmap_sparse))
    vmax = max(np.nanmax(dmap_dense_masked), np.nanmax(dmap_sparse))

    # (a) Stage A dense
    ax = axes[row, 0]
    im = ax.imshow(dmap_dense_masked, cmap='turbo', vmin=vmin, vmax=vmax)
    ax.set_title(f'Cam {cam_id} - Stage A Dense', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, label='mm')

    # (b) Sparse
    ax = axes[row, 1]
    im = ax.imshow(dmap_sparse, cmap='turbo', vmin=vmin, vmax=vmax)
    ax.set_title(f'Cam {cam_id} - COLMAP Sparse ({valid_sp.sum()} pts)', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, label='mm')

    # (c) Error
    ax = axes[row, 2]
    d_dense_at_sparse = dmap_dense[v_sp, u_sp]
    error = np.abs(d_dense_at_sparse - d_sp)
    scatter = ax.scatter(u_sp, v_sp, c=error, cmap='hot', s=3, alpha=0.8, vmin=0, vmax=50)
    ax.invert_yaxis()
    ax.set_xlim(0, 1440)
    ax.set_ylim(1080, 0)
    ax.set_title(f'Cam {cam_id} - |error| (mean={error.mean():.1f}mm)', fontsize=10)
    ax.axis('off')
    plt.colorbar(scatter, ax=ax, fraction=0.046, label='|error| mm')

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'depth_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved depth_comparison.png')

# ============================================================
# 2. 3D backprojection comparison
# ============================================================
fig = plt.figure(figsize=(18, 8))

# (a) Stage A dense (subsampled)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_title(f'Stage A - Dense Backprojection ({n_cam} cameras)', fontsize=12)
colors_dense = plt.cm.tab20(np.linspace(0, 1, n_cam))

for cam_id in range(n_cam):
    mask = masks_bool[cam_id]
    rows, cols = np.where(mask)
    step = max(1, len(rows) // 3000)
    r_sub = rows[::step]
    c_sub = cols[::step]
    uv_sub = np.stack([c_sub.astype(np.float32), r_sub.astype(np.float32)], axis=-1)

    dmap = trainer.predict_dense_depth_map(cam_id, mask)
    d_sub = dmap[r_sub, c_sub]
    X_sub = backproject(uv_sub, d_sub, geometries[cam_id].K,
                        geometries[cam_id].R, geometries[cam_id].t)
    ax1.scatter(X_sub[:, 0], X_sub[:, 1], X_sub[:, 2],
                c=[colors_dense[cam_id]], s=1, alpha=0.6, label=f'Cam {cam_id}')
    C = geometries[cam_id].centre
    ax1.scatter(*C, c='black', s=30, marker='^')

ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_zlabel('Z (mm)')
ax1.legend(loc='upper right', fontsize=6, ncol=2)

# (b) Sparse COLMAP
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_title(f'COLMAP Sparse - {len(pts3d)} points', fontsize=12)
z_vals = pts3d[:, 2]
ax2.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c=z_vals, cmap='turbo',
            s=15, alpha=0.9)
for cam_id in range(n_cam):
    C = geometries[cam_id].centre
    ax2.scatter(*C, c='black', s=30, marker='^')
    ax2.text(C[0], C[1], C[2], f'{cam_id}', fontsize=7)

ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Y (mm)')
ax2.set_zlabel('Z (mm)')

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '3d_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved 3d_comparison.png')

# ============================================================
# 3. Per-camera error histogram
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle('Per-Camera Depth Error Distribution (Stage A vs COLMAP)', fontsize=14, fontweight='bold')

all_errors = []
for cam_id in range(n_cam):
    ax = axes[cam_id // 4, cam_id % 4]
    mask = masks_bool[cam_id]
    geo = geometries[cam_id]

    uv_sp, depth_sp = project(pts3d, geo.K, geo.R, geo.t)
    valid_sp = (
        (depth_sp > 0) &
        (uv_sp[:, 0] >= 0) & (uv_sp[:, 0] < 1440) &
        (uv_sp[:, 1] >= 0) & (uv_sp[:, 1] < 1080) &
        mask[uv_sp[:, 1].astype(int), uv_sp[:, 0].astype(int)]
    )
    u_sp = uv_sp[valid_sp, 0].astype(int)
    v_sp = uv_sp[valid_sp, 1].astype(int)
    d_sp = depth_sp[valid_sp]

    dmap_dense = trainer.predict_dense_depth_map(cam_id, mask)
    d_dense = dmap_dense[v_sp, u_sp]
    errors = d_dense - d_sp
    all_errors.append(errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    ax.hist(errors, bins=40, color='steelblue', edgecolor='white', alpha=0.8, density=True)
    ax.axvline(0, color='red', linestyle='--', linewidth=1)
    ax.axvline(errors.mean(), color='orange', linestyle='-', linewidth=1.5,
               label=f'mean={errors.mean():.1f}mm')
    ax.set_title(f'Cam {cam_id} (RMSE={rmse:.1f}mm)', fontsize=9)
    ax.set_xlabel('depth error (mm)')
    ax.set_ylabel('density')
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'error_histograms.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved error_histograms.png')

# Summary
all_e = np.concatenate(all_errors)
print(f'\nGlobal depth error stats ({len(all_e)} sparse-sample pairs):')
print(f'  Mean bias: {all_e.mean():+.1f} mm')
print(f'  Std:       {all_e.std():.1f} mm')
print(f'  RMSE:      {np.sqrt(np.mean(all_e**2)):.1f} mm')
print(f'  MAE:       {np.mean(np.abs(all_e)):.1f} mm')
print(f'  P95:       {np.percentile(np.abs(all_e), 95):.1f} mm')
print(f'\nAll visualizations saved to {vis_dir}/')
