#!/usr/bin/env python
"""
Test PINN-Stereo end-to-end on CylinderDIC data.
"""
import os, sys, time
import numpy as np
import cv2
import torch
from scipy.io import loadmat

# Ensure ndef_dic is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ndef_dic.pinn_stereo import PINNStereoConfig, PINNStereo

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(DATA_DIR, "calibration")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

def load_calib():
    """Load cameras.mat and dense ground truth points."""
    cdata = loadmat(os.path.join(CALIB_DIR, "cameras.mat"))
    n_cam = int(cdata["num_cameras"][0, 0])

    def _extract(arr, idx, shape):
        item = arr[idx]
        if arr.dtype == object:
            flat = np.array([float(item.flat[k].item()) for k in range(item.size)])
            return flat.reshape(shape)
        return np.array(item).reshape(shape)

    K_list = [_extract(cdata["K_list"], i, (3, 3)) for i in range(n_cam)]
    R_list = [_extract(cdata["cam_from_world_R"], i, (3, 3)) for i in range(n_cam)]

    t_raw = cdata["cam_from_world_t"]
    t_list = []
    for i in range(n_cam):
        if t_raw.dtype == object:
            item = t_raw[i]
            if hasattr(item, 'flat'):
                flat = [float(item.flat[k].item()) for k in range(item.size)]
            else:
                flat = np.array(item).ravel().tolist()
            t_list.append(np.array(flat).reshape(3, 1))
        else:
            t_list.append(np.array(t_raw[i]).reshape(3, 1))

    # Use dense ground truth points for better Stage 1 supervision
    gt_path = os.path.join(DATA_DIR, "ground_truth", "points_ref.npy")
    if os.path.exists(gt_path):
        print(f"  Loading dense GT points from {gt_path}...")
        gt_pts = np.load(gt_path)
        # Subsample to ~100K for manageable training
        if len(gt_pts) > 100000:
            idx = np.random.RandomState(42).choice(len(gt_pts), 100000, replace=False)
            sparse_pts = gt_pts[idx]
        else:
            sparse_pts = gt_pts
    else:
        # Fallback to sparse COLMAP points
        pdata = loadmat(os.path.join(CALIB_DIR, "points3D.mat"))
        sparse_pts = pdata["points3D"]
        if sparse_pts.dtype == object:
            flat = np.array([float(sparse_pts.flat[k].item())
                           for k in range(sparse_pts.size)])
            sparse_pts = flat.reshape(sparse_pts.shape[0], 3)

    return K_list, R_list, t_list, sparse_pts, n_cam

def load_images():
    """Load reference image per camera."""
    images = []
    cam_dirs = sorted([d for d in os.listdir(IMAGE_DIR)
                      if os.path.isdir(os.path.join(IMAGE_DIR, d))])
    for cd in cam_dirs:
        cam_path = os.path.join(IMAGE_DIR, cd)
        files = sorted([f for f in os.listdir(cam_path)
                       if f.lower().endswith((".bmp", ".png", ".jpg"))])
        if files:
            img = cv2.imread(os.path.join(cam_path, files[0]), cv2.IMREAD_GRAYSCALE)
            images.append(torch.from_numpy(img.astype(np.float32) / 255.0))
    return images, cam_dirs

def main():
    print("=" * 60)
    print("  PINN-Stereo Test — CylinderDIC")
    print("=" * 60)

    K_list, R_list, t_list, sparse_pts, n_cam = load_calib()
    images, cam_dirs = load_images()

    W, H = images[0].shape[1], images[0].shape[0]
    print(f"  {n_cam} cameras, {W}×{H}, {len(sparse_pts)} sparse pts")

    cfg = PINNStereoConfig(
        stage1_epochs=1000,             # more epochs with 100K supervision pts
        stage1_lr=1e-3,
        stage2_epochs_max=500,
        stage2_lr=1e-4,
        stage2_patience=30,
        stage1_batch_size=4096,
        stage2_batch_size=128,
        stage2_batches_per_epoch=200,
        patch_radius=5,
        roi_dilation=15,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    stereo = PINNStereo(
        config=cfg,
        K_list=K_list,
        R_list=R_list,
        t_list=t_list,
        images=images,
        sparse_points=sparse_pts,
        image_dims=(W, H),
    )

    # Stage 1
    s1 = stereo.train_stage1()

    # Stage 2
    s2 = stereo.train_stage2()

    # Fuse
    pts, nrm = stereo.fuse_point_cloud()
    print(f"\n  Fused: {len(pts)} points")

    # Save
    out_dir = os.path.join(DATA_DIR, "pinn_stereo_out")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "dense_points.npy"), pts)
    stereo.save_depth_maps(os.path.join(out_dir, "depth_maps"))

    print(f"\n  Output → {out_dir}/")
    print(f"\n  DONE")

if __name__ == "__main__":
    main()
