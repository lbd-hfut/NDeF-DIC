"""
Step 1 Pipeline: Geometric Reconstruction.

Orchestrates the full Step 1 workflow:
  sparse SfM (or skip) → dense MVS (optional) → post-processing.

Modes:
  sparse_mode="full"      Run COLMAP sparse SfM from images
  sparse_mode="skip_sfm"  Load pre-computed cameras.mat (from Step 0 or prior run)

  dense_path="patchmatch" Run COLMAP PatchMatch Stereo + Fusion (requires CUDA)
  dense_path="pinn_stereo" [stub] PINN-Stereo neural depth estimation
  dense_path="none"       Skip dense MVS, use sparse points only
"""

import os
import json
import numpy as np
from typing import Optional, Dict, List, Literal, Tuple
from dataclasses import dataclass, field
from scipy.io import loadmat, savemat


# =========================================================================
# Output dataclass
# =========================================================================

@dataclass
class Step1Output:
    """Structured output of Step 1, consumed by Step 2 (SurfaceProvider)."""

    # ---- Calibration ----
    K_list: List[np.ndarray]
    R_list: List[np.ndarray]
    t_list: List[np.ndarray]
    dist_list: List[np.ndarray]
    camera_models: List[str]
    num_cameras: int

    # ---- Points ----
    points3D: np.ndarray          # (N, 3) dense (or sparse if dense skipped)
    normals3D: np.ndarray         # (N, 3) unit normals
    vis_mask: np.ndarray          # (N, N_cam) bool visibility matrix

    # ---- Metadata ----
    image_width: int
    image_height: int
    dense_status: str             # "ok" | "skipped_cuda" | "none"
    num_points_sparse: int = 0
    num_points_dense: int = 0

    # ---- Projection matrices (derived) ----
    P_list: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        if not self.P_list:
            self.P_list = [
                K @ np.hstack((R, t.reshape(3, 1)))
                for K, R, t in zip(self.K_list, self.R_list, self.t_list)
            ]


# =========================================================================
# Loading helpers
# =========================================================================

def load_calibration(calib_dir: str) -> Dict:
    """
    Load cameras.mat from a calibration directory.

    Handles both:
      - Old format: dtype=object with nested arrays
      - New format: stacked float64 arrays

    Returns dict with keys:
      K_list, R_list, t_list, dist_list, camera_models, P_list, num_cameras
    """
    cameras_path = os.path.join(calib_dir, "cameras.mat")
    if not os.path.exists(cameras_path):
        raise FileNotFoundError(f"Calibration file not found: {cameras_path}")

    calib = loadmat(cameras_path)
    n_cam = int(calib["num_cameras"][0, 0])

    # ---- Extract K ----
    K_raw = calib["K_list"]
    if K_raw.dtype == object:
        # Old format
        K_list = [_extract_matrix(K_raw[i], (3, 3)) for i in range(n_cam)]
    else:
        # New stacked format: (N, 3, 3)
        K_list = [K_raw[i] for i in range(n_cam)]

    # ---- Extract R ----
    R_raw = calib["cam_from_world_R"]
    if R_raw.dtype == object:
        R_list = [_extract_matrix(R_raw[i], (3, 3)) for i in range(n_cam)]
    else:
        R_list = [R_raw[i] for i in range(n_cam)]

    # ---- Extract t ----
    t_raw = calib["cam_from_world_t"]
    if t_raw.dtype == object:
        t_list = [_extract_matrix(t_raw[i], (3, 1)) for i in range(n_cam)]
    else:
        # New format: (N, 1, 3) or (N, 3, 1) or (N, 3)
        shape = t_raw.shape
        if len(shape) == 3:
            if shape[2] == 1:
                t_list = [t_raw[i].reshape(3, 1) for i in range(n_cam)]
            elif shape[1] == 1:
                t_list = [t_raw[i].reshape(3, 1) for i in range(n_cam)]
            else:
                t_list = [t_raw[i].reshape(3, 1) for i in range(n_cam)]
        else:
            t_list = [t_raw[i].reshape(3, 1) for i in range(n_cam)]

    # ---- Extract dist ----
    dist_raw = calib.get("dist_list")
    if dist_raw is not None:
        if dist_raw.dtype == object:
            dist_list = [_extract_matrix(dist_raw[i], (5,)) for i in range(n_cam)]
        else:
            dist_list = [dist_raw[i] for i in range(n_cam)]
    else:
        dist_list = [np.zeros(5) for _ in range(n_cam)]

    # ---- Extract camera models ----
    models_raw = calib.get("camera_models")
    if models_raw is not None:
        # Flatten regardless of shape: (1, N), (N,), (N, 1)
        flat_models = models_raw.flatten()
        if models_raw.dtype == object:
            camera_models = [
                str(flat_models[i].flat[0]) if hasattr(flat_models[i], 'flat')
                else str(flat_models[i])
                for i in range(min(n_cam, len(flat_models)))
            ]
        else:
            camera_models = [str(m) for m in flat_models[:n_cam]]
    else:
        camera_models = ["PINHOLE"] * n_cam

    # ---- Derive P ----
    P_list = [
        K_list[i] @ np.hstack((R_list[i], t_list[i]))
        for i in range(n_cam)
    ]

    return {
        "K_list": K_list,
        "R_list": R_list,
        "t_list": t_list,
        "dist_list": dist_list,
        "camera_models": camera_models,
        "P_list": P_list,
        "num_cameras": n_cam,
    }


def _extract_matrix(obj: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Extract a clean float array from a nested object array."""
    flat = np.array([float(obj.flat[i].item()) for i in range(obj.size)])
    return flat.reshape(shape).astype(np.float64)


# =========================================================================
# Image discovery
# =========================================================================

def _discover_images(image_dir: str, ref_name: str = "001") -> Tuple[List[str], List[str]]:
    """
    Discover camera directories and reference image paths.

    Expects: image_dir/cam_0/<ref_name>.bmp, image_dir/cam_1/<ref_name>.bmp, ...

    Returns:
        cam_names:   List of camera folder names
        ref_paths:   List of absolute paths to reference images
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    cam_names = sorted([
        d for d in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, d)) and d.startswith("cam_")
    ])

    if not cam_names:
        raise FileNotFoundError(f"No cam_* directories under {image_dir}")

    ref_paths = []
    for cam in cam_names:
        cam_dir = os.path.join(image_dir, cam)
        # Try ref_name.bmp, ref_name.png, etc.
        ref_file = None
        for ext in [".bmp", ".png", ".jpg", ".tif"]:
            candidate = os.path.join(cam_dir, f"{ref_name}{ext}")
            if os.path.exists(candidate):
                ref_file = candidate
                break
        if ref_file is None:
            # Fallback: first image file in directory
            files = sorted([f for f in os.listdir(cam_dir)
                           if f.lower().endswith((".bmp", ".png", ".jpg", ".tif"))])
            if files:
                ref_file = os.path.join(cam_dir, files[0])
        if ref_file is None:
            raise FileNotFoundError(f"No reference image found in {cam_dir}")
        ref_paths.append(ref_file)

    return cam_names, ref_paths


# =========================================================================
# Merge sparse + ground truth
# =========================================================================

def _load_sparse_points(calib_dir: str) -> Optional[np.ndarray]:
    """Load sparse points3D from points3D.mat if available."""
    pts_path = os.path.join(calib_dir, "points3D.mat")
    if not os.path.exists(pts_path):
        return None
    data = loadmat(pts_path)
    if "points3D" in data:
        pts = data["points3D"]
        if pts.dtype == object:
            return _extract_matrix(pts, (pts.shape[0], 3))
        return pts.astype(np.float64)
    return None


# =========================================================================
# Main pipeline
# =========================================================================

def run_step1(
    data_dir: str,
    image_dir: str = "images",
    sparse_mode: Literal["full", "skip_sfm"] = "full",
    dense_path: Literal["patchmatch", "pinn_stereo", "none"] = "patchmatch",
    calib_dir: Optional[str] = None,
    dense_workspace: Optional[str] = None,
    ref_name: str = "001",
    image_width: int = 1440,
    image_height: int = 1080,
    post_config: Optional[Dict] = None,
    clean: bool = False,
) -> Step1Output:
    """
    Run Step 1: Geometric Reconstruction.

    Args:
        data_dir:        Root data directory (e.g., case/CylinderDIC/).
        image_dir:       Subdirectory containing camera folders (default: "images").
        sparse_mode:     "full" = run COLMAP SfM; "skip_sfm" = load existing.
        dense_path:      "patchmatch" = COLMAP PatchMatch; "pinn_stereo" = [stub]; "none" = skip.
        calib_dir:       Calibration output directory (default: data_dir/calibration).
                         For skip_sfm, this is where cameras.mat is loaded from.
        dense_workspace: Dense MVS workspace directory (default: data_dir/dense_workspace).
        ref_name:        Reference image base name (default: "001").
        image_width:     Image width (used for visibility computation).
        image_height:    Image height (used for visibility computation).
        post_config:     Optional dict for post-processing parameters.
        clean:           Remove existing results before running.

    Returns:
        Step1Output with calibration, points, normals, and visibility.
    """
    from . import colmap_calib
    from . import dense_mvs
    from . import postprocess

    calib_dir = calib_dir or os.path.join(data_dir, "calibration")
    dense_workspace = dense_workspace or os.path.join(data_dir, "dense_workspace")
    os.makedirs(calib_dir, exist_ok=True)

    # ---- Discover images ----
    img_root = os.path.join(data_dir, image_dir)
    cam_names, ref_paths = _discover_images(img_root, ref_name)
    num_cameras = len(cam_names)
    print(f"[Step 1] Discovered {num_cameras} cameras in {img_root}")

    # =====================================================================
    # Sparse reconstruction
    # =====================================================================
    if sparse_mode == "full":
        print(f"[Step 1] Sparse mode: full (COLMAP SfM)")
        # Copy images to a flat directory for COLMAP
        colmap_image_dir = os.path.join(calib_dir, "colmap_images")
        if clean and os.path.exists(colmap_image_dir):
            import shutil
            shutil.rmtree(colmap_image_dir)
        os.makedirs(colmap_image_dir, exist_ok=True)

        import cv2
        for cam_name, ref_path in zip(cam_names, ref_paths):
            new_name = f"{cam_name}_{os.path.basename(ref_path)}"
            img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read: {ref_path}")
            cv2.imwrite(os.path.join(colmap_image_dir, new_name), img)

        colmap_calib.run_colmap_calibration(
            data_dir=data_dir,
            image_dir=image_dir,
            ref_mode="named",
            ref_name=ref_name,
            output_dir=calib_dir,
            clean=clean,
        )

    elif sparse_mode == "skip_sfm":
        print(f"[Step 1] Sparse mode: skip_sfm (loading from {calib_dir})")
        if not colmap_calib.calibration_exists(data_dir, calib_dir):
            raise FileNotFoundError(
                f"No calibration found at {calib_dir}/cameras.mat. "
                f"Run with sparse_mode='full' first, or provide calibration from Step 0."
            )

    else:
        raise ValueError(f"Unknown sparse_mode: {sparse_mode}")

    # ---- Load calibration (common to both modes) ----
    calib_data = load_calibration(calib_dir)
    K_list = calib_data["K_list"]
    R_list = calib_data["R_list"]
    t_list = calib_data["t_list"]
    dist_list = calib_data["dist_list"]
    camera_models = calib_data["camera_models"]
    P_list = calib_data["P_list"]

    print(f"[Step 1] Loaded calibration: {num_cameras} cameras")

    # =====================================================================
    # Dense reconstruction
    # =====================================================================
    dense_status = "none"
    dense_points = None
    dense_normals = None
    depth_map_dir = None

    if dense_path == "patchmatch":
        print(f"[Step 1] Dense path: patchmatch")

        # We need a flat image directory and the SfM path
        sfm_path = os.path.join(calib_dir, "colmap_sfm")
        if not os.path.exists(sfm_path):
            raise FileNotFoundError(
                f"SfM output not found at {sfm_path}. "
                f"Run sparse_mode='full' first."
            )

        colmap_image_dir = os.path.join(calib_dir, "colmap_images")
        if not os.path.exists(colmap_image_dir):
            # Re-create from reference paths
            import cv2
            os.makedirs(colmap_image_dir, exist_ok=True)
            for cam_name, ref_path in zip(cam_names, ref_paths):
                new_name = f"{cam_name}_{os.path.basename(ref_path)}"
                img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(os.path.join(colmap_image_dir, new_name), img)

        dense_result = dense_mvs.run_dense_mvs(
            image_dir=colmap_image_dir,
            sfm_path=sfm_path,
            workspace_dir=dense_workspace,
            clean=clean,
        )
        dense_status = dense_result["status"]
        dense_points = dense_result["dense_points"]
        dense_normals = dense_result["dense_normals"]
        depth_map_dir = dense_result.get("depth_map_dir")

    elif dense_path == "pinn_stereo":
        print(f"[Step 1] Dense path: pinn_stereo")

        from . import pinn_stereo
        dense_result = pinn_stereo.run_pinn_stereo(
            image_dir=img_root,
            sfm_path=os.path.join(calib_dir, "colmap_sfm"),
            workspace_dir=dense_workspace,
            calib_dir=calib_dir,
            K_list=K_list,
            R_list=R_list,
            t_list=t_list,
            image_width=image_width,
            image_height=image_height,
            ref_paths=ref_paths,
            clean=clean,
        )
        dense_status = dense_result["status"]
        dense_points = dense_result["dense_points"]
        dense_normals = dense_result["dense_normals"]
        depth_map_dir = dense_result.get("depth_map_dir")

    elif dense_path == "none":
        print(f"[Step 1] Dense path: none (using sparse points only)")

    # =====================================================================
    # Assemble point cloud for post-processing
    # =====================================================================
    # Priority: dense MVS > ground truth (Step 0) > sparse SfM
    sparse_pts = _load_sparse_points(calib_dir)
    num_sparse = len(sparse_pts) if sparse_pts is not None else 0
    gt_path = os.path.join(data_dir, "ground_truth", "points_ref.npy")
    gt_pts = np.load(gt_path) if os.path.exists(gt_path) else None

    from_gt = False
    if dense_points is not None and len(dense_points) > 0:
        post_points = dense_points
        post_normals = dense_normals
        print(f"[Step 1] Using dense points: {len(post_points)}")
    elif gt_pts is not None and len(gt_pts) > 0:
        post_points = gt_pts.astype(np.float64)
        post_normals = None
        from_gt = True
        print(f"[Step 1] Using ground truth points (Step 0): {len(post_points)}")
    elif sparse_pts is not None and len(sparse_pts) > 0:
        post_points = sparse_pts.astype(np.float64)
        post_normals = None
        print(f"[Step 1] Using sparse points (fallback): {len(post_points)}")
    else:
        raise RuntimeError("No point cloud available. Run SfM or provide ground truth.")

    # =====================================================================
    # Post-processing
    # =====================================================================
    # Merge auto-detected flags with user config
    post_cfg = dict(post_config) if post_config else {}
    if from_gt:
        post_cfg.setdefault("skip_sor", True)       # Ground truth is clean
        post_cfg.setdefault("subsample_target", 200_000)  # Manageable size

    dense_out_dir = os.path.join(calib_dir, "dense")
    os.makedirs(dense_out_dir, exist_ok=True)

    post_result = postprocess.run_postprocess(
        points=post_points.astype(np.float64),
        normals=post_normals,
        K_list=K_list,
        R_list=R_list,
        t_list=t_list,
        image_width=image_width,
        image_height=image_height,
        output_dir=dense_out_dir,
        config=post_cfg,
    )

    # =====================================================================
    # Assemble output
    # =====================================================================
    output = Step1Output(
        K_list=K_list,
        R_list=R_list,
        t_list=t_list,
        dist_list=dist_list,
        camera_models=camera_models,
        num_cameras=num_cameras,
        points3D=post_result["points"],
        normals3D=post_result["normals"],
        vis_mask=post_result["vis_mask"],
        image_width=image_width,
        image_height=image_height,
        dense_status=dense_status,
        num_points_sparse=num_sparse,
        num_points_dense=len(dense_points) if dense_points is not None else 0,
        P_list=P_list,
    )

    print(f"\n[Step 1] Complete.")
    print(f"  Cameras:  {output.num_cameras}")
    print(f"  Points:   {len(output.points3D)} (dense: {output.num_points_dense}, "
          f"sparse: {output.num_points_sparse})")
    print(f"  Normals:  {output.normals3D.shape}")
    print(f"  Vis mask: {output.vis_mask.shape}, "
          f"mean visible: {output.vis_mask.sum(axis=1).mean():.1f}")
    print(f"  Dense:    {output.dense_status}")

    return output
