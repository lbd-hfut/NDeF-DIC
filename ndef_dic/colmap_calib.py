"""
COLMAP self-calibration for NDeF-DIC.

Collects the reference image from each camera, runs COLMAP's SfM pipeline
(SIFT extraction → exhaustive matching → incremental mapping), and saves:

  - calibration/cameras.mat   : K, dist, R, t per camera
  - calibration/points3D.mat  : sparse 3D point cloud

Called automatically as Stage 0 of the NDeF-DIC pipeline when
calibration files are missing or --recalibrate is specified.
"""

import os
import shutil
import numpy as np
from typing import Dict, List, Tuple, Optional


# =========================================================================
# Image collection
# =========================================================================

def _is_image_file(fname: str) -> bool:
    exts = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
    return os.path.splitext(fname)[1].lower() in exts


def _discover_camera_dirs(image_dir: str) -> List[str]:
    """Discover and sort camera sub-directories under image_dir."""
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    dirs = sorted([
        d for d in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, d))
    ])
    if not dirs:
        raise FileNotFoundError(f"No camera folders found under {image_dir}")
    return dirs


def _collect_reference_images(
    data_dir: str,
    image_dir: str = "images",
    ref_mode: str = "first",
    ref_name: str = "ref",
) -> Dict[str, str]:
    """
    Collect one reference image per camera.

    Returns:
        Dict[camera_folder_name] -> absolute path to reference image.
    """
    img_root = os.path.join(data_dir, image_dir)
    cam_dirs = _discover_camera_dirs(img_root)
    ref_images: Dict[str, str] = {}

    for cam_name in cam_dirs:
        cam_path = os.path.join(img_root, cam_name)
        files = sorted([f for f in os.listdir(cam_path) if _is_image_file(f)])
        if not files:
            raise FileNotFoundError(f"No image files found in {cam_path}")

        if ref_mode == "named":
            found = [f for f in files if os.path.splitext(f)[0] == ref_name]
            if not found:
                raise FileNotFoundError(
                    f"No '{ref_name}.*' found in {cam_path}. Files: {files}"
                )
            ref_path = os.path.join(cam_path, found[0])
        else:
            ref_path = os.path.join(cam_path, files[0])

        ref_images[cam_name] = ref_path

    return ref_images


# =========================================================================
# Camera parameter extraction
# =========================================================================

def _extract_camera_params(camera) -> Tuple[np.ndarray, np.ndarray]:
    """Extract K (3,3) and dist (5,) from a pycolmap Camera object."""
    params = camera.params
    model_raw = str(camera.model)
    model = model_raw.split(".")[-1].upper() if "." in model_raw else model_raw.upper()

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[0], params[1], params[2]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params[0], params[1], params[2], params[3]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = params[0], params[1], params[2], params[3], params[4]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2, 0.0], dtype=np.float64)
    elif model == "FULL_OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = params[:12]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    else:
        K = camera.calibration_matrix().astype(np.float64)
        dist = np.zeros(5, dtype=np.float64)
        print(f"  [WARNING] Unknown camera model '{model_raw}', using fallback K")

    return K, dist


# =========================================================================
# Main calibration function
# =========================================================================

def run_colmap_calibration(
    data_dir: str,
    image_dir: str = "images",
    ref_mode: str = "first",
    ref_name: str = "ref",
    output_dir: Optional[str] = None,
    max_features: int = 8192,
    first_octave: int = 0,
    cross_check: bool = True,
    min_num_matches: int = 15,
    min_model_size: int = 3,
    ba_global_max_refinements: int = 5,
    clean: bool = False,
) -> dict:
    """
    Run COLMAP self-calibration on multi-camera reference images.

    Automatically called before training if calibration files are missing.
    Can also be triggered manually with --recalibrate.

    Args:
        data_dir: Root data directory.
        image_dir: Subdirectory containing camera folders.
        ref_mode: "first" (first file is ref) or "named" (ref_name.ext).
        ref_name: Reference image base name (named mode only).
        output_dir: Output directory for .mat files (default: data_dir/calibration).
        max_features: Max SIFT features per image.
        first_octave: SIFT first octave.
        cross_check: Enable cross-check matching.
        min_num_matches: Minimum matches per image pair.
        min_model_size: Minimum registered images for a valid model.
        ba_global_max_refinements: Max BA refinement passes.
        clean: Remove previous results before running.

    Returns:
        dict with keys: num_cameras, K_list, dist_list, cam_from_world_R,
                        cam_from_world_t, P_list, camera_models, num_points3D.

    Raises:
        RuntimeError: If COLMAP SfM fails.
        FileNotFoundError: If no valid images found.
    """
    import pycolmap
    import cv2
    from scipy.io import savemat

    calib_dir = output_dir or os.path.join(data_dir, "calibration")
    os.makedirs(calib_dir, exist_ok=True)

    # ---- 1. Collect reference images ----
    print(f"[Stage 0] Collecting reference images from {os.path.join(data_dir, image_dir)}")
    ref_images = _collect_reference_images(data_dir, image_dir, ref_mode, ref_name)
    cam_names = sorted(ref_images.keys())
    num_cameras = len(cam_names)
    print(f"  Found {num_cameras} cameras: {cam_names}")

    # ---- 2. Copy to temp directory ----
    colmap_image_dir = os.path.join(calib_dir, "colmap_calib_images")
    if clean and os.path.exists(colmap_image_dir):
        shutil.rmtree(colmap_image_dir)
    os.makedirs(colmap_image_dir, exist_ok=True)

    for cam_name, ref_path in ref_images.items():
        basename = os.path.basename(ref_path)
        new_name = f"{cam_name}_{basename}"
        new_path = os.path.join(colmap_image_dir, new_name)
        img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {ref_path}")
        cv2.imwrite(new_path, img)

    # ---- 3. Database and SfM paths ----
    database_path = os.path.join(calib_dir, "colmap.db")
    sfm_path = os.path.join(calib_dir, "colmap_sfm")
    if clean:
        if os.path.exists(database_path):
            os.remove(database_path)
        if os.path.exists(sfm_path):
            shutil.rmtree(sfm_path)
    os.makedirs(sfm_path, exist_ok=True)

    # ---- 4. Feature extraction ----
    print(f"[Stage 0] Extracting SIFT features (max_features={max_features})...")
    pycolmap.set_random_seed(0)
    pycolmap.extract_features(
        database_path, colmap_image_dir,
        extraction_options={"sift": {"max_num_features": max_features,
                                      "first_octave": first_octave}}
    )

    # ---- 5. Exhaustive matching ----
    print(f"[Stage 0] Exhaustive matching (cross_check={cross_check})...")
    pycolmap.match_exhaustive(
        database_path,
        matching_options={"sift": {"cross_check": cross_check}}
    )

    # ---- 6. Incremental SfM ----
    print(f"[Stage 0] Running incremental SfM...")
    sfm_opts = {
        "ba_global_max_refinements": ba_global_max_refinements,
        "min_num_matches": min_num_matches,
        "multiple_models": False,
        "min_model_size": min(min_model_size, num_cameras),
        "min_focal_length_ratio": 0.1,
        "max_focal_length_ratio": 10.0,
    }
    reconstructions = pycolmap.incremental_mapping(
        database_path, colmap_image_dir, sfm_path, options=sfm_opts
    )

    if not reconstructions:
        raise RuntimeError(
            "COLMAP SfM failed. Possible reasons:\n"
            "  - Images have insufficient texture for SIFT features.\n"
            "  - Not enough visual overlap between camera views.\n"
            "  - Try increasing max_features or check image quality."
        )

    rec = reconstructions[0]
    print(f"  Registered: {rec.num_reg_images()}/{num_cameras} images, "
          f"{rec.num_points3D()} 3D points")

    # ---- 7. Extract camera parameters ----
    cam_image_ids = {name: [] for name in cam_names}
    for image_id, image in rec.images.items():
        for cam_name in cam_names:
            if image.name.startswith(f"{cam_name}_"):
                cam_image_ids[cam_name].append(image_id)
                break

    K_list, dist_list = [], []
    cam_from_world_R, cam_from_world_t, P_list = [], [], []
    camera_models = []

    for cam_name in cam_names:
        ids = cam_image_ids[cam_name]
        if not ids:
            raise RuntimeError(
                f"Camera '{cam_name}': no registered images in COLMAP output.\n"
                f"  The reference image may lack sufficient texture for SIFT.\n"
                f"  Check: {ref_images[cam_name]}"
            )
        cam_id = rec.images[ids[0]].camera_id
        camera = rec.cameras[cam_id]
        K, dist = _extract_camera_params(camera)
        K_list.append(K)
        dist_list.append(dist)
        camera_models.append(str(camera.model).split(".")[-1])

        cfw = rec.images[ids[0]].cam_from_world
        if callable(cfw):
            cfw = cfw()
        cam_from_world_R.append(cfw.rotation.matrix().astype(np.float64))
        cam_from_world_t.append(cfw.translation.astype(np.float64).reshape(3, 1))
        P_list.append(K @ np.hstack((cam_from_world_R[-1], cam_from_world_t[-1])))

    # ---- 8. Extract sparse 3D points ----
    point3D_ids = rec.point3D_ids
    if callable(point3D_ids):
        point3D_ids = point3D_ids()
    points3D_list = []
    for pid in point3D_ids:
        pt = rec.point3D(pid)
        if pt.error < 4.0:
            points3D_list.append(pt.xyz)
    points3D = np.array(points3D_list, dtype=np.float64) if points3D_list else np.zeros((0, 3))

    # ---- 9. Save results (proper float arrays, not dtype=object) ----
    cameras_result = {
        "num_cameras": num_cameras,
        "K_list": np.stack(K_list, axis=0).astype(np.float64),
        "dist_list": np.stack(dist_list, axis=0).astype(np.float64),
        "cam_from_world_R": np.stack(cam_from_world_R, axis=0).astype(np.float64),
        "cam_from_world_t": np.stack(
            [t.reshape(1, 3) for t in cam_from_world_t], axis=0
        ).astype(np.float64),
        "P_list": np.stack(P_list, axis=0).astype(np.float64),
        "camera_models": np.array(camera_models, dtype=object),
        "cam_names": np.array(cam_names, dtype=object),
        "num_registered_images": rec.num_reg_images(),
    }
    savemat(os.path.join(calib_dir, "cameras.mat"), cameras_result)

    points_result = {
        "points3D": points3D.astype(np.float64),
        "num_points": len(points3D),
    }
    savemat(os.path.join(calib_dir, "points3D.mat"), points_result)

    # ---- 10. Summary ----
    print(f"\n[Step 1 SfM] COLMAP sparse reconstruction complete")
    print(f"  cameras.mat  → {calib_dir}/cameras.mat")
    print(f"  points3D.mat → {calib_dir}/points3D.mat")
    for i, name in enumerate(cam_names):
        print(f"  Camera {name}: {camera_models[i]}, "
              f"f=({K_list[i][0,0]:.1f}, {K_list[i][1,1]:.1f}), "
              f"cx={K_list[i][0,2]:.1f}, cy={K_list[i][1,2]:.1f}")
    print(f"  Sparse points: {len(points3D)}")
    print()

    return cameras_result


def calibration_exists(data_dir: str, output_dir: Optional[str] = None) -> bool:
    """Check if COLMAP calibration files already exist."""
    calib_dir = output_dir or os.path.join(data_dir, "calibration")
    cameras_path = os.path.join(calib_dir, "cameras.mat")
    points_path = os.path.join(calib_dir, "points3D.mat")
    return os.path.exists(cameras_path) and os.path.exists(points_path)
