"""
DataLoader for NDeF-DIC result viewer.

Loads all available data from a data_dir into a structured LoadedData container.
Handles partial/missing data gracefully — every field is Optional.
"""

import os
import json
import glob
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class LoadedData:
    """All data that may exist in a data_dir. Every field is Optional."""

    data_dir: str = ""

    # ---- Calibration (cameras.mat) ----
    K_list: Optional[List[np.ndarray]] = None
    R_list: Optional[List[np.ndarray]] = None
    t_list: Optional[List[np.ndarray]] = None
    dist_list: Optional[List[np.ndarray]] = None
    camera_models: Optional[List[str]] = None
    P_list: Optional[List[np.ndarray]] = None
    num_cameras: int = 0

    # ---- Camera centers (derived) ----
    camera_centers: Optional[np.ndarray] = None  # (N, 3) world coords

    # ---- Sparse SfM points ----
    sparse_points: Optional[np.ndarray] = None   # (M, 3)

    # ---- Dense MVS ----
    dense_points: Optional[np.ndarray] = None    # (K, 3)
    dense_normals: Optional[np.ndarray] = None   # (K, 3)
    dense_vis_mask: Optional[np.ndarray] = None  # (K, N_cam)
    dense_meta: Optional[Dict] = None

    # ---- Displacement results ----
    ref_points: Optional[np.ndarray] = None      # (P, 3)
    disp_fields: Optional[Dict[int, np.ndarray]] = None   # {step: (P, 3)}
    def_points: Optional[Dict[int, np.ndarray]] = None    # {step: (P, 3)}
    n_steps: int = 0
    results_meta: Optional[Dict] = None

    # ---- Ground truth (optional) ----
    gt_ref_points: Optional[np.ndarray] = None      # (N_gt, 3)
    gt_disp_fields: Optional[Dict[int, np.ndarray]] = None  # {step: (N_gt, 3)}
    gt_n_steps: int = 0

    # ---- Images ----
    cam_names: Optional[List[str]] = None
    ref_image_paths: Optional[List[str]] = None
    def_image_paths: Optional[Dict[int, List[str]]] = None  # {step: [cam_paths]}
    images_per_step: int = 0  # number of deformed image sets found


class DataLoader:
    """Load all available NDeF-DIC results from a data directory.

    Usage:
        loader = DataLoader()
        data = loader.load("case/CylinderDIC")
        if loader.errors:
            print("Warnings:", loader.errors)
    """

    def __init__(self):
        self._errors: List[str] = []

    @property
    def errors(self) -> List[str]:
        return self._errors

    def load(self, data_dir: str) -> LoadedData:
        """Main entry point. Never raises — returns partial LoadedData on error."""
        self._errors = []
        data = LoadedData(data_dir=os.path.abspath(data_dir))

        # 1. Calibration
        self._load_calibration(data)

        # 2. Sparse points
        self._load_sparse(data)

        # 3. Dense data
        self._load_dense(data)

        # 4. Displacement results
        self._load_displacement(data)

        # 5. Ground truth (optional)
        self._load_ground_truth(data)

        # 6. Images
        self._load_images(data)

        return data

    # =================================================================
    # Internal loaders
    # =================================================================

    def _load_calibration(self, data: LoadedData):
        """Load cameras.mat."""
        calib_dir = os.path.join(data.data_dir, "calibration")
        cameras_path = os.path.join(calib_dir, "cameras.mat")

        if not os.path.exists(cameras_path):
            self._errors.append(f"cameras.mat not found: {cameras_path}")
            return

        try:
            # Use the project's own loader
            from ndef_dic.step1_pipeline import load_calibration
            calib = load_calibration(calib_dir)

            data.K_list = [K.astype(np.float32) for K in calib["K_list"]]
            data.R_list = [R.astype(np.float32) for R in calib["R_list"]]
            data.t_list = [t.astype(np.float32).reshape(3) for t in calib["t_list"]]
            data.dist_list = calib.get("dist_list")
            data.camera_models = calib.get("camera_models")
            data.P_list = calib.get("P_list")
            data.num_cameras = calib["num_cameras"]

            # Derive camera centers
            centers = []
            for R, t in zip(data.R_list, data.t_list):
                C = (-R.T @ t.reshape(3, 1)).ravel()
                centers.append(C)
            data.camera_centers = np.array(centers, dtype=np.float32)

            print(f"[DataLoader] Loaded {data.num_cameras} cameras")

        except Exception as e:
            self._errors.append(f"Failed to load calibration: {e}")

    def _load_sparse(self, data: LoadedData):
        """Load sparse SfM points from points3D.mat."""
        calib_dir = os.path.join(data.data_dir, "calibration")
        pts_path = os.path.join(calib_dir, "points3D.mat")

        if not os.path.exists(pts_path):
            return  # Not an error — may not exist

        try:
            from ndef_dic.step1_pipeline import _load_sparse_points
            pts = _load_sparse_points(calib_dir)
            if pts is not None:
                data.sparse_points = pts.astype(np.float32)
                print(f"[DataLoader] Loaded {len(pts)} sparse points")
        except Exception as e:
            self._errors.append(f"Failed to load sparse points: {e}")

    def _load_dense(self, data: LoadedData):
        """Load dense point cloud and visibility data."""
        dense_dir = os.path.join(data.data_dir, "calibration", "dense")

        # PLY point cloud
        ply_path = os.path.join(dense_dir, "dense_points.ply")
        npy_path = os.path.join(dense_dir, "dense_points.npy")

        try:
            if os.path.exists(npy_path):
                data.dense_points = np.load(npy_path).astype(np.float32)
                print(f"[DataLoader] Loaded dense points from .npy: {data.dense_points.shape}")
            elif os.path.exists(ply_path):
                from ndef_dic.dense_mvs import load_ply
                pts, nrm = load_ply(ply_path)
                data.dense_points = pts.astype(np.float32)
                if nrm is not None:
                    data.dense_normals = nrm.astype(np.float32)
                print(f"[DataLoader] Loaded dense points from PLY: {data.dense_points.shape}")
        except Exception as e:
            self._errors.append(f"Failed to load dense points: {e}")

        # Normals
        normals_path = os.path.join(dense_dir, "dense_normals.npy")
        if data.dense_normals is None and os.path.exists(normals_path):
            try:
                data.dense_normals = np.load(normals_path).astype(np.float32)
            except Exception as e:
                self._errors.append(f"Failed to load dense normals: {e}")

        # Visibility mask
        vis_path = os.path.join(dense_dir, "vis_mask.npy")
        if os.path.exists(vis_path):
            try:
                data.dense_vis_mask = np.load(vis_path)
                print(f"[DataLoader] Loaded vis_mask: {data.dense_vis_mask.shape}")
            except Exception as e:
                self._errors.append(f"Failed to load vis_mask: {e}")

        # Meta
        meta_path = os.path.join(dense_dir, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    data.dense_meta = json.load(f)
            except Exception as e:
                self._errors.append(f"Failed to load dense meta: {e}")

    def _load_displacement(self, data: LoadedData):
        """Load displacement field results."""
        results_dir = os.path.join(data.data_dir, "results")

        # Reference points
        ref_path = os.path.join(results_dir, "ref_points.npy")
        if not os.path.exists(ref_path):
            return

        try:
            data.ref_points = np.load(ref_path).astype(np.float32)
            print(f"[DataLoader] Loaded ref_points: {data.ref_points.shape}")
        except Exception as e:
            self._errors.append(f"Failed to load ref_points: {e}")
            return

        # Discover displacement steps
        disp_files = sorted(glob.glob(os.path.join(results_dir, "disp_step*.npy")))
        if not disp_files:
            return

        data.disp_fields = {}
        data.def_points = {}

        for f in disp_files:
            basename = os.path.basename(f)
            # Parse step number from "disp_step003.npy"
            step_str = basename.replace("disp_step", "").replace(".npy", "")
            try:
                step = int(step_str)
            except ValueError:
                continue

            try:
                data.disp_fields[step] = np.load(f).astype(np.float32)
            except Exception as e:
                self._errors.append(f"Failed to load {basename}: {e}")
                continue

            # Corresponding deformed points
            def_f = os.path.join(results_dir, f"def_points_step{step:03d}.npy")
            if os.path.exists(def_f):
                try:
                    data.def_points[step] = np.load(def_f).astype(np.float32)
                except Exception as e:
                    self._errors.append(f"Failed to load def_points_step{step:03d}.npy: {e}")

        data.n_steps = len(data.disp_fields)
        print(f"[DataLoader] Loaded {data.n_steps} displacement steps")

        # Meta
        meta_path = os.path.join(results_dir, "results_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    data.results_meta = json.load(f)
            except Exception as e:
                self._errors.append(f"Failed to load results_meta: {e}")

    def _load_ground_truth(self, data: LoadedData):
        """Load ground truth displacement data if available.

        Looks for:
          ground_truth/points_ref.npy          — reference points
          ground_truth/displacement_step*.npy  — displacement per step
        """
        gt_dir = os.path.join(data.data_dir, "ground_truth")
        if not os.path.isdir(gt_dir):
            return

        # Reference points
        ref_path = os.path.join(gt_dir, "points_ref.npy")
        if not os.path.exists(ref_path):
            return

        try:
            data.gt_ref_points = np.load(ref_path).astype(np.float32)
            print(f"[DataLoader] Loaded GT ref_points: {data.gt_ref_points.shape}")
        except Exception as e:
            self._errors.append(f"Failed to load GT ref_points: {e}")
            return

        # Discover displacement steps
        disp_files = sorted(glob.glob(os.path.join(gt_dir, "displacement_step*.npy")))
        if not disp_files:
            return

        data.gt_disp_fields = {}
        for f in disp_files:
            basename = os.path.basename(f)
            step_str = basename.replace("displacement_step", "").replace(".npy", "")
            try:
                step = int(step_str)
            except ValueError:
                continue

            try:
                data.gt_disp_fields[step] = np.load(f).astype(np.float32)
            except Exception as e:
                self._errors.append(f"Failed to load GT {basename}: {e}")

        data.gt_n_steps = len(data.gt_disp_fields)
        print(f"[DataLoader] Loaded {data.gt_n_steps} GT displacement steps")

    def _load_images(self, data: LoadedData):
        """Discover camera directories and image paths."""
        image_dir = os.path.join(data.data_dir, "images")
        if not os.path.isdir(image_dir):
            return

        # Discover camera directories (cam_0, cam_1, ...)
        cam_names = sorted([
            d for d in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, d)) and d.startswith("cam_")
        ])

        if not cam_names:
            # Try non-prefixed directories (e.g., "0", "1", ...)
            cam_names = sorted([
                d for d in os.listdir(image_dir)
                if os.path.isdir(os.path.join(image_dir, d))
            ])
            # Filter: only keep directories that contain image files
            cam_names = [
                d for d in cam_names
                if any(
                    f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"))
                    for f in os.listdir(os.path.join(image_dir, d))
                )
            ]

        if not cam_names:
            return

        data.cam_names = cam_names
        n_cam = len(cam_names)

        # Find reference image per camera
        ref_paths = []
        for cam in cam_names:
            cam_dir = os.path.join(image_dir, cam)
            ref_file = self._find_image(cam_dir, "001")
            ref_paths.append(ref_file)

        data.ref_image_paths = ref_paths
        print(f"[DataLoader] Found {n_cam} cameras with reference images")

        # Find deformed images across all cameras
        # Collect all image files, group by step number
        def_paths: Dict[int, List[Optional[str]]] = {}

        for cam in cam_names:
            cam_dir = os.path.join(image_dir, cam)
            all_imgs = sorted([
                f for f in os.listdir(cam_dir)
                if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"))
            ])

            for img_name in all_imgs:
                # Try to parse step number from filename
                stem = os.path.splitext(img_name)[0]
                try:
                    step_num = int(stem)
                except ValueError:
                    continue

                if step_num == 1:
                    continue  # This is the reference, already handled

                # Convert 1-indexed to our step indexing
                step_idx = step_num - 1
                if step_idx not in def_paths:
                    def_paths[step_idx] = [None] * n_cam

                cam_idx = cam_names.index(cam)
                def_paths[step_idx][cam_idx] = os.path.join(cam_dir, img_name)

        if def_paths:
            data.def_image_paths = def_paths
            data.images_per_step = len(def_paths)
            print(f"[DataLoader] Found {data.images_per_step} deformed image sets")

    @staticmethod
    def _find_image(cam_dir: str, ref_name: str) -> Optional[str]:
        """Find an image by base name, trying common extensions."""
        for ext in [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            candidate = os.path.join(cam_dir, f"{ref_name}{ext}")
            if os.path.exists(candidate):
                return candidate
        # Fallback: first image file
        files = sorted([
            f for f in os.listdir(cam_dir)
            if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ])
        return os.path.join(cam_dir, files[0]) if files else None
