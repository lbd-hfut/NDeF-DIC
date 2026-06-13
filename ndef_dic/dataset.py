"""
Step 3 multi-camera DIC image dataset.

Slim replacement for temp/ndef_dic/dataset.py, focused on Step 3's needs:
  - Reference and deformed image loading.
  - Camera parameter access (delegates to step1_pipeline.load_calibration).
  - Patch extraction (delegates to dic_losses.extract_patches).

No COLMAP points, mask generation, or pixel sampling — those are handled
by SurfaceProvider (Step 2) and the old pipeline.
"""

import os
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict


# Recognized image extensions (searched in order)
_IMAGE_EXTS = (".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg")


# =========================================================================
# Discovery helpers
# =========================================================================

def _is_image_file(fname: str) -> bool:
    return fname.lower().endswith(_IMAGE_EXTS)


def _list_image_files(directory: str) -> List[str]:
    """Sorted list of image files in a directory."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = sorted([f for f in os.listdir(directory) if _is_image_file(f)])
    if not files:
        raise FileNotFoundError(f"No image files in {directory}")
    return files


# =========================================================================
# MultiCamDataset
# =========================================================================

class MultiCamDataset:
    """Multi-camera DIC image dataset for Step 3.

    Data layout:
        data_dir/
          images/
            cam_0/
              001.bmp         ← reference (first in sorted order, or named)
              002.bmp         ← deformed step 1
              003.bmp         ← deformed step 2
            cam_1/
              ...

    Provides per-camera image access and a patch extraction convenience method.
    """

    def __init__(
        self,
        data_dir: str,
        image_dir: str = "images",
        calib_dir: str = "calibration",
        ref_mode: str = "first",     # "first" | "named"
        ref_name: str = "001",
        image_width: int = 1440,
        image_height: int = 1080,
        device: str = "cuda",
    ):
        self.data_dir = data_dir
        self.H = image_height
        self.W = image_width
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ---- Discover camera directories ----
        img_root = os.path.join(data_dir, image_dir)
        self.cam_names = sorted([
            d for d in os.listdir(img_root)
            if os.path.isdir(os.path.join(img_root, d))
        ])
        if not self.cam_names:
            raise FileNotFoundError(f"No camera folders under {img_root}")
        self.n_cameras = len(self.cam_names)

        # ---- Load camera parameters ----
        from .step1_pipeline import load_calibration
        calib_path = os.path.join(data_dir, calib_dir)
        calib_data = load_calibration(calib_path)
        self.calib = calib_data  # dict with K_list, R_list, t_list, etc.

        # ---- Load images ----
        self.ref_images: List[torch.Tensor] = []
        self.def_images: Dict[int, List[torch.Tensor]] = {}  # {step: [cam]}

        self._load_images(img_root, ref_mode, ref_name)
        self.n_steps = len(self.def_images)

        print(f"[Dataset] {self.n_cameras} cameras, {self.n_steps} deformed steps, "
              f"{self.H}×{self.W}")

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_images(self, img_root: str, ref_mode: str, ref_name: str):
        """Load reference and deformed images for all cameras."""
        import cv2

        for cam_id, cam_name in enumerate(self.cam_names):
            cam_dir = os.path.join(img_root, cam_name)
            files = _list_image_files(cam_dir)

            # Find reference index
            if ref_mode == "named":
                ref_idx = None
                for i, fname in enumerate(files):
                    base, _ = os.path.splitext(fname)
                    if base == ref_name:
                        ref_idx = i
                        break
                if ref_idx is None:
                    raise FileNotFoundError(
                        f"No '{ref_name}.*' in {cam_dir}. "
                        f"Set ref_mode='first' to use first image as reference."
                    )
            else:
                ref_idx = 0

            # Load reference
            ref_path = os.path.join(cam_dir, files[ref_idx])
            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            if ref_img is None:
                raise RuntimeError(f"Failed to read: {ref_path}")
            self.ref_images.append(
                torch.from_numpy(ref_img.astype(np.float32) / 255.0)
            )

            # Load deformed (all except reference)
            for step_idx, file_idx in enumerate(
                [i for i in range(len(files)) if i != ref_idx], start=1
            ):
                if step_idx not in self.def_images:
                    self.def_images[step_idx] = [None] * self.n_cameras
                def_path = os.path.join(cam_dir, files[file_idx])
                def_img = cv2.imread(def_path, cv2.IMREAD_GRAYSCALE)
                if def_img is None:
                    print(f"  [WARN] Failed to read deformed: {def_path}")
                    continue
                self.def_images[step_idx][cam_id] = torch.from_numpy(
                    def_img.astype(np.float32) / 255.0
                )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_ref_image(self, cam_id: int) -> torch.Tensor:
        """Return reference image for a camera as (H, W) float [0, 1] tensor."""
        return self.ref_images[cam_id].to(self.device)

    def get_def_image(self, cam_id: int, step: int) -> torch.Tensor:
        """Return deformed image at step for a camera as (H, W) float [0, 1] tensor."""
        if step not in self.def_images:
            raise KeyError(f"No deformed images for step {step}")
        img = self.def_images[step][cam_id]
        if img is None:
            raise ValueError(f"Camera {cam_id} has no deformed image at step {step}")
        return img.to(self.device)

    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------

    def extract_patches(
        self,
        image: torch.Tensor,        # (H, W) float tensor
        uv_centers: torch.Tensor,   # (N, 2) pixel coords (col, row)
        patch_size: int,
    ) -> torch.Tensor:
        """Extract square patches from an image.

        Thin wrapper around dic_losses.extract_patches().
        Returns (N, 1, patch_size, patch_size).
        """
        from .dic_losses import extract_patches
        return extract_patches(image, uv_centers, patch_size, self.H, self.W)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def image_shape(self) -> Tuple[int, int]:
        return (self.H, self.W)
