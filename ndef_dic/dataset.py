"""
Dataset classes for NDeF-DIC.

Organizes multi-camera, multi-frame DIC data:
  - Reference images (t=0) from all cameras.
  - Deformed images (t>0) from all cameras.
  - Camera parameters (K, R, t) from COLMAP.
  - Speckle masks per camera.
  - COLMAP sparse 3D points (filtered).
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import scipy.io as sio


@dataclass
class CameraParams:
    """Camera parameters for one camera."""
    K: np.ndarray       # (3, 3) intrinsic matrix
    R: np.ndarray       # (3, 3) world-to-camera rotation
    t: np.ndarray       # (3,) world-to-camera translation
    dist: np.ndarray    # (5,) distortion coefficients
    width: int
    height: int


class MultiCamDataset(Dataset):
    """
    Multi-camera DIC dataset.

    Data layout (expected directory structure):
        data_dir/
          cam_0/
            ref.png           (or .tif, .bmp)
            frame_001.png
            frame_002.png
            ...
          cam_1/
            ref.png
            frame_001.png
            ...
          ...
          calibration/
            cameras.mat       (COLMAP export: K, R, t, dist per camera)
            points3D.mat      (COLMAP sparse points, optional)
          masks/
            cam_0_mask.png    (optional, will auto-generate if missing)
            cam_1_mask.png
            ...
    """

    def __init__(
        self,
        data_dir: str,
        n_cameras: int = 4,
        n_load_steps: int = 10,
        image_height: int = 1200,
        image_width: int = 1920,
        ref_name: str = "ref",
        frame_pattern: str = "frame_{:03d}",
        image_ext: str = ".png",
        device: str = "cpu",
    ):
        self.data_dir = data_dir
        self.n_cameras = n_cameras
        self.n_load_steps = n_load_steps
        self.image_height = image_height
        self.image_width = image_width
        self.device = device

        # Load camera parameters
        self.camera_params = self._load_camera_params()

        # Load images
        self.ref_images: List[np.ndarray] = []        # [n_cameras] (H, W)
        self.def_images: Dict[int, List[np.ndarray]] = {}  # {step: [n_cameras] (H, W)}
        self.masks: List[np.ndarray] = []              # [n_cameras] (H, W)

        self._load_images(ref_name, frame_pattern, image_ext)
        self._load_or_generate_masks()

        # Load COLMAP points (if available)
        self.colmap_points: Optional[np.ndarray] = None
        self._load_colmap_points()

    def _load_camera_params(self) -> List[CameraParams]:
        """Load camera parameters from COLMAP calibration file."""
        calib_path = os.path.join(self.data_dir, "calibration", "cameras.mat")

        params = []
        if os.path.exists(calib_path):
            data = sio.loadmat(calib_path)
            K_list = data.get("K_list")
            R_list = data.get("cam_from_world_R")
            t_list = data.get("cam_from_world_t")
            dist_list = data.get("dist_list")

            for i in range(self.n_cameras):
                K = K_list[i] if K_list is not None else np.eye(3)
                R = R_list[i] if R_list is not None else np.eye(3)
                t_vec = t_list[i].ravel() if t_list is not None else np.zeros(3)
                dist = dist_list[i].ravel() if dist_list is not None else np.zeros(5)
                params.append(CameraParams(
                    K=K.astype(np.float32),
                    R=R.astype(np.float32),
                    t=t_vec.astype(np.float32),
                    dist=dist.astype(np.float32),
                    width=self.image_width,
                    height=self.image_height,
                ))
        else:
            # Fallback: identity camera params (for development)
            print(f"[WARNING] No calibration file found at {calib_path}")
            print(f"          Using identity camera parameters.")
            for i in range(self.n_cameras):
                K = np.array([
                    [2000, 0, self.image_width / 2],
                    [0, 2000, self.image_height / 2],
                    [0, 0, 1],
                ], dtype=np.float32)
                params.append(CameraParams(
                    K=K, R=np.eye(3, dtype=np.float32),
                    t=np.zeros(3, dtype=np.float32),
                    dist=np.zeros(5, dtype=np.float32),
                    width=self.image_width, height=self.image_height,
                ))

        return params

    def _load_images(self, ref_name: str, frame_pattern: str, image_ext: str):
        """Load all reference and deformed images."""
        import imageio.v3 as iio

        for cam_id in range(self.n_cameras):
            cam_dir = os.path.join(self.data_dir, f"cam_{cam_id}")

            # Reference image
            ref_path = None
            for ext in [image_ext, ".tif", ".tiff", ".bmp", ".jpg"]:
                candidate = os.path.join(cam_dir, f"{ref_name}{ext}")
                if os.path.exists(candidate):
                    ref_path = candidate
                    break

            if ref_path is None:
                raise FileNotFoundError(
                    f"Reference image not found in {cam_dir}/ "
                    f"(looked for {ref_name}{{.png,.tif,.bmp,.jpg}})"
                )

            ref_img = iio.imread(ref_path)
            if ref_img.ndim == 3:
                ref_img = ref_img[..., 0]  # take first channel if RGB
            ref_img = ref_img.astype(np.float32) / 255.0
            self.ref_images.append(ref_img)

            # Deformed images
            for step in range(1, self.n_load_steps):
                frame_name = frame_pattern.format(step)
                frame_path = None
                for ext in [image_ext, ".tif", ".tiff", ".bmp", ".jpg"]:
                    candidate = os.path.join(cam_dir, f"{frame_name}{ext}")
                    if os.path.exists(candidate):
                        frame_path = candidate
                        break

                if frame_path is None:
                    print(f"[WARNING] Frame {step} not found for camera {cam_id}, skipping")
                    continue

                def_img = iio.imread(frame_path)
                if def_img.ndim == 3:
                    def_img = def_img[..., 0]
                def_img = def_img.astype(np.float32) / 255.0

                if step not in self.def_images:
                    self.def_images[step] = [None] * self.n_cameras
                self.def_images[step][cam_id] = def_img

        # Update n_load_steps to actual loaded frames
        actual_steps = len(self.def_images) + 1  # +1 for ref
        if actual_steps < self.n_load_steps:
            print(f"[INFO] Loaded {actual_steps} load steps (requested {self.n_load_steps})")
            self.n_load_steps = actual_steps

    def _load_or_generate_masks(self):
        """Load pre-computed masks or mark for auto-generation."""
        masks_dir = os.path.join(self.data_dir, "masks")
        from .speckle_mask import SpeckleMaskGenerator, SpeckleMaskConfig

        for cam_id in range(self.n_cameras):
            mask_path = os.path.join(masks_dir, f"cam_{cam_id}_mask.png")
            if os.path.exists(mask_path):
                mask = SpeckleMaskGenerator.load_mask(mask_path)
                self.masks.append(mask)
            else:
                # Will be auto-generated
                self.masks.append(None)

    def _load_colmap_points(self):
        """Load filtered COLMAP sparse 3D points."""
        points_path = os.path.join(self.data_dir, "calibration", "points3D.mat")
        if os.path.exists(points_path):
            data = sio.loadmat(points_path)
            pts = data.get("points3D")  # (N, 3)
            if pts is not None:
                self.colmap_points = pts.astype(np.float32)
                print(f"[INFO] Loaded {len(self.colmap_points)} COLMAP 3D points")

    def ensure_masks(self, config: "SpeckleMaskConfig" = None):
        """
        Generate speckle masks if they don't exist.
        Should be called before training.
        """
        from .speckle_mask import SpeckleMaskGenerator
        generator = SpeckleMaskGenerator(config)
        refs = self.ref_images
        masks, _ = generator.generate(refs)
        for cam_id in range(self.n_cameras):
            self.masks[cam_id] = masks[cam_id]

    def get_camera_tensors(self, cam_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get camera parameters as tensors on the device."""
        p = self.camera_params[cam_id]
        return (
            torch.from_numpy(p.K).to(self.device),
            torch.from_numpy(p.R).to(self.device),
            torch.from_numpy(p.t).to(self.device),
        )

    def get_ref_image(self, cam_id: int) -> torch.Tensor:
        """Get reference image for a camera."""
        return torch.from_numpy(self.ref_images[cam_id].copy()).to(self.device)

    def get_def_image(self, cam_id: int, step: int) -> torch.Tensor:
        """Get deformed image for a camera at a load step."""
        img = self.def_images[step][cam_id]
        return torch.from_numpy(img.copy()).to(self.device)

    def get_mask(self, cam_id: int) -> Optional[torch.Tensor]:
        """Get speckle mask for a camera."""
        mask = self.masks[cam_id]
        if mask is None:
            return None
        return torch.from_numpy(mask.astype(np.float32) / 255.0).to(self.device)

    def get_colmap_points_tensor(self) -> Optional[torch.Tensor]:
        """Get COLMAP sparse points as a tensor."""
        if self.colmap_points is None:
            return None
        return torch.from_numpy(self.colmap_points).to(self.device)

    def sample_pixels(
        self,
        cam_id: int,
        n_pixels: int,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample random pixels from a camera's valid (masked) region.

        Args:
            cam_id: Camera index.
            n_pixels: Number of pixels to sample.
            step: Load step (0 = reference).

        Returns:
            pixels_uv: (N, 2) pixel coordinates (col, row).
            obs_image: (H, W) observed image tensor.
            mask: (H, W) or None — speckle mask.
        """
        mask = self.get_mask(cam_id)
        if step == 0:
            obs = self.get_ref_image(cam_id)
        else:
            obs = self.get_def_image(cam_id, step)

        H, W = self.image_height, self.image_width

        if mask is not None:
            # Sample only from masked (speckle) region
            valid_ys, valid_xs = torch.where(mask > 0.5)
            if len(valid_ys) == 0:
                # Fallback: sample from whole image
                xs = torch.randint(0, W, (n_pixels,), device=self.device)
                ys = torch.randint(0, H, (n_pixels,), device=self.device)
            else:
                idx = torch.randint(0, len(valid_ys), (n_pixels,), device=self.device)
                xs = valid_xs[idx].float()
                ys = valid_ys[idx].float()
        else:
            xs = torch.randint(0, W, (n_pixels,), device=self.device).float()
            ys = torch.randint(0, H, (n_pixels,), device=self.device).float()

        pixels_uv = torch.stack([xs, ys], dim=-1)  # (N, 2)

        return pixels_uv, obs, mask


class PixelBatchSampler:
    """
    Infinite sampler that yields random (camera, load_step, pixels) batches.

    Usage:
        sampler = PixelBatchSampler(dataset, batch_size=4096)
        for cam_id, step, pixels, obs, mask in sampler:
            ...
    """

    def __init__(
        self,
        dataset: MultiCamDataset,
        batch_size: int = 4096,
        stages: List[str] | None = None,
    ):
        """
        Args:
            dataset: MultiCamDataset.
            batch_size: Pixels per batch.
            stages: List of stage names to sample from.
                    None = all stages.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.stages = stages

    def __iter__(self):
        return self._sample_loop()

    def _sample_loop(self):
        """Yield batches indefinitely."""
        n_cameras = self.dataset.n_cameras
        n_steps = self.dataset.n_load_steps

        while True:
            # Random camera
            cam_id = np.random.randint(0, n_cameras)

            # Random load step (0 = reference, 1..n_steps-1 = deformed)
            if self.stages is not None and "intensity" in self.stages:
                step = 0  # only reference frames
            else:
                step = np.random.randint(0, n_steps)

            pixels, obs, mask = self.dataset.sample_pixels(
                cam_id, self.batch_size, step
            )

            yield cam_id, step, pixels, obs, mask
