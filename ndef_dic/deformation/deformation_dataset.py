"""Dataset utilities for reference-surface deformation training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


IMAGE_EXTENSIONS = (".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg")


@dataclass
class DeformationDatasetConfig:
    data_dir: str = "case/CylinderDIC"
    sfm_dir: str | None = None
    surface_dataset_path: str | None = None
    image_dir: str = "images"
    reference_name: str = "001"
    current_name: str = "002"
    device: str = "auto"


class SurfaceDeformationDataset:
    """Reference surface samples, cameras, and reference/current images."""

    def __init__(self, config: DeformationDatasetConfig | None = None) -> None:
        self.config = config or DeformationDatasetConfig()
        self.data_dir = Path(self.config.data_dir)
        self.sfm_dir = Path(self.config.sfm_dir) if self.config.sfm_dir else self.data_dir / "result" / "sfm"
        self.surface_dataset_path = (
            Path(self.config.surface_dataset_path)
            if self.config.surface_dataset_path
            else self.data_dir / "result" / "dense" / "surface_sampler" / "deformation_surface_dataset.npz"
        )
        self.device = select_device(self.config.device)

        surface = np.load(self.surface_dataset_path, allow_pickle=True)
        self.points = torch.as_tensor(surface["points"], dtype=torch.float32, device=self.device)
        self.normals = torch.as_tensor(surface["normals"], dtype=torch.float32, device=self.device)
        self.visibility_mask = torch.as_tensor(surface["visibility_mask"], dtype=torch.bool, device=self.device)
        self.projected_uv = torch.as_tensor(surface["projected_uv"], dtype=torch.float32, device=self.device)
        self.visible_counts = torch.as_tensor(surface["visible_counts"], dtype=torch.float32, device=self.device)
        self.cam_names = [str(x) for x in surface["cam_names"]]
        if len(self.points) == 0:
            raise ValueError(f"No surface points in {self.surface_dataset_path}")

        cameras_np = load_npz(self.sfm_dir / "cameras.npz")
        self.cameras = {
            "K": torch.as_tensor(cameras_np["K"], dtype=torch.float32, device=self.device),
            "dist": torch.as_tensor(cameras_np["dist"], dtype=torch.float32, device=self.device),
            "R": torch.as_tensor(cameras_np["R"], dtype=torch.float32, device=self.device),
            "t": torch.as_tensor(cameras_np["t"].reshape(len(self.cam_names), 3), dtype=torch.float32, device=self.device),
        }
        self.image_sizes, self.reference_images, self.current_images = load_image_pair_stack(
            data_dir=self.data_dir,
            image_dir=self.config.image_dir,
            cam_names=self.cam_names,
            reference_name=self.config.reference_name,
            current_name=self.config.current_name,
            device=self.device,
        )

        points_cpu = self.points.detach().cpu()
        coord_min = points_cpu.min(dim=0).values
        coord_max = points_cpu.max(dim=0).values
        self.coord_center = ((coord_min + coord_max) * 0.5).to(self.device)
        self.coord_scale = ((coord_max - coord_min) * 0.5).clamp_min(1e-8).to(self.device)

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_cameras(self) -> int:
        return len(self.cam_names)

    def sample_indices(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.n_points, (int(batch_size),), device=self.device)

    def batch(self, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "points": self.points[indices],
            "visibility_mask": self.visibility_mask[indices],
            "projected_uv": self.projected_uv[indices],
            "visible_counts": self.visible_counts[indices].clamp_min(1.0),
        }


def load_image_pair_stack(
    data_dir: Path,
    image_dir: str,
    cam_names: List[str],
    reference_name: str,
    current_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import cv2

    ref_images = []
    cur_images = []
    sizes = []
    image_root = data_dir / image_dir
    for cam_name in cam_names:
        cam_dir = image_root / cam_name
        ref_path = find_named_image(cam_dir, reference_name)
        cur_path = find_named_image(cam_dir, current_name)
        ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
        cur = cv2.imread(str(cur_path), cv2.IMREAD_GRAYSCALE)
        if ref is None:
            raise FileNotFoundError(str(ref_path))
        if cur is None:
            raise FileNotFoundError(str(cur_path))
        if ref.shape != cur.shape:
            raise ValueError(f"Reference/current shape mismatch in {cam_name}: {ref.shape} vs {cur.shape}")
        height, width = ref.shape
        sizes.append((width, height))
        ref_images.append(torch.as_tensor(ref.astype(np.float32) / 255.0)[None])
        cur_images.append(torch.as_tensor(cur.astype(np.float32) / 255.0)[None])
    return (
        torch.as_tensor(np.asarray(sizes, dtype=np.float32), device=device),
        torch.stack(ref_images, dim=0).to(device),
        torch.stack(cur_images, dim=0).to(device),
    )


def find_named_image(cam_dir: Path, stem: str) -> Path:
    if not cam_dir.is_dir():
        raise FileNotFoundError(str(cam_dir))
    for ext in IMAGE_EXTENSIONS:
        path = cam_dir / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image named {stem} with supported extension in {cam_dir}")


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[Deformation] CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)
