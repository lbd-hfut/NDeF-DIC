"""Build the multi-view reconstruction dataset for future ZNSSD optimisation.

The dataset is centred on source-camera ROI pixels.  Each source pixel already
has an initial SfM-scale 3-D point from ``model_init``.  We keep a source sample
only when this 3-D point projects into the ROI masks of the selected adjacent
camera(s).  Patch pixels for ZNSSD are not expanded per sample; instead, a
global patch-offset table is saved and the loss can generate patch coordinates
on demand.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ReconstructionDatasetConfig:
    data_dir: str = "case/CylinderDIC"
    sfm_dir: str | None = None
    model_init_dir: str | None = None
    output_dir: str | None = None
    neighbor_angle_deg: float = 45.0
    max_neighbors: int = 2
    patch_radius: int = 2
    projection_batch_size: int = 262144
    require_source_patch_inside_image: bool = True
    require_target_patch_inside_image: bool = True
    save_combined_npz: bool = False


def run_reconstruction_dataset(
    config: ReconstructionDatasetConfig | None = None,
) -> Dict[str, str]:
    cfg = config or ReconstructionDatasetConfig()
    data_dir = Path(cfg.data_dir)
    sfm_dir = Path(cfg.sfm_dir) if cfg.sfm_dir else data_dir / "result" / "sfm"
    model_init_dir = (
        Path(cfg.model_init_dir)
        if cfg.model_init_dir
        else data_dir / "result" / "dense" / "model_init"
    )
    output_dir = (
        Path(cfg.output_dir)
        if cfg.output_dir
        else data_dir / "result" / "dense" / "reconstruction_dataset"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = _load_npz(sfm_dir / "cameras.npz")
    cam_names = [str(x) for x in cameras["cam_names"]]
    image_sizes = _load_image_sizes(cameras["image_paths"])
    K = cameras["K"].astype(np.float64)
    dist = cameras["dist"].astype(np.float64)
    R = cameras["R"].astype(np.float64)
    t = cameras["t"].astype(np.float64).reshape(len(cam_names), 3)

    dense_items = _load_model_init_dense(model_init_dir, cam_names)
    neighbor_table = _select_camera_neighbors(R, cfg.neighbor_angle_deg, cfg.max_neighbors)
    patch_offsets = _make_patch_offsets(cfg.patch_radius)

    shard_records = []
    n_samples_total = 0
    shard_root = output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    for cam_id, cam_name in enumerate(cam_names):
        neighbors = neighbor_table[cam_id]
        if not neighbors:
            print(
                f"[ReconDataset] {cam_name}: no adjacent cameras within "
                f"{cfg.neighbor_angle_deg:.1f} deg. Adjust camera angles or threshold."
            )
            continue

        item = dense_items[cam_id]
        samples = _build_single_camera_samples(
            cam_id=cam_id,
            source_uv=item["pixels"].astype(np.float64),
            source_world=item["world"].astype(np.float64),
            source_mask=item["roi_mask"].astype(bool),
            neighbors=neighbors,
            dense_items=dense_items,
            image_sizes=image_sizes,
            K=K,
            dist=dist,
            R=R,
            t=t,
            cfg=cfg,
        )
        shard_dir = shard_root / cam_name
        shard_record = _save_sample_shard(shard_dir, cam_name, samples, n_samples_total)
        shard_records.append(shard_record)
        n_samples_total += int(shard_record["n_samples"])

        kept = len(samples["source_uv"])
        total = len(item["pixels"])
        print(
            f"[ReconDataset] {cam_name}: neighbors={samples['selected_neighbor_ids'].tolist()} "
            f"kept={kept}/{total} samples"
        )

    np.save(output_dir / "patch_offsets.npy", patch_offsets)
    if cfg.save_combined_npz:
        _save_optional_combined_npz(output_dir / "reconstruction_dataset.npz", shard_records, patch_offsets)
    _save_manifest(output_dir, cfg, cam_names, image_sizes, shard_records, patch_offsets)
    _save_meta(
        output_dir=output_dir,
        cfg=cfg,
        cam_names=cam_names,
        image_sizes=image_sizes,
        neighbor_table=neighbor_table,
        shard_records=shard_records,
        n_samples=n_samples_total,
        patch_offsets=patch_offsets,
    )
    return {
        "manifest": str(output_dir / "dataset_manifest.json"),
        "meta": str(output_dir / "reconstruction_dataset_meta.json"),
    }


class ReconstructionMemmapDataset:
    """Memory-mapped reconstruction dataset for training-time reads.

    The constructor opens ``.npy`` files with ``mmap_mode='r'``.  It does not
    load all samples into RAM or GPU memory.  A training loop can either use
    ``__getitem__`` through a PyTorch ``DataLoader`` or call ``get_batch`` with
    integer indices for grouped shard reads.
    """

    fields = ("source_cam", "source_uv", "source_world", "neighbor_ids", "neighbor_uv")

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.patch_offsets = np.load(self.manifest["patch_offsets"]["path"], mmap_mode="r")
        self.shards = []
        self.starts = []
        self.stops = []
        for record in self.manifest["shards"]:
            arrays = {
                name: np.load(record["arrays"][name]["path"], mmap_mode="r")
                for name in self.fields
            }
            self.shards.append({"record": record, "arrays": arrays})
            self.starts.append(int(record["start"]))
            self.stops.append(int(record["stop"]))
        self.n_samples = int(self.manifest["n_samples"])

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        if index < 0:
            index += self.n_samples
        shard_idx, local_idx = self._locate(index)
        arrays = self.shards[shard_idx]["arrays"]
        return {name: arrays[name][local_idx] for name in self.fields}

    def get_batch(self, indices: np.ndarray | List[int]) -> Dict[str, np.ndarray]:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("indices must be a 1-D integer array")
        if len(indices) == 0:
            return {name: _empty_array_for(name, self.max_neighbors) for name in self.fields}

        order = np.argsort(indices)
        sorted_indices = indices[order]
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))

        chunks = {name: [] for name in self.fields}
        cursor = 0
        while cursor < len(sorted_indices):
            shard_idx, local_idx = self._locate(int(sorted_indices[cursor]))
            shard_stop = self.stops[shard_idx]
            next_cursor = cursor
            local_indices = []
            while next_cursor < len(sorted_indices) and sorted_indices[next_cursor] < shard_stop:
                local_indices.append(int(sorted_indices[next_cursor] - self.starts[shard_idx]))
                next_cursor += 1
            arrays = self.shards[shard_idx]["arrays"]
            local_indices_arr = np.asarray(local_indices, dtype=np.int64)
            for name in self.fields:
                chunks[name].append(np.asarray(arrays[name][local_indices_arr]))
            cursor = next_cursor

        batch = {}
        for name in self.fields:
            stacked = np.concatenate(chunks[name], axis=0)
            batch[name] = stacked[inverse]
        return batch

    @property
    def max_neighbors(self) -> int:
        return int(self.manifest["config"]["max_neighbors"])

    def _locate(self, index: int) -> Tuple[int, int]:
        if index < 0 or index >= self.n_samples:
            raise IndexError(index)
        shard_idx = bisect_right(self.stops, index)
        local_idx = index - self.starts[shard_idx]
        return shard_idx, local_idx


class BalancedPerCameraBatchLoader:
    """Balanced source-camera batch iterator.

    Each iteration draws ``per_camera_batch`` samples from every non-empty source
    camera shard and concatenates them into one batch.  Per-camera sample order
    is shuffled independently.  If a camera exhausts before the epoch ends, it
    reshuffles and wraps around, so every iteration remains camera-balanced.

    One epoch is defined as enough iterations for the largest camera shard to be
    traversed once:

        ``ceil(max_i(N_i) / per_camera_batch)``

    This makes every epoch cover all samples at least once while keeping every
    iteration physically balanced across cameras.
    """

    fields = ReconstructionMemmapDataset.fields

    def __init__(
        self,
        manifest_path: str | Path,
        per_camera_batch: int,
        shuffle: bool = True,
        seed: int | None = None,
        drop_last: bool = False,
    ):
        if per_camera_batch <= 0:
            raise ValueError("per_camera_batch must be positive")
        self.dataset = ReconstructionMemmapDataset(manifest_path)
        self.per_camera_batch = int(per_camera_batch)
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.drop_last = bool(drop_last)
        self.rng = np.random.default_rng(seed)
        self.active_shards = [
            idx
            for idx, shard in enumerate(self.dataset.shards)
            if int(shard["record"]["n_samples"]) > 0
        ]
        if not self.active_shards:
            raise ValueError("No non-empty camera shards found")
        self.steps_per_epoch = self._compute_steps_per_epoch()
        self._orders: Dict[int, np.ndarray] = {}
        self._positions: Dict[int, int] = {}
        self._step = 0
        self.reset_epoch()

    def __iter__(self):
        self.reset_epoch()
        return self

    def __next__(self) -> Dict[str, np.ndarray]:
        if self._step >= self.steps_per_epoch:
            raise StopIteration
        chunks = {name: [] for name in self.fields}
        source_shards = []
        for shard_idx in self.active_shards:
            local_indices = self._take_local_indices(shard_idx, self.per_camera_batch)
            arrays = self.dataset.shards[shard_idx]["arrays"]
            for name in self.fields:
                chunks[name].append(np.asarray(arrays[name][local_indices]))
            source_shards.append(
                np.full(len(local_indices), shard_idx, dtype=np.int16)
            )

        batch = {}
        for name in self.fields:
            batch[name] = np.concatenate(chunks[name], axis=0)
        batch["source_shard"] = np.concatenate(source_shards, axis=0)
        batch["patch_offsets"] = self.dataset.patch_offsets
        self._step += 1
        return batch

    def __len__(self) -> int:
        return self.steps_per_epoch

    @property
    def batch_size(self) -> int:
        return self.per_camera_batch * len(self.active_shards)

    def reset_epoch(self) -> None:
        self._step = 0
        for shard_idx in self.active_shards:
            n = int(self.dataset.shards[shard_idx]["record"]["n_samples"])
            self._orders[shard_idx] = self._new_order(n)
            self._positions[shard_idx] = 0

    def _compute_steps_per_epoch(self) -> int:
        counts = [
            int(self.dataset.shards[idx]["record"]["n_samples"])
            for idx in self.active_shards
        ]
        if self.drop_last:
            return max(1, max(count // self.per_camera_batch for count in counts))
        return max(1, math.ceil(max(counts) / self.per_camera_batch))

    def _new_order(self, n: int) -> np.ndarray:
        if self.shuffle:
            return self.rng.permutation(n).astype(np.int64)
        return np.arange(n, dtype=np.int64)

    def _take_local_indices(self, shard_idx: int, count: int) -> np.ndarray:
        record = self.dataset.shards[shard_idx]["record"]
        n = int(record["n_samples"])
        if n == 0:
            raise ValueError(f"Shard {record['cam_name']} is empty")
        pieces = []
        remaining = count
        while remaining > 0:
            pos = self._positions[shard_idx]
            order = self._orders[shard_idx]
            available = n - pos
            take = min(remaining, available)
            pieces.append(order[pos : pos + take])
            pos += take
            remaining -= take
            if pos >= n:
                order = self._new_order(n)
                pos = 0
            self._orders[shard_idx] = order
            self._positions[shard_idx] = pos
        return np.concatenate(pieces, axis=0)


def _build_single_camera_samples(
    cam_id: int,
    source_uv: np.ndarray,
    source_world: np.ndarray,
    source_mask: np.ndarray,
    neighbors: List[Tuple[int, float]],
    dense_items: List[Dict[str, np.ndarray]],
    image_sizes: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    cfg: ReconstructionDatasetConfig,
) -> Dict[str, np.ndarray]:
    selected_ids = [idx for idx, _ in neighbors]
    n_neighbors = len(selected_ids)
    n_total = len(source_uv)
    keep = np.ones(n_total, dtype=bool)

    if cfg.require_source_patch_inside_image and cfg.patch_radius > 0:
        keep &= _inside_image_with_margin(source_uv, image_sizes[cam_id], cfg.patch_radius)

    projected_by_neighbor = np.full((n_total, cfg.max_neighbors, 2), np.nan, dtype=np.float32)
    neighbor_ids = np.full((n_total, cfg.max_neighbors), -1, dtype=np.int16)

    for local_idx, neighbor_id in enumerate(selected_ids):
        uv_proj, depth = _project_world_points_batched(
            source_world,
            K[neighbor_id],
            dist[neighbor_id],
            R[neighbor_id],
            t[neighbor_id],
            cfg.projection_batch_size,
        )
        target_mask = dense_items[neighbor_id]["roi_mask"].astype(bool)
        valid = depth > 1e-8
        valid &= _points_inside_mask(uv_proj, target_mask)
        if cfg.require_target_patch_inside_image and cfg.patch_radius > 0:
            valid &= _inside_image_with_margin(uv_proj, image_sizes[neighbor_id], cfg.patch_radius)
        keep &= valid
        projected_by_neighbor[:, local_idx, :] = uv_proj.astype(np.float32)
        neighbor_ids[:, local_idx] = neighbor_id

    return {
        "source_cam": np.full(keep.sum(), cam_id, dtype=np.int16),
        "source_uv": source_uv[keep].astype(np.float32),
        "source_world": source_world[keep].astype(np.float32),
        "neighbor_ids": neighbor_ids[keep],
        "neighbor_uv": projected_by_neighbor[keep],
        "selected_neighbor_ids": np.asarray(selected_ids, dtype=np.int16),
        "selected_neighbor_angles_deg": np.asarray([angle for _, angle in neighbors], dtype=np.float32),
        "n_source_roi_pixels": np.asarray(n_total, dtype=np.int64),
        "n_kept_samples": np.asarray(keep.sum(), dtype=np.int64),
    }


def _save_sample_shard(
    shard_dir: Path,
    cam_name: str,
    samples: Dict[str, np.ndarray],
    start_index: int,
) -> Dict:
    shard_dir.mkdir(parents=True, exist_ok=True)
    array_names = ["source_cam", "source_uv", "source_world", "neighbor_ids", "neighbor_uv"]
    arrays = {name: samples[name] for name in array_names}
    for name, array in arrays.items():
        np.save(shard_dir / f"{name}.npy", array)
    np.save(shard_dir / "selected_neighbor_ids.npy", samples["selected_neighbor_ids"])
    np.save(shard_dir / "selected_neighbor_angles_deg.npy", samples["selected_neighbor_angles_deg"])

    n_samples = int(len(samples["source_uv"]))
    return {
        "cam_name": cam_name,
        "path": str(shard_dir),
        "start": int(start_index),
        "stop": int(start_index + n_samples),
        "n_samples": n_samples,
        "arrays": {
            name: {
                "path": str(shard_dir / f"{name}.npy"),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            for name, array in arrays.items()
        },
        "selected_neighbor_ids": samples["selected_neighbor_ids"].astype(int).tolist(),
        "selected_neighbor_angles_deg": samples["selected_neighbor_angles_deg"].astype(float).tolist(),
        "n_source_roi_pixels": int(samples["n_source_roi_pixels"]),
        "n_kept_samples": int(samples["n_kept_samples"]),
    }


def _save_optional_combined_npz(
    path: Path,
    shard_records: List[Dict],
    patch_offsets: np.ndarray,
) -> None:
    arrays = {
        "source_cam": [],
        "source_uv": [],
        "source_world": [],
        "neighbor_ids": [],
        "neighbor_uv": [],
    }
    for record in shard_records:
        if record["n_samples"] == 0:
            continue
        for name in arrays:
            arrays[name].append(np.load(record["arrays"][name]["path"], mmap_mode="r"))
    combined = {}
    for name, chunks in arrays.items():
        combined[name] = (
            np.concatenate(chunks, axis=0)
            if chunks
            else _empty_array_for(name, max(record["arrays"]["neighbor_ids"]["shape"][1] for record in shard_records))
        )
    np.savez_compressed(path, **combined, patch_offsets=patch_offsets)


def _empty_array_for(name: str, max_neighbors: int) -> np.ndarray:
    if name == "source_cam":
        return np.empty((0,), dtype=np.int16)
    if name == "source_uv":
        return np.empty((0, 2), dtype=np.float32)
    if name == "source_world":
        return np.empty((0, 3), dtype=np.float32)
    if name == "neighbor_ids":
        return np.empty((0, max_neighbors), dtype=np.int16)
    if name == "neighbor_uv":
        return np.empty((0, max_neighbors, 2), dtype=np.float32)
    raise KeyError(name)


def _save_manifest(
    output_dir: Path,
    cfg: ReconstructionDatasetConfig,
    cam_names: List[str],
    image_sizes: np.ndarray,
    shard_records: List[Dict],
    patch_offsets: np.ndarray,
) -> None:
    manifest = {
        "format": "ndef_dic.reconstruction_dataset.sharded_npy.v1",
        "loading": "Use np.load(path, mmap_mode='r') or ReconstructionMemmapDataset.",
        "config": asdict(cfg),
        "cam_names": cam_names,
        "image_sizes_wh": image_sizes.tolist(),
        "n_samples": int(sum(record["n_samples"] for record in shard_records)),
        "patch_offsets": {
            "path": str(output_dir / "patch_offsets.npy"),
            "shape": list(patch_offsets.shape),
            "dtype": str(patch_offsets.dtype),
        },
        "shards": shard_records,
    }
    with open(output_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _select_camera_neighbors(
    R: np.ndarray,
    max_angle_deg: float,
    max_neighbors: int,
) -> List[List[Tuple[int, float]]]:
    axes = np.asarray([_optical_axis_world(r) for r in R], dtype=np.float64)
    table: List[List[Tuple[int, float]]] = []
    for i in range(len(R)):
        candidates = []
        for j in range(len(R)):
            if i == j:
                continue
            dot = float(np.clip(np.dot(axes[i], axes[j]), -1.0, 1.0))
            angle = math.degrees(math.acos(dot))
            if angle <= max_angle_deg:
                candidates.append((j, angle))
        candidates.sort(key=lambda item: item[1])
        table.append(candidates[:max_neighbors])
    return table


def _optical_axis_world(R: np.ndarray) -> np.ndarray:
    axis = R.T @ np.array([0.0, 0.0, 1.0])
    return axis / np.linalg.norm(axis)


def _project_world_points_batched(
    points_world: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    import cv2

    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    uv_chunks = []
    depth_chunks = []
    for start in range(0, len(points_world), batch_size):
        stop = min(start + batch_size, len(points_world))
        pts = points_world[start:stop].astype(np.float64)
        uv, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, t.reshape(3, 1), K, dist)
        uv_chunks.append(uv.reshape(-1, 2))
        depth = (R @ pts.T + t.reshape(3, 1))[2]
        depth_chunks.append(depth)
    return np.concatenate(uv_chunks, axis=0), np.concatenate(depth_chunks, axis=0)


def _points_inside_mask(uv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    rounded = np.rint(uv).astype(np.int64)
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    out = np.zeros(len(uv), dtype=bool)
    valid_idx = np.where(inside)[0]
    out[valid_idx] = mask[rounded[valid_idx, 1], rounded[valid_idx, 0]]
    return out


def _inside_image_with_margin(uv: np.ndarray, image_size: np.ndarray, margin: int) -> np.ndarray:
    width, height = int(image_size[0]), int(image_size[1])
    return (
        (uv[:, 0] >= margin)
        & (uv[:, 0] <= width - 1 - margin)
        & (uv[:, 1] >= margin)
        & (uv[:, 1] <= height - 1 - margin)
    )


def _make_patch_offsets(radius: int) -> np.ndarray:
    offsets = []
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            offsets.append((du, dv))
    return np.asarray(offsets, dtype=np.float32)


def _load_model_init_dense(model_init_dir: Path, cam_names: List[str]) -> List[Dict[str, np.ndarray]]:
    dense_dir = model_init_dir / "per_camera_dense"
    if not dense_dir.exists():
        raise FileNotFoundError(f"Model-init dense directory not found: {dense_dir}")
    items = []
    for cam_name in cam_names:
        path = dense_dir / f"{cam_name}_dense_init.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True)
        if "roi_mask" in data.files:
            roi_mask = data["roi_mask"].astype(bool)
        elif "mask" in data.files:
            roi_mask = data["mask"].astype(bool)
        else:
            raise KeyError(f"{path} has no roi_mask/mask field")
        items.append(
            {
                "pixels": data["pixels"],
                "world": data["world"],
                "roi_mask": roi_mask,
            }
        )
    return items


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _load_image_sizes(image_paths: np.ndarray) -> np.ndarray:
    import cv2

    sizes = []
    for raw_path in image_paths:
        image = cv2.imread(str(Path(str(raw_path))), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(str(raw_path))
        height, width = image.shape[:2]
        sizes.append((width, height))
    return np.asarray(sizes, dtype=np.int64)


def _save_meta(
    output_dir: Path,
    cfg: ReconstructionDatasetConfig,
    cam_names: List[str],
    image_sizes: np.ndarray,
    neighbor_table: List[List[Tuple[int, float]]],
    shard_records: List[Dict],
    n_samples: int,
    patch_offsets: np.ndarray,
) -> None:
    meta = {
        "purpose": "Multi-view source-pixel dataset for future ZNSSD residual minimisation",
        "storage": {
            "format": "sharded uncompressed .npy arrays",
            "reason": "npz compression is not mmap-friendly; shards support training-time batch reads.",
            "manifest": str(output_dir / "dataset_manifest.json"),
        },
        "sample_schema": {
            "source_cam": "(N,) int camera id",
            "source_uv": "(N,2) source integer pixel centre [u,v]",
            "source_world": "(N,3) initial SfM-scale 3-D point from model_init",
            "neighbor_ids": f"(N,{cfg.max_neighbors}) target camera ids, -1 for absent",
            "neighbor_uv": f"(N,{cfg.max_neighbors},2) initial target projection centres",
            "patch_offsets": "(patch_size^2,2) [du,dv], applied on demand in the loss",
        },
        "config": asdict(cfg),
        "cam_names": cam_names,
        "image_sizes_wh": image_sizes.tolist(),
        "neighbors": [
            {
                "source_cam": cam_names[i],
                "targets": [
                    {"cam_id": int(j), "cam_name": cam_names[j], "angle_deg": float(angle)}
                    for j, angle in row
                ],
            }
            for i, row in enumerate(neighbor_table)
        ],
        "n_samples": int(n_samples),
        "patch_size": int(2 * cfg.patch_radius + 1),
        "patch_offsets_count": int(len(patch_offsets)),
        "shards": shard_records,
    }
    with open(output_dir / "reconstruction_dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
