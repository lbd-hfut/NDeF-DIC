"""Reference-camera-centred SfM products for NDeF-DIC.

This module is the clean SfM boundary used by the project:

* input: first/reference image from each camera folder;
* output: all SfM products under ``case/<case>/result/sfm``;
* world frame: origin at sparse-point centroid, axes parallel to a reference
  camera, usually ``cam_0``;
* observations: true COLMAP track observations, not inferred FOV visibility.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SfMProducts:
    out_dir: str
    cam_names: List[str]
    K_list: List[np.ndarray]
    dist_list: List[np.ndarray]
    R_list: List[np.ndarray]
    t_list: List[np.ndarray]
    P_list: List[np.ndarray]
    points3D: np.ndarray
    observations: Dict[str, np.ndarray]


def _natural_key(text: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def _is_image_file(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in {
        ".bmp",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }


def _discover_camera_dirs(image_dir: str) -> List[str]:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    dirs = sorted(
        [
            d
            for d in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, d))
        ],
        key=_natural_key,
    )
    if not dirs:
        raise FileNotFoundError(f"No camera folders found under {image_dir}")
    return dirs


def _collect_reference_images(
    data_dir: str,
    image_dir: str = "images",
    ref_mode: str = "first",
    ref_name: str = "001",
) -> Dict[str, str]:
    """Collect one reference image per camera folder."""
    img_root = os.path.join(data_dir, image_dir)
    cam_dirs = _discover_camera_dirs(img_root)
    ref_images: Dict[str, str] = {}

    for cam_name in cam_dirs:
        cam_path = os.path.join(img_root, cam_name)
        files = sorted([f for f in os.listdir(cam_path) if _is_image_file(f)], key=_natural_key)
        if not files:
            raise FileNotFoundError(f"No image files found in {cam_path}")

        if ref_mode == "named":
            candidates = [f for f in files if os.path.splitext(f)[0] == ref_name]
            if not candidates:
                raise FileNotFoundError(f"No '{ref_name}.*' found in {cam_path}")
            ref_file = candidates[0]
        else:
            ref_file = files[0]
        ref_images[cam_name] = os.path.join(cam_path, ref_file)

    return ref_images


def _extract_camera_params(camera):
    """Extract K and an OpenCV-style 5-parameter distortion vector."""
    params = camera.params
    model_raw = str(camera.model)
    model = model_raw.split(".")[-1].upper() if "." in model_raw else model_raw.upper()

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params[:4]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = params[:5]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2, 0.0], dtype=np.float64)
    elif model == "FULL_OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = params[:9]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    else:
        K = camera.calibration_matrix().astype(np.float64)
        dist = np.zeros(5, dtype=np.float64)
        print(f"[SfM] Warning: unknown camera model {model_raw!r}; using K fallback.")

    return K, dist


def _camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -R.T @ np.asarray(t).reshape(3)


def _point_ids(rec) -> List[int]:
    ids = rec.point3D_ids
    if callable(ids):
        ids = ids()
    return list(ids)


def _get_cam_from_world(image):
    cfw = image.cam_from_world
    if callable(cfw):
        cfw = cfw()
    return cfw.rotation.matrix().astype(np.float64), cfw.translation.astype(np.float64).reshape(3, 1)


def _to_reference_centroid_world(
    points_colmap: np.ndarray,
    R_colmap: List[np.ndarray],
    t_colmap: List[np.ndarray],
    ref_idx: int,
):
    centroid = (
        points_colmap.mean(axis=0).astype(np.float64)
        if len(points_colmap) else np.zeros(3, dtype=np.float64)
    )
    R_ref = R_colmap[ref_idx].astype(np.float64)
    points_world = (R_ref @ (points_colmap - centroid).T).T if len(points_colmap) else points_colmap

    R_world, t_world = [], []
    for R, t in zip(R_colmap, t_colmap):
        t_vec = np.asarray(t).reshape(3)
        Rw = R @ R_ref.T
        tw = R @ centroid + t_vec
        R_world.append(Rw.astype(np.float64))
        t_world.append(tw.reshape(3, 1).astype(np.float64))
    return points_world.astype(np.float64), R_world, t_world, centroid, R_ref


def _extract_observations(
    rec,
    image_id_to_cam: Dict[int, int],
    valid_point_ids: np.ndarray,
    points_world: np.ndarray,
    R_list: List[np.ndarray],
    t_list: List[np.ndarray],
    max_reproj_error: float,
) -> Dict[str, np.ndarray]:
    pid_to_idx = {int(pid): i for i, pid in enumerate(valid_point_ids)}
    pid_to_xyz = {int(pid): points_world[i] for i, pid in enumerate(valid_point_ids)}

    point_indices, point_ids, cam_indices = [], [], []
    uv, depth, error = [], [], []
    visibility = np.zeros((len(valid_point_ids), len(R_list)), dtype=bool)

    for image_id, image in rec.images.items():
        if image_id not in image_id_to_cam:
            continue
        cam_idx = image_id_to_cam[image_id]
        R = R_list[cam_idx]
        t = t_list[cam_idx]

        for p2d in image.points2D:
            if not p2d.has_point3D():
                continue
            pid = int(p2d.point3D_id)
            if pid not in pid_to_idx:
                continue
            pt = rec.point3D(pid)
            if float(pt.error) > max_reproj_error:
                continue

            pidx = pid_to_idx[pid]
            xyz = pid_to_xyz[pid].reshape(3, 1)
            z = float((R @ xyz + t)[2, 0])
            if z <= 1e-8:
                continue

            point_indices.append(pidx)
            point_ids.append(pid)
            cam_indices.append(cam_idx)
            uv.append(np.asarray(p2d.xy, dtype=np.float64).reshape(2))
            depth.append(z)
            error.append(float(pt.error))
            visibility[pidx, cam_idx] = True

    track_lengths = visibility.sum(axis=1).astype(np.int32)
    return {
        "point_indices": np.asarray(point_indices, dtype=np.int64),
        "point_ids": np.asarray(point_ids, dtype=np.int64),
        "cam_indices": np.asarray(cam_indices, dtype=np.int32),
        "uv": np.asarray(uv, dtype=np.float64).reshape(-1, 2),
        "depth": np.asarray(depth, dtype=np.float64),
        "reproj_error": np.asarray(error, dtype=np.float64),
        "visibility": visibility,
        "track_lengths": track_lengths,
    }


def _save_ply(path: str, points: np.ndarray):
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")


def _save_data_products(
    out_dir: str,
    cam_names: List[str],
    image_names: List[str],
    image_paths: List[str],
    camera_models: List[str],
    K_list: List[np.ndarray],
    dist_list: List[np.ndarray],
    R_list: List[np.ndarray],
    t_list: List[np.ndarray],
    P_list: List[np.ndarray],
    points: np.ndarray,
    point_ids: np.ndarray,
    point_errors: np.ndarray,
    observations: Dict[str, np.ndarray],
    reference_camera: str,
    centroid_colmap: np.ndarray,
    R_ref_colmap: np.ndarray,
    num_registered: int,
):
    from scipy.io import savemat

    os.makedirs(out_dir, exist_ok=True)
    K = np.stack(K_list, axis=0)
    dist = np.stack(dist_list, axis=0)
    R = np.stack(R_list, axis=0)
    t = np.stack([x.reshape(3) for x in t_list], axis=0)
    P = np.stack(P_list, axis=0)
    centers = np.stack([_camera_center(R_list[i], t_list[i]) for i in range(len(cam_names))])

    np.savez(
        os.path.join(out_dir, "cameras.npz"),
        cam_names=np.array(cam_names),
        image_names=np.array(image_names),
        image_paths=np.array(image_paths),
        camera_models=np.array(camera_models),
        K=K,
        dist=dist,
        R=R,
        t=t,
        P=P,
        camera_centers_world=centers,
        reference_camera=np.array(reference_camera),
        sparse_centroid_colmap=centroid_colmap,
        reference_rotation_colmap=R_ref_colmap,
    )
    np.savez(
        os.path.join(out_dir, "sparse_points.npz"),
        points3D=points,
        point_ids=point_ids,
        reproj_error=point_errors,
        visibility=observations["visibility"],
        track_lengths=observations["track_lengths"],
    )
    np.savez(
        os.path.join(out_dir, "observations.npz"),
        cam_names=np.array(cam_names),
        point_indices=observations["point_indices"],
        point_ids=observations["point_ids"],
        cam_indices=observations["cam_indices"],
        uv=observations["uv"],
        depth=observations["depth"],
        reproj_error=observations["reproj_error"],
    )

    # Compatibility with existing loaders.
    savemat(
        os.path.join(out_dir, "cameras.mat"),
        {
            "num_cameras": len(cam_names),
            "K_list": K,
            "dist_list": dist,
            "cam_from_world_R": R,
            "cam_from_world_t": t,
            "P_list": P,
            "camera_centers_world": centers,
            "camera_models": np.array(camera_models, dtype=object),
            "cam_names": np.array(cam_names, dtype=object),
            "image_names": np.array(image_names, dtype=object),
            "reference_camera": reference_camera,
            "sparse_centroid_colmap": centroid_colmap,
            "reference_rotation_colmap": R_ref_colmap,
            "num_registered_images": num_registered,
        },
    )
    savemat(
        os.path.join(out_dir, "points3D.mat"),
        {
            "points3D": points,
            "point_ids": point_ids,
            "reproj_error": point_errors,
            "visibility": observations["visibility"].astype(np.uint8),
            "track_lengths": observations["track_lengths"],
            "num_points": len(points),
        },
    )

    data = {
        "coordinate_system": {
            "origin": "centroid of retained sparse COLMAP points",
            "axes": f"parallel to {reference_camera} camera axes",
            "note": "COLMAP scale is still arbitrary until an external scale constraint is applied.",
            "sparse_centroid_colmap": centroid_colmap.tolist(),
        },
        "num_cameras": len(cam_names),
        "num_registered_images": int(num_registered),
        "cameras": [],
    }
    for i, name in enumerate(cam_names):
        data["cameras"].append(
            {
                "index": i,
                "name": name,
                "image_name": image_names[i],
                "image_path": image_paths[i],
                "model": camera_models[i],
                "K": K[i].tolist(),
                "distortion": dist[i].tolist(),
                "cam_from_world_R": R[i].tolist(),
                "cam_from_world_t": t[i].tolist(),
                "world_position": centers[i].tolist(),
                "projection_matrix": P[i].tolist(),
            }
        )
    with open(os.path.join(out_dir, "cameras.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    lines = [
        "SfM camera parameters",
        f"Output frame: origin=sparse centroid, axes={reference_camera}",
        f"Registered images: {num_registered}",
        f"Sparse points: {len(points)}",
        f"Track observations: {len(observations['uv'])}",
        "",
    ]
    for i, name in enumerate(cam_names):
        C = centers[i]
        lines.extend(
            [
                f"{i:02d} {name}",
                f"  image: {image_names[i]}",
                f"  model: {camera_models[i]}",
                f"  f_px: ({K[i, 0, 0]:.9g}, {K[i, 1, 1]:.9g})",
                f"  principal_point_px: ({K[i, 0, 2]:.9g}, {K[i, 1, 2]:.9g})",
                f"  world_position: ({C[0]:.9g}, {C[1]:.9g}, {C[2]:.9g})",
                "",
            ]
        )
    with open(os.path.join(out_dir, "cameras.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    _save_ply(os.path.join(out_dir, "sparse_points.ply"), points)
    _save_ply(os.path.join(out_dir, "camera_centers.ply"), centers)


def _axis_limits(points: np.ndarray, centers: np.ndarray):
    all_pts = points if len(points) else centers
    all_pts = np.vstack([all_pts, centers])
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    mid = 0.5 * (lo + hi)
    span = max(float(np.max(hi - lo)), 1.0)
    half = 0.55 * span
    return np.stack([mid - half, mid + half], axis=0)


def _apply_limits(ax, lim):
    ax.set_xlim(lim[0, 0], lim[1, 0])
    ax.set_ylim(lim[0, 1], lim[1, 1])
    ax.set_zlim(lim[0, 2], lim[1, 2])


def _visualize_scene(out_dir, cam_names, R_list, t_list, points, dpi):
    import matplotlib.pyplot as plt

    centers = np.stack([_camera_center(R_list[i], t_list[i]) for i in range(len(cam_names))])
    lim = _axis_limits(points, centers)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    if len(points):
        n = min(len(points), 30000)
        idx = np.random.RandomState(0).choice(len(points), n, replace=False)
        pts = points[idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.0, c="0.25", alpha=0.55)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="red", s=45, marker="^")
    scale = 0.08 * max(lim[1] - lim[0])
    for i, C in enumerate(centers):
        ax.text(C[0], C[1], C[2], cam_names[i], fontsize=7)
        axes = scale * R_list[i].T
        for axis, color in zip(axes.T, ["r", "g", "b"]):
            ax.quiver(C[0], C[1], C[2], axis[0], axis[1], axis[2], color=color, linewidth=1)
    _apply_limits(ax, lim)
    ax.set_title("Sparse points and camera poses")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sparse_scene.png"), dpi=dpi)
    plt.close(fig)
    return lim


def _visualize_observations_3d(out_dir, cam_names, points, observations, lim, dpi):
    import matplotlib.pyplot as plt

    n_cam = len(cam_names)
    fig = plt.figure(figsize=(16, 12))
    for i, name in enumerate(cam_names):
        ax = fig.add_subplot(3, 4, i + 1, projection="3d")
        mask = observations["cam_indices"] == i
        pidx = observations["point_indices"][mask]
        pts = points[pidx] if len(pidx) else np.zeros((0, 3))
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c="tab:red", alpha=0.75)
        _apply_limits(ax, lim)
        ax.set_title(f"{name}: {len(pts)} obs", fontsize=9)
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "camera_observations_3d.png"), dpi=dpi)
    plt.close(fig)


def _visualize_observations_2d(out_dir, cam_names, image_paths, observations, dpi):
    import cv2
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.ravel()
    for i, name in enumerate(cam_names):
        ax = axes[i]
        img = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            ax.imshow(img, cmap="gray")
        mask = observations["cam_indices"] == i
        uv = observations["uv"][mask]
        if len(uv):
            ax.scatter(uv[:, 0], uv[:, 1], s=18, c="red", marker="o", linewidths=0)
        ax.set_title(f"{name}: {len(uv)} obs", fontsize=10)
        ax.set_xlim(0, img.shape[1] if img is not None else 1)
        ax.set_ylim(img.shape[0] if img is not None else 1, 0)
        ax.axis("off")
    for j in range(len(cam_names), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "camera_observations_2d.png"), dpi=dpi)
    plt.close(fig)


def _visualize_products(
    out_dir: str,
    cam_names: List[str],
    image_paths: List[str],
    R_list: List[np.ndarray],
    t_list: List[np.ndarray],
    points: np.ndarray,
    observations: Dict[str, np.ndarray],
    dpi: int,
):
    try:
        lim = _visualize_scene(out_dir, cam_names, R_list, t_list, points, dpi)
        _visualize_observations_3d(out_dir, cam_names, points, observations, lim, dpi)
        _visualize_observations_2d(out_dir, cam_names, image_paths, observations, dpi)
    except ImportError as exc:
        print(f"[SfM] Visualization skipped, missing dependency: {exc}")


def run_reference_sfm(
    data_dir: str,
    image_dir: str = "images",
    ref_mode: str = "first",
    ref_name: str = "001",
    output_dir: Optional[str] = None,
    reference_camera: str = "cam_0",
    max_features: int = 8192,
    first_octave: int = 0,
    cross_check: bool = False,
    min_num_matches: int = 8,
    min_model_size: int = 3,
    ba_global_max_refinements: int = 5,
    clean: bool = False,
    max_reproj_error: float = 4.0,
    dpi: int = 180,
) -> SfMProducts:
    """Run self-calibration and export the SfM dataset requested for CylinderDIC."""
    import cv2
    import pycolmap

    out_dir = output_dir or os.path.join(data_dir, "result", "sfm")
    os.makedirs(out_dir, exist_ok=True)
    if clean:
        for name in ["colmap.db", "colmap_sfm", "colmap_images"]:
            path = os.path.join(out_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)

    ref_images = _collect_reference_images(data_dir, image_dir, ref_mode, ref_name)
    cam_names = sorted(ref_images.keys(), key=_natural_key)
    if reference_camera not in cam_names:
        raise ValueError(f"Reference camera {reference_camera!r} not in {cam_names}")
    ref_idx = cam_names.index(reference_camera)
    image_paths = [ref_images[name] for name in cam_names]

    flat_image_dir = os.path.join(out_dir, "colmap_images")
    os.makedirs(flat_image_dir, exist_ok=True)
    for name in cam_names:
        src = ref_images[name]
        dst = os.path.join(flat_image_dir, f"{name}_{os.path.basename(src)}")
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {src}")
        cv2.imwrite(dst, img)

    database_path = os.path.join(out_dir, "colmap.db")
    sfm_path = os.path.join(out_dir, "colmap_sfm")
    os.makedirs(sfm_path, exist_ok=True)

    print(f"[SfM] Extracting SIFT from {len(cam_names)} reference images")
    pycolmap.set_random_seed(0)
    pycolmap.extract_features(
        database_path,
        flat_image_dir,
        extraction_options={"sift": {"max_num_features": max_features, "first_octave": first_octave}},
    )
    print("[SfM] Exhaustive matching")
    pycolmap.match_exhaustive(database_path, matching_options={"sift": {"cross_check": cross_check}})
    print("[SfM] Incremental mapping")
    reconstructions = pycolmap.incremental_mapping(
        database_path,
        flat_image_dir,
        sfm_path,
        options={
            "ba_global_max_refinements": ba_global_max_refinements,
            "min_num_matches": min_num_matches,
            "multiple_models": True,
            "min_model_size": min(min_model_size, len(cam_names)),
            "min_focal_length_ratio": 0.1,
            "max_focal_length_ratio": 10.0,
        },
    )
    if isinstance(reconstructions, dict):
        recs = [r for r in reconstructions.values() if hasattr(r, "num_reg_images")]
    elif isinstance(reconstructions, list):
        recs = [r for r in reconstructions if hasattr(r, "num_reg_images")]
    else:
        recs = []
    if not recs:
        raise RuntimeError("COLMAP SfM returned no valid reconstruction.")
    rec = max(recs, key=lambda r: r.num_reg_images())
    print(f"[SfM] Registered {rec.num_reg_images()}/{len(cam_names)} images, {rec.num_points3D()} points")

    cam_image_ids = {name: [] for name in cam_names}
    image_id_to_cam = {}
    for image_id, image in rec.images.items():
        for i, name in enumerate(cam_names):
            if image.name.startswith(f"{name}_"):
                cam_image_ids[name].append(image_id)
                image_id_to_cam[image_id] = i
                break

    K_list, dist_list, R_colmap, t_colmap, models, image_names = [], [], [], [], [], [""] * len(cam_names)
    for i, name in enumerate(cam_names):
        ids = cam_image_ids[name]
        if not ids:
            raise RuntimeError(f"Camera {name} was not registered by COLMAP.")
        image = rec.images[ids[0]]
        image_names[i] = image.name
        camera = rec.cameras[image.camera_id]
        K, dist = _extract_camera_params(camera)
        R, t = _get_cam_from_world(image)
        K_list.append(K)
        dist_list.append(dist)
        R_colmap.append(R)
        t_colmap.append(t)
        models.append(str(camera.model).split(".")[-1])

    ids, pts, errs = [], [], []
    for pid in _point_ids(rec):
        pt = rec.point3D(pid)
        if float(pt.error) <= max_reproj_error:
            ids.append(int(pid))
            pts.append(np.asarray(pt.xyz, dtype=np.float64))
            errs.append(float(pt.error))
    point_ids = np.asarray(ids, dtype=np.int64)
    point_errors = np.asarray(errs, dtype=np.float64)
    points_colmap = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    points, R_list, t_list, centroid, R_ref = _to_reference_centroid_world(
        points_colmap, R_colmap, t_colmap, ref_idx
    )
    P_list = [K @ np.hstack((R, t.reshape(3, 1))) for K, R, t in zip(K_list, R_list, t_list)]
    observations = _extract_observations(
        rec,
        image_id_to_cam,
        point_ids,
        points,
        R_list,
        t_list,
        max_reproj_error,
    )

    _save_data_products(
        out_dir,
        cam_names,
        image_names,
        image_paths,
        models,
        K_list,
        dist_list,
        R_list,
        t_list,
        P_list,
        points,
        point_ids,
        point_errors,
        observations,
        reference_camera,
        centroid,
        R_ref,
        rec.num_reg_images(),
    )
    _visualize_products(out_dir, cam_names, image_paths, R_list, t_list, points, observations, dpi)

    print(f"[SfM] Saved products to {out_dir}")
    print(f"[SfM] Sparse points: {len(points)}, observations: {len(observations['uv'])}")
    return SfMProducts(out_dir, cam_names, K_list, dist_list, R_list, t_list, P_list, points, observations)


def reference_sfm_exists(data_dir: str, output_dir: Optional[str] = None) -> bool:
    out_dir = output_dir or os.path.join(data_dir, "result", "sfm")
    required = ["cameras.mat", "points3D.mat", "cameras.npz", "sparse_points.npz", "observations.npz"]
    return all(os.path.exists(os.path.join(out_dir, name)) for name in required)


def load_observations(sfm_dir: str, filename: str = "observations.npz") -> Optional[Dict[str, Dict]]:
    """Load track observations as a per-camera compatibility dictionary."""
    obs_path = os.path.join(sfm_dir, filename)
    cam_path = os.path.join(sfm_dir, "cameras.npz")
    pts_path = os.path.join(sfm_dir, "sparse_points.npz")
    if not (os.path.exists(obs_path) and os.path.exists(cam_path) and os.path.exists(pts_path)):
        return None

    obs = np.load(obs_path, allow_pickle=False)
    cam = np.load(cam_path, allow_pickle=False)
    pts = np.load(pts_path, allow_pickle=False)

    cam_names = [str(x) for x in cam["cam_names"]]
    points = pts["points3D"]
    result: Dict[str, Dict] = {}
    for cam_idx, name in enumerate(cam_names):
        mask = obs["cam_indices"] == cam_idx
        pidx = obs["point_indices"][mask]
        result[name] = {
            "uv": obs["uv"][mask].astype(np.float32),
            "depth": obs["depth"][mask].astype(np.float32),
            "xyz": points[pidx].astype(np.float32),
            "error": obs["reproj_error"][mask].astype(np.float32),
            "K": cam["K"][cam_idx].astype(np.float32),
            "R": cam["R"][cam_idx].astype(np.float32),
            "t": cam["t"][cam_idx].astype(np.float32),
        }
    return result
