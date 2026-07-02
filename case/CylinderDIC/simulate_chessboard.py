"""Simulate calibration chessboard images for the CylinderDIC camera array.

This script reuses ``CylinderSimConfig`` and ``build_camera_array`` from
``simulate_cylinder.py`` so the rendered calibration images share the same
camera intrinsics, extrinsics, working distance, and object-scale convention as
the synthetic DIC data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from typing import Dict, Tuple

import cv2
import imageio.v3 as iio
import numpy as np

from simulate_cylinder import CylinderSimConfig, build_camera_array


def _project_points(
    points: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    dist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project world points to pixels using the same pinhole model as the simulator."""
    p_cam = R @ points.T + t.reshape(3, 1)
    z = p_cam[2]
    valid_z = z > 1e-9

    xn = p_cam[0] / z
    yn = p_cam[1] / z

    k1, k2 = float(dist[0]), float(dist[1])
    if abs(k1) > 1e-12 or abs(k2) > 1e-12:
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        xn = xn * radial
        yn = yn * radial

    uv = np.stack(
        [
            K[0, 0] * xn + K[0, 1] * yn + K[0, 2],
            K[1, 1] * yn + K[1, 2],
        ],
        axis=1,
    )
    return uv, valid_z


def _fixed_board_pose(normal: np.ndarray) -> Dict[str, np.ndarray]:
    """Place one fixed board at the world origin with the requested normal."""
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    # Default board normal is +X. The local board X axis is chosen along -Z
    # and local board Y along +Y, giving cross(x_axis, y_axis) = +X.
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(normal, world_up))) > 0.95:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    x_axis = np.cross(world_up, normal)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    return {
        "center": np.zeros(3, dtype=np.float64),
        "x_axis": x_axis,
        "y_axis": y_axis,
        "normal": normal,
    }


def _make_board_points(
    pose: Dict[str, np.ndarray],
    cols: int,
    rows: int,
    square_size: float,
) -> np.ndarray:
    """Return grid vertices with shape ``(rows + 1, cols + 1, 3)``."""
    width = cols * square_size
    height = rows * square_size
    xs = np.linspace(-0.5 * width, 0.5 * width, cols + 1)
    ys = np.linspace(-0.5 * height, 0.5 * height, rows + 1)

    pts = np.empty((rows + 1, cols + 1, 3), dtype=np.float64)
    center = pose["center"]
    x_axis = pose["x_axis"]
    y_axis = pose["y_axis"]
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            pts[iy, ix] = center + x * x_axis + y * y_axis
    return pts


def _make_inner_corners(
    pose: Dict[str, np.ndarray],
    cols: int,
    rows: int,
    square_size: float,
) -> np.ndarray:
    """Return internal chessboard corners in board row-major order."""
    width = cols * square_size
    height = rows * square_size
    xs = -0.5 * width + square_size * np.arange(1, cols)
    ys = -0.5 * height + square_size * np.arange(1, rows)

    corners = []
    for y in ys:
        for x in xs:
            corners.append(pose["center"] + x * pose["x_axis"] + y * pose["y_axis"])
    return np.asarray(corners, dtype=np.float64)


def _render_chessboard(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    dist: np.ndarray,
    cam_center: np.ndarray,
    config: CylinderSimConfig,
    pose: Dict[str, np.ndarray],
    cols: int,
    rows: int,
    square_size: float,
    blur_sigma: float,
    front_face_threshold: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Render a fixed chessboard from one camera."""
    grid = _make_board_points(pose, cols, rows, square_size)
    inner = _make_inner_corners(pose, cols, rows, square_size)

    h, w = config.image_height, config.image_width
    img = np.full((h, w), 230, dtype=np.uint8)

    board_to_camera = cam_center.astype(np.float64) - pose["center"]
    board_to_camera = board_to_camera / np.linalg.norm(board_to_camera)
    view_cos = float(np.dot(pose["normal"], board_to_camera))
    front_facing = view_cos > front_face_threshold

    visible_squares = 0
    if front_facing:
        for iy in range(rows):
            for ix in range(cols):
                poly3d = np.stack(
                    [
                        grid[iy, ix],
                        grid[iy, ix + 1],
                        grid[iy + 1, ix + 1],
                        grid[iy + 1, ix],
                    ],
                    axis=0,
                )
                uv, valid_z = _project_points(poly3d, K, R, t, dist)
                if not np.all(valid_z):
                    continue

                poly = np.round(uv).astype(np.int32)
                if (
                    np.max(poly[:, 0]) < 0
                    or np.min(poly[:, 0]) >= w
                    or np.max(poly[:, 1]) < 0
                    or np.min(poly[:, 1]) >= h
                ):
                    continue

                color = 35 if (ix + iy) % 2 == 0 else 245
                cv2.fillConvexPoly(img, poly, color, lineType=cv2.LINE_AA)
                visible_squares += 1

    # Draw a thin physical board border so the plate extent is explicit.
    outer3d = np.stack([grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0]], axis=0)
    outer_uv, outer_valid = _project_points(outer3d, K, R, t, dist)
    if front_facing and np.all(outer_valid):
        cv2.polylines(
            img,
            [np.round(outer_uv).astype(np.int32)],
            isClosed=True,
            color=15,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    if blur_sigma > 0:
        ksize = max(3, int(np.ceil(blur_sigma * 6)) | 1)
        img = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)

    inner_uv, inner_valid_z = _project_points(inner, K, R, t, dist)
    visible_inner = (
        front_facing
        & inner_valid_z
        & (inner_uv[:, 0] >= 0)
        & (inner_uv[:, 0] < w)
        & (inner_uv[:, 1] >= 0)
        & (inner_uv[:, 1] < h)
    )

    meta = {
        "board_center_world": pose["center"].tolist(),
        "board_x_axis_world": pose["x_axis"].tolist(),
        "board_y_axis_world": pose["y_axis"].tolist(),
        "board_normal_world": pose["normal"].tolist(),
        "front_facing": bool(front_facing),
        "front_view_cos": view_cos,
        "visible_squares": int(visible_squares),
        "visible_inner_corners": int(np.count_nonzero(visible_inner)),
        "inner_corners_world": inner.tolist(),
        "inner_corners_uv": inner_uv.tolist(),
    }
    return img, meta


def simulate_chessboard_images(
    config: CylinderSimConfig,
    output_dir: str,
    cols: int,
    rows: int,
    square_size: float,
    blur_sigma: float,
    board_normal: np.ndarray,
    front_face_threshold: float,
    clean: bool = True,
) -> None:
    """Render one fixed chessboard image for each simulator camera."""
    K_list, R_list, t_list, dist_list, cam_centers = build_camera_array(config)
    pose = _fixed_board_pose(board_normal)

    if clean and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    meta = {
        "config": asdict(config),
        "board": {
            "square_size_mm": square_size,
            "cols_squares": cols,
            "rows_squares": rows,
            "inner_corners_cols": cols - 1,
            "inner_corners_rows": rows - 1,
            "placement": "one fixed board centered at the world origin",
            "board_center_world": pose["center"].tolist(),
            "board_x_axis_world": pose["x_axis"].tolist(),
            "board_y_axis_world": pose["y_axis"].tolist(),
            "board_normal_world": pose["normal"].tolist(),
            "front_face_threshold": front_face_threshold,
        },
        "cameras": [],
    }

    print("=" * 60)
    print("CylinderDIC Chessboard Calibration Simulation")
    print("=" * 60)
    print(f"  Cameras:       {config.num_cameras}")
    print(f"  Image:         {config.image_width} x {config.image_height}")
    print(f"  Board:         {cols} x {rows} squares, {square_size:g} mm/square")
    print(f"  Board normal:  {pose['normal'].tolist()}")
    print(f"  Output:        {output_dir}")
    print()

    for cam_id in range(config.num_cameras):
        img, cam_meta = _render_chessboard(
            K_list[cam_id],
            R_list[cam_id],
            t_list[cam_id],
            dist_list[cam_id],
            cam_centers[cam_id],
            config,
            pose,
            cols,
            rows,
            square_size,
            blur_sigma,
            front_face_threshold,
        )

        cam_dir = os.path.join(output_dir, f"cam_{cam_id}")
        os.makedirs(cam_dir, exist_ok=True)
        image_path = os.path.join(cam_dir, "001.bmp")
        iio.imwrite(image_path, img)

        meta["cameras"].append(
            {
                "camera_id": cam_id,
                "camera_name": f"cam_{cam_id}",
                "image_path": image_path,
                "K": K_list[cam_id].tolist(),
                "R_world_to_camera": R_list[cam_id].tolist(),
                "t_world_to_camera": t_list[cam_id].reshape(3).tolist(),
                "distortion": dist_list[cam_id].tolist(),
                "camera_center_world": cam_centers[cam_id].tolist(),
                **cam_meta,
            }
        )

        print(
            f"  cam_{cam_id}: front={cam_meta['front_facing']} "
            f"cos={cam_meta['front_view_cos']:.3f}, visible inner corners "
            f"{cam_meta['visible_inner_corners']}/{(cols - 1) * (rows - 1)}"
        )

    meta_path = os.path.join(output_dir, "chessboard_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print()
    print("[Done]")
    print(f"  Images: {output_dir}/cam_*/001.bmp")
    print(f"  Meta:   {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render chessboard calibration images for CylinderDIC cameras."
    )
    parser.add_argument("--output_dir", type=str, default="case/CylinderDIC")
    parser.add_argument("--calibrate_dir", type=str, default="")
    parser.add_argument("--num_cameras", type=int, default=12)
    parser.add_argument("--working_distance", type=float, default=400.0)
    parser.add_argument("--cylinder_radius", type=float, default=80.0)
    parser.add_argument("--cylinder_height", type=float, default=120.0)
    parser.add_argument("--focal_length", type=float, default=8.0)
    parser.add_argument("--image_width", type=int, default=1440)
    parser.add_argument("--image_height", type=int, default=1080)
    parser.add_argument("--pixel_size", type=float, default=3.45e-3)
    parser.add_argument("--k1", type=float, default=0.0)
    parser.add_argument("--k2", type=float, default=0.0)
    parser.add_argument("--cols", type=int, default=10, help="Number of chessboard squares along board X.")
    parser.add_argument("--rows", type=int, default=8, help="Number of chessboard squares along board Y.")
    parser.add_argument("--square_size", type=float, default=10.0, help="Chessboard square size in mm.")
    parser.add_argument("--blur_sigma", type=float, default=0.35)
    parser.add_argument(
        "--board_normal",
        type=float,
        nargs=3,
        default=(1.0, 0.0, 0.0),
        help="Fixed board normal in world coordinates. Default: +X.",
    )
    parser.add_argument(
        "--front_face_threshold",
        type=float,
        default=0.05,
        help="Minimum dot(board_normal, board_to_camera) for the chessboard face to be visible.",
    )
    parser.add_argument("--no_clean", action="store_true", help="Do not delete existing calibration images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CylinderSimConfig(
        output_dir=args.output_dir,
        num_cameras=args.num_cameras,
        working_distance=args.working_distance,
        cylinder_radius=args.cylinder_radius,
        cylinder_height=args.cylinder_height,
        focal_length=args.focal_length,
        image_width=args.image_width,
        image_height=args.image_height,
        pixel_size=args.pixel_size,
        k1=args.k1,
        k2=args.k2,
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_root = args.calibrate_dir
    if not out_root:
        out_root = os.path.join(script_dir, "calibrate_images")
    elif not os.path.isabs(out_root):
        out_root = os.path.join(script_dir, out_root)

    simulate_chessboard_images(
        config=config,
        output_dir=out_root,
        cols=args.cols,
        rows=args.rows,
        square_size=args.square_size,
        blur_sigma=args.blur_sigma,
        board_normal=np.asarray(args.board_normal, dtype=np.float64),
        front_face_threshold=args.front_face_threshold,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    main()
