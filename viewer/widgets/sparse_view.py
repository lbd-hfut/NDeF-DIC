"""
Sparse3DView: 3D visualization of sparse SfM points and camera poses.

Displays:
  - Sparse 3D points (white spheres)
  - Camera wireframe frustums
  - Camera orientation axes (RGB = XYZ)
  - World coordinate axes
"""

import numpy as np
from typing import Optional, List, Dict

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

import pyvista as pv
from pyvistaqt import QtInteractor


class Sparse3DView(QWidget):
    """Tab: Sparse reconstruction + camera poses in 3D."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._camera_actors = []  # track camera-related actors for cleanup

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # PyVista Qt interactor (the 3D viewport)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        # Configure initial scene
        self.plotter.set_background("#1a1a2e")
        self.plotter.add_axes(
            xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)',
            line_width=1, color='white',
        )
        self.plotter.show_grid(color='gray', all_edges=True, opacity=0.15)
        self._show_placeholder("No data loaded. Use Load button to load a data directory.")

    # =================================================================
    # Public API
    # =================================================================

    def set_data(
        self,
        sparse_points: Optional[np.ndarray],
        cameras: Optional[List[Dict]],
    ) -> None:
        """Load and display sparse points and camera poses.

        Args:
            sparse_points: (M, 3) float32 array, or None
            cameras: list of dicts with keys K, R, t, id, model; or None
        """
        self.clear()

        if sparse_points is None and cameras is None:
            self._show_placeholder("No sparse data available.\nRun Step 1 SfM first.")
            return

        # Render sparse points
        if sparse_points is not None and len(sparse_points) > 0:
            cloud = pv.PolyData(sparse_points)
            self.plotter.add_mesh(
                cloud, color='#e0e0e0', point_size=4,
                render_points_as_spheres=True, label='Sparse Points',
            )

        # Render cameras
        if cameras is not None and len(cameras) > 0:
            self._add_cameras(cameras, sparse_points)

        self.plotter.view_isometric()
        self.plotter.render()

        n_pts = len(sparse_points) if sparse_points is not None else 0
        n_cam = len(cameras) if cameras is not None else 0
        print(f"[SparseView] Displayed {n_pts} points, {n_cam} cameras")

    def clear(self) -> None:
        """Remove all actors and reset view."""
        self.plotter.clear()
        self.plotter.set_background("#1a1a2e")
        self.plotter.add_axes(
            xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)',
            line_width=1, color='white',
        )
        self.plotter.show_grid(color='gray', all_edges=True, opacity=0.15)
        self._camera_actors = []

    def reset_camera(self) -> None:
        """Reset camera to see all data."""
        self.plotter.view_isometric()
        self.plotter.render()

    # =================================================================
    # Camera rendering
    # =================================================================

    def _add_cameras(self, cameras: List[Dict], points: Optional[np.ndarray]):
        """Add camera frustums and axes for all cameras."""
        # Estimate frustum scale from point cloud extent
        if points is not None and len(points) > 0:
            extent = np.ptp(points, axis=0)
            frustum_scale = np.mean(extent) * 0.12
            axis_length = frustum_scale * 0.6
        else:
            # Estimate from camera positions
            centers = np.array([(-c["R"].T @ c["t"].reshape(3, 1)).ravel()
                               for c in cameras])
            extent = np.ptp(centers, axis=0)
            frustum_scale = np.mean(extent) * 0.15 if np.mean(extent) > 1e-6 else 10.0
            axis_length = frustum_scale * 0.6

        for cam in cameras:
            self._add_camera_frustum(cam, frustum_scale)
            self._add_camera_axes(cam, axis_length)
            self._add_camera_label(cam, frustum_scale)

    def _add_camera_frustum(self, cam: Dict, scale: float):
        """Build and add a wireframe frustum for one camera."""
        K = cam["K"]
        R = cam["R"]
        t = cam["t"].reshape(3, 1)

        # Camera center in world
        C = (-R.T @ t).ravel()

        # Image plane corners at distance `scale`
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        hw = scale * cx / fx
        hh = scale * cy / fy

        corners_cam = np.array([
            [-hw, -hh, scale],
            [ hw, -hh, scale],
            [ hw,  hh, scale],
            [-hw,  hh, scale],
        ], dtype=np.float32)

        # Transform corners to world
        R_c2w = R.T
        corners_world = (R_c2w @ corners_cam.T).T + C

        # Build lines: center → corners + image plane edges
        all_pts = np.vstack([C.reshape(1, 3), corners_world])

        lines_data = []
        # Rays from center to corners
        for i in range(4):
            lines_data.extend([2, 0, i + 1])
        # Image plane edges
        for i in range(4):
            lines_data.extend([2, i + 1, (i + 1) % 4 + 1])

        lines = np.array(lines_data, dtype=np.int32)

        frustum = pv.PolyData()
        frustum.points = all_pts.astype(np.float32)
        frustum.lines = lines

        self.plotter.add_mesh(
            frustum, color='#00bcd4', line_width=1.5, style='wireframe',
            label=f'Cam {cam["id"]}',
        )

    def _add_camera_axes(self, cam: Dict, length: float):
        """Add RGB orientation arrows at camera center."""
        R = cam["R"]
        t = cam["t"].reshape(3, 1)
        C = (-R.T @ t).ravel()
        R_c2w = R.T

        colors = ['#ff4444', '#44ff44', '#4488ff']  # R, G, B
        for axis in range(3):
            direction = R_c2w[:, axis] * length
            self.plotter.add_arrows(
                cent=C, direction=direction, mag=1.0,
                color=colors[axis],
            )

    def _add_camera_label(self, cam: Dict, frustum_scale: float):
        """Add a text label near the camera center."""
        R = cam["R"]
        t = cam["t"].reshape(3, 1)
        C = (-R.T @ t).ravel()
        label_pos = C + np.array([0, 0, frustum_scale * 0.3])
        self.plotter.add_point_labels(
            np.array([label_pos]), [f"Cam {cam['id']}"],
            font_size=10, text_color='white',
            point_size=1, shape_opacity=0.0,
        )

    # =================================================================
    # Helpers
    # =================================================================

    def _show_placeholder(self, message: str):
        """Show placeholder text in the 3D view."""
        self.plotter.add_text(
            message,
            position='upper_left', font_size=12, color='gray',
        )
        self.plotter.render()
