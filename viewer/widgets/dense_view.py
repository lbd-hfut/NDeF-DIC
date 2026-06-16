"""
Dense3DView: 3D visualization of dense point cloud.

Displays:
  - Dense point cloud colored by visibility count or normal direction
  - QComboBox to switch color mode
  - Scalar bar
"""

import numpy as np
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
)
from PySide6.QtCore import Qt

import pyvista as pv
from pyvistaqt import QtInteractor


# Subsample target for rendering performance
RENDER_MAX_POINTS = 100_000


class Dense3DView(QWidget):
    """Tab: Dense point cloud visualization."""

    COLOR_VISIBILITY = "visibility"
    COLOR_NORMAL_X = "normal_x"
    COLOR_NORMAL_Y = "normal_y"
    COLOR_NORMAL_Z = "normal_z"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cloud_actor = None
        self._render_points = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Color mode selector
        control_bar = QHBoxLayout()
        control_bar.addWidget(QLabel("Color by:"))
        self.color_combo = QComboBox()
        self.color_combo.addItem("Visible Cameras", self.COLOR_VISIBILITY)
        self.color_combo.addItem("Normal X", self.COLOR_NORMAL_X)
        self.color_combo.addItem("Normal Y", self.COLOR_NORMAL_Y)
        self.color_combo.addItem("Normal Z", self.COLOR_NORMAL_Z)
        self.color_combo.currentIndexChanged.connect(self._on_color_mode_changed)
        control_bar.addWidget(self.color_combo)
        control_bar.addStretch()

        self.point_count_label = QLabel("")
        control_bar.addWidget(self.point_count_label)

        layout.addLayout(control_bar)

        # PyVista 3D viewport
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        self.plotter.set_background("#1a1a2e")
        self.plotter.add_axes(
            xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)',
            line_width=1, color='white',
        )
        self.plotter.show_grid(color='gray', all_edges=True)
        self._show_placeholder("No dense data. Run Step 1 dense MVS first.")

    # =================================================================
    # Public API
    # =================================================================

    def set_data(
        self,
        dense_points: Optional[np.ndarray],
        dense_normals: Optional[np.ndarray],
        dense_vis_mask: Optional[np.ndarray],
    ) -> None:
        """Load and display dense point cloud.

        Args:
            dense_points: (K, 3) float32, or None
            dense_normals: (K, 3) float32, or None
            dense_vis_mask: (K, N_cam) bool, or None
        """
        self.clear()

        if dense_points is None or len(dense_points) == 0:
            self._show_placeholder("No dense data available.\nRun Step 1 dense MVS first.")
            return

        # Subsample for rendering
        n = len(dense_points)
        if n > RENDER_MAX_POINTS:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, RENDER_MAX_POINTS, replace=False)
            idx.sort()
            self._render_points = dense_points[idx]
            self._render_normals = (
                dense_normals[idx] if dense_normals is not None else None
            )
            self._render_vis = (
                dense_vis_mask[idx] if dense_vis_mask is not None else None
            )
            self.point_count_label.setText(
                f"Showing {RENDER_MAX_POINTS:,} / {n:,} points"
            )
        else:
            self._render_points = dense_points
            self._render_normals = dense_normals
            self._render_vis = dense_vis_mask
            self.point_count_label.setText(f"{n:,} points")

        # Compute visibility counts
        if self._render_vis is not None:
            vis_counts = self._render_vis.sum(axis=1).astype(np.float32)
        else:
            vis_counts = np.zeros(len(self._render_points), dtype=np.float32)

        # Build PolyData
        cloud = pv.PolyData(self._render_points)

        # Attach scalar data
        cloud.point_data['visibility'] = vis_counts
        if self._render_normals is not None:
            cloud.point_data['normal_x'] = self._render_normals[:, 0]
            cloud.point_data['normal_y'] = self._render_normals[:, 1]
            cloud.point_data['normal_z'] = self._render_normals[:, 2]

        # Render with default color mode
        self._current_mode = self.color_combo.currentData()
        self._render_cloud(cloud)

        self.plotter.view_isometric()
        self.plotter.render()

        print(f"[DenseView] Displayed {len(self._render_points)} points")

    def clear(self) -> None:
        """Remove all actors."""
        self.plotter.clear()
        self.plotter.set_background("#1a1a2e")
        self.plotter.add_axes(
            xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)',
            line_width=1, color='white',
        )
        self.plotter.show_grid(color='gray', all_edges=True)
        self._cloud_actor = None
        self._render_points = None

    def reset_camera(self) -> None:
        """Reset camera view."""
        self.plotter.view_isometric()
        self.plotter.render()

    # =================================================================
    # Internal
    # =================================================================

    def _render_cloud(self, cloud: pv.PolyData):
        """Add or replace the cloud actor with current color mode."""
        mode = self._current_mode or self.COLOR_VISIBILITY

        cmap = 'plasma' if mode == self.COLOR_VISIBILITY else 'RdBu_r'

        # Remove previous actor
        if self._cloud_actor is not None:
            self.plotter.remove_actor(self._cloud_actor)

        # Remove previous scalar bar
        if hasattr(self, '_scalar_bar_added') and self._scalar_bar_added:
            self.plotter.remove_scalar_bar()

        self._cloud_actor = self.plotter.add_mesh(
            cloud,
            scalars=mode,
            cmap=cmap,
            point_size=3,
            render_points_as_spheres=True,
            scalar_bar_args={
                'title': self._scalar_bar_title(mode),
                'vertical': True,
                'position_x': 0.88,
                'position_y': 0.05,
                'width': 0.05,
                'height': 0.4,
            },
        )
        self._scalar_bar_added = True

    def _scalar_bar_title(self, mode: str) -> str:
        titles = {
            self.COLOR_VISIBILITY: "Visible\nCameras",
            self.COLOR_NORMAL_X: "Normal X",
            self.COLOR_NORMAL_Y: "Normal Y",
            self.COLOR_NORMAL_Z: "Normal Z",
        }
        return titles.get(mode, mode)

    # =================================================================
    # Slots
    # =================================================================

    def _on_color_mode_changed(self, index: int):
        """Handle color mode combo change."""
        if self._render_points is None:
            return

        self._current_mode = self.color_combo.currentData()

        # Rebuild cloud with new scalars
        cloud = pv.PolyData(self._render_points)
        if self._render_vis is not None:
            cloud.point_data['visibility'] = self._render_vis.sum(axis=1).astype(np.float32)
        if self._render_normals is not None:
            cloud.point_data['normal_x'] = self._render_normals[:, 0]
            cloud.point_data['normal_y'] = self._render_normals[:, 1]
            cloud.point_data['normal_z'] = self._render_normals[:, 2]

        self._render_cloud(cloud)
        self.plotter.render()

    # =================================================================
    # Helpers
    # =================================================================

    def _show_placeholder(self, message: str):
        self.plotter.add_text(
            message,
            position='upper_left', font_size=12, color='gray',
        )
        self.plotter.render()
