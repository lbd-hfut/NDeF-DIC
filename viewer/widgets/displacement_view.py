"""
DisplacementView: 3D displacement field with timeline slider.

Displays:
  - 3D point cloud of deformed shape, colored by displacement magnitude
  - QSlider to switch between load steps in real time
  - Step information label
"""

import numpy as np
from typing import Optional, Dict, Tuple

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
)
from PySide6.QtCore import Qt

import pyvista as pv
from pyvistaqt import QtInteractor


RENDER_MAX_POINTS = 50_000


class DisplacementView(QWidget):
    """Tab: Displacement field visualization with timeline."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cloud_actor = None
        self._steps: list = []
        self._render_mags: Dict[int, np.ndarray] = {}
        self._render_deformed: Dict[int, np.ndarray] = {}
        self._render_ref: Optional[np.ndarray] = None
        self._global_vmin: float = 0.0
        self._global_vmax: float = 1.0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 3D viewport
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("#1a1a2e")
        self.plotter.add_axes(
            xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)',
            line_width=1, color='white',
        )
        self.plotter.show_grid(color='gray', all_edges=True, opacity=0.15)

        # ---- Timeline control bar ----
        control = QWidget()
        control_layout = QVBoxLayout(control)
        control_layout.setContentsMargins(8, 4, 8, 4)

        # Step info row
        info_row = QHBoxLayout()
        self.step_label = QLabel("No data")
        self.step_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        info_row.addWidget(self.step_label)

        self.mag_label = QLabel("")
        info_row.addWidget(self.mag_label)
        info_row.addStretch()

        self.point_count_label = QLabel("")
        info_row.addWidget(self.point_count_label)
        control_layout.addLayout(info_row)

        # Slider row
        slider_row = QHBoxLayout()
        self.slider_start_label = QLabel("Step 0")
        slider_row.addWidget(self.slider_start_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setTracking(True)  # fire valueChanged continuously while dragging
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.setEnabled(False)
        slider_row.addWidget(self.slider)

        self.slider_end_label = QLabel("Step 0")
        slider_row.addWidget(self.slider_end_label)
        control_layout.addLayout(slider_row)

        layout.addWidget(control)

        self._show_placeholder("No displacement data.\nRun Step 3 training first.")

    # =================================================================
    # Public API
    # =================================================================

    def set_data(
        self,
        ref_points: Optional[np.ndarray],
        disp_fields: Optional[Dict[int, np.ndarray]],
        def_points: Optional[Dict[int, np.ndarray]],
    ) -> None:
        """Load displacement results. Pre-computes all step data for fast slider response.

        Args:
            ref_points: (P, 3) float32 reference surface points
            disp_fields: {step: (P, 3)} displacement vectors
            def_points: {step: (P, 3)} deformed point positions (optional)
        """
        self.clear()

        if ref_points is None or disp_fields is None or len(disp_fields) == 0:
            self._show_placeholder("No displacement data available.\nRun Step 3 training first.")
            return

        n_pts = len(ref_points)
        self._steps = sorted(disp_fields.keys())
        n_steps = len(self._steps)

        # ---- Subsample for rendering ----
        if n_pts > RENDER_MAX_POINTS:
            rng = np.random.default_rng(42)
            self._render_idx = rng.choice(n_pts, RENDER_MAX_POINTS, replace=False)
            self._render_idx.sort()
        else:
            self._render_idx = np.arange(n_pts)

        self._render_ref = ref_points[self._render_idx]
        render_n = len(self._render_ref)

        # ---- Pre-compute all step magnitudes and deformed positions ----
        self._render_mags = {}
        self._render_deformed = {}
        all_mags = []

        for step in self._steps:
            disp = disp_fields[step][self._render_idx]
            mag = np.linalg.norm(disp, axis=1).astype(np.float32)
            self._render_mags[step] = mag
            all_mags.append(mag)

            if def_points is not None and step in def_points:
                self._render_deformed[step] = def_points[step][self._render_idx].astype(np.float32)
            else:
                self._render_deformed[step] = (self._render_ref + disp).astype(np.float32)

        # ---- Global color range (consistent across steps) ----
        global_mag = np.concatenate(all_mags)
        self._global_vmin = float(np.percentile(global_mag, 2))
        self._global_vmax = float(np.percentile(global_mag, 98))
        if self._global_vmax - self._global_vmin < 1e-8:
            self._global_vmax = self._global_vmin + 1e-4

        # ---- Configure slider ----
        self.slider.blockSignals(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(n_steps - 1)
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self.slider.setVisible(n_steps > 1)
        self.slider.blockSignals(False)

        self.slider_start_label.setText(f"Step {self._steps[0]}")
        self.slider_end_label.setText(f"Step {self._steps[-1]}")
        self.point_count_label.setText(f"{render_n:,} pts rendered / {n_pts:,} total")

        # ---- Show first step ----
        self._update_display(0)
        self.plotter.view_isometric()
        self.plotter.render()

        print(f"[DisplacementView] Loaded {n_steps} steps, "
              f"{render_n} render pts, "
              f"color range [{self._global_vmin:.4f}, {self._global_vmax:.4f}]")

    def clear(self) -> None:
        """Remove all actors and reset."""
        self.plotter.clear()
        self.plotter.set_background("#1a1a2e")
        self.plotter.add_axes(
            xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)',
            line_width=1, color='white',
        )
        self.plotter.show_grid(color='gray', all_edges=True, opacity=0.15)
        self._cloud_actor = None
        self._steps = []
        self._render_mags = {}
        self._render_deformed = {}
        self.slider.setEnabled(False)
        self.step_label.setText("No data")
        self.mag_label.setText("")

    def reset_camera(self) -> None:
        self.plotter.view_isometric()
        self.plotter.render()

    # =================================================================
    # Internal: slider → 3D update
    # =================================================================

    def _on_slider_changed(self, slider_idx: int):
        """Called when user drags the timeline slider."""
        if not self._steps or slider_idx >= len(self._steps):
            return
        self._update_display(slider_idx)

    def _update_display(self, slider_idx: int):
        """Update the 3D view to show the selected step."""
        step = self._steps[slider_idx]
        mag = self._render_mags[step]
        deformed = self._render_deformed[step]

        # Build new PolyData
        cloud = pv.PolyData(deformed)
        cloud.point_data['|Φ| (mm)'] = mag

        # Replace actor
        if self._cloud_actor is not None:
            self.plotter.remove_actor(self._cloud_actor)

        self._cloud_actor = self.plotter.add_mesh(
            cloud,
            scalars='|Φ| (mm)',
            cmap='turbo',
            point_size=4,
            render_points_as_spheres=True,
            clim=[self._global_vmin, self._global_vmax],
            scalar_bar_args={
                'title': '|Φ| (mm)',
                'vertical': True,
                'position_x': 0.88,
                'position_y': 0.08,
                'width': 0.05,
                'height': 0.35,
            },
        )

        # Update labels
        mean_mag = float(mag.mean())
        max_mag = float(mag.max())
        self.step_label.setText(f"Step {step}")
        self.mag_label.setText(
            f"|Φ| mean: {mean_mag:.4f} mm  |  max: {max_mag:.4f} mm"
        )

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
        self.slider.setEnabled(False)
