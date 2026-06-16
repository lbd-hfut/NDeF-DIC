"""
DisplacementView: 3D displacement field with timeline slider.

Displays:
  - 3D point cloud of deformed shape, colored by |Φ| or U/V/W component
  - QSlider to switch between load steps in real time
  - QComboBox to select data source (DIC / Ground Truth)
  - QComboBox to select component (|Φ| / U / V / W)
"""

import numpy as np
from typing import Optional, Dict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QComboBox,
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

        # ---- Active dataset (DIC) ----
        self._steps: list = []                                # sorted step numbers
        self._render_deformed: Dict[int, np.ndarray] = {}     # step → (N, 3)
        self._render_mags: Dict[int, np.ndarray] = {}         # step → (N,)
        self._render_U: Dict[int, np.ndarray] = {}            # step → (N,)
        self._render_V: Dict[int, np.ndarray] = {}            # step → (N,)
        self._render_W: Dict[int, np.ndarray] = {}            # step → (N,)
        self._render_ref: Optional[np.ndarray] = None
        self._global_mag_vmin: float = 0.0
        self._global_mag_vmax: float = 1.0
        self._global_cmp_vmax: float = 1.0                    # symmetric for U/V/W
        self._n_total: int = 0

        # ---- Ground truth dataset (optional) ----
        self._has_gt: bool = False
        self._gt_steps: list = []
        self._gt_render_deformed: Dict[int, np.ndarray] = {}
        self._gt_render_mags: Dict[int, np.ndarray] = {}
        self._gt_render_U: Dict[int, np.ndarray] = {}
        self._gt_render_V: Dict[int, np.ndarray] = {}
        self._gt_render_W: Dict[int, np.ndarray] = {}
        self._gt_render_ref: Optional[np.ndarray] = None
        self._gt_n_total: int = 0
        # GT color ranges
        self._gt_mag_vmin: float = 0.0
        self._gt_mag_vmax: float = 1.0
        self._gt_cmp_vmax: float = 1.0

        # UI state
        self._source: str = "dic"          # "dic" | "gt"
        self._component: str = "mag"        # "mag" | "U" | "V" | "W"

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
        self.plotter.show_grid(color='gray', all_edges=True)

        # ---- Control bar ----
        control = QWidget()
        control_layout = QVBoxLayout(control)
        control_layout.setContentsMargins(8, 4, 8, 4)

        # Row 1: source + component selectors
        selector_row = QHBoxLayout()

        selector_row.addWidget(QLabel("Data Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItem("DIC Result", "dic")
        self.source_combo.currentIndexChanged.connect(self._on_controls_changed)
        selector_row.addWidget(self.source_combo)

        selector_row.addSpacing(16)
        selector_row.addWidget(QLabel("Component:"))
        self.component_combo = QComboBox()
        self.component_combo.addItem("|Φ| (Magnitude)", "mag")
        self.component_combo.addItem("U (X disp.)", "U")
        self.component_combo.addItem("V (Y disp.)", "V")
        self.component_combo.addItem("W (Z disp.)", "W")
        self.component_combo.currentIndexChanged.connect(self._on_controls_changed)
        selector_row.addWidget(self.component_combo)

        selector_row.addStretch()
        self.point_count_label = QLabel("")
        selector_row.addWidget(self.point_count_label)
        control_layout.addLayout(selector_row)

        # Row 2: step info
        info_row = QHBoxLayout()
        self.step_label = QLabel("No data")
        self.step_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        info_row.addWidget(self.step_label)

        self.stat_label = QLabel("")
        info_row.addWidget(self.stat_label)
        info_row.addStretch()
        control_layout.addLayout(info_row)

        # Row 3: slider
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
        self.slider.setTracking(True)
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
        gt_ref_points: Optional[np.ndarray] = None,
        gt_disp_fields: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """Load DIC and optional ground truth displacement results.

        Args:
            ref_points:   (P, 3) DIC reference surface points
            disp_fields:  {step: (P, 3)} DIC displacement vectors
            def_points:   {step: (P, 3)} DIC deformed point positions (optional)
            gt_ref_points:   (P_gt, 3) GT reference points (optional)
            gt_disp_fields:  {step: (P_gt, 3)} GT displacement (optional)
        """
        self.clear()

        if ref_points is None or disp_fields is None or len(disp_fields) == 0:
            self._show_placeholder("No displacement data available.\nRun Step 3 training first.")
            return

        # ---- DIC pre-computation ----
        self._steps = sorted(disp_fields.keys())
        self._n_total = len(ref_points)

        # Subsample index
        if self._n_total > RENDER_MAX_POINTS:
            rng = np.random.default_rng(42)
            self._render_idx = rng.choice(self._n_total, RENDER_MAX_POINTS, replace=False)
            self._render_idx.sort()
        else:
            self._render_idx = np.arange(self._n_total)

        self._render_ref = ref_points[self._render_idx]
        render_n = len(self._render_ref)

        self._render_mags = {}
        self._render_U = {}
        self._render_V = {}
        self._render_W = {}
        self._render_deformed = {}
        all_mags = []
        all_abs_u, all_abs_v, all_abs_w = [], [], []

        for step in self._steps:
            disp = disp_fields[step][self._render_idx].astype(np.float32)
            # Components
            self._render_U[step] = disp[:, 0]
            self._render_V[step] = disp[:, 1]
            self._render_W[step] = disp[:, 2]
            # Magnitude
            mag = np.linalg.norm(disp, axis=1).astype(np.float32)
            self._render_mags[step] = mag
            all_mags.append(mag)
            all_abs_u.append(np.abs(disp[:, 0]))
            all_abs_v.append(np.abs(disp[:, 1]))
            all_abs_w.append(np.abs(disp[:, 2]))
            # Deformed positions
            if def_points is not None and step in def_points:
                self._render_deformed[step] = def_points[step][self._render_idx].astype(np.float32)
            else:
                self._render_deformed[step] = (self._render_ref + disp).astype(np.float32)

        # DIC color ranges
        global_mag = np.concatenate(all_mags)
        self._global_mag_vmin = float(np.percentile(global_mag, 2))
        self._global_mag_vmax = float(np.percentile(global_mag, 98))
        if self._global_mag_vmax - self._global_mag_vmin < 1e-8:
            self._global_mag_vmax = self._global_mag_vmin + 1e-4

        # Symmetric vmax for U/V/W (max |value| at 98th percentile pooled across components)
        global_abs = np.concatenate([np.concatenate(all_abs_u),
                                      np.concatenate(all_abs_v),
                                      np.concatenate(all_abs_w)])
        self._global_cmp_vmax = float(np.percentile(global_abs, 98))
        if self._global_cmp_vmax < 1e-8:
            self._global_cmp_vmax = 1e-4

        # ---- GT pre-computation ----
        self._has_gt = False
        if gt_ref_points is not None and gt_disp_fields is not None and len(gt_disp_fields) > 0:
            self._has_gt = True
            self._gt_steps = sorted(gt_disp_fields.keys())
            self._gt_n_total = len(gt_ref_points)

            if self._gt_n_total > RENDER_MAX_POINTS:
                rng = np.random.default_rng(42)
                self._gt_render_idx = rng.choice(self._gt_n_total, RENDER_MAX_POINTS, replace=False)
                self._gt_render_idx.sort()
            else:
                self._gt_render_idx = np.arange(self._gt_n_total)

            self._gt_render_ref = gt_ref_points[self._gt_render_idx]

            self._gt_render_mags = {}
            self._gt_render_U = {}
            self._gt_render_V = {}
            self._gt_render_W = {}
            self._gt_render_deformed = {}
            gt_all_mags = []
            gt_all_abs = []

            for step in self._gt_steps:
                disp = gt_disp_fields[step][self._gt_render_idx].astype(np.float32)
                self._gt_render_U[step] = disp[:, 0]
                self._gt_render_V[step] = disp[:, 1]
                self._gt_render_W[step] = disp[:, 2]
                mag = np.linalg.norm(disp, axis=1).astype(np.float32)
                self._gt_render_mags[step] = mag
                gt_all_mags.append(mag)
                gt_all_abs.append(np.abs(disp[:, 0]))
                gt_all_abs.append(np.abs(disp[:, 1]))
                gt_all_abs.append(np.abs(disp[:, 2]))
                self._gt_render_deformed[step] = (self._gt_render_ref + disp).astype(np.float32)

            gt_mag = np.concatenate(gt_all_mags)
            self._gt_mag_vmin = float(np.percentile(gt_mag, 2))
            self._gt_mag_vmax = float(np.percentile(gt_mag, 98))
            if self._gt_mag_vmax - self._gt_mag_vmin < 1e-8:
                self._gt_mag_vmax = self._gt_mag_vmin + 1e-4

            gt_abs = np.concatenate(gt_all_abs)
            self._gt_cmp_vmax = float(np.percentile(gt_abs, 98))
            if self._gt_cmp_vmax < 1e-8:
                self._gt_cmp_vmax = 1e-4

        # ---- Configure source combo ----
        self.source_combo.blockSignals(True)
        current_source = self.source_combo.currentData()
        self.source_combo.clear()
        self.source_combo.addItem("DIC Result", "dic")
        if self._has_gt:
            self.source_combo.addItem("Ground Truth", "gt")
        # Restore previous selection if still valid
        idx = self.source_combo.findData(current_source)
        self.source_combo.setCurrentIndex(max(idx, 0))
        self.source_combo.blockSignals(False)

        # ---- Configure component combo ----
        self.component_combo.setCurrentIndex(0)  # Reset to magnitude

        # ---- Update source state ----
        self._source = self.source_combo.currentData()

        # ---- Configure slider for active source ----
        active_steps = self._steps if self._source == "dic" else self._gt_steps
        n_active = len(active_steps)
        self._configure_slider(active_steps)
        self.point_count_label.setText(
            f"{render_n:,} rendered / {self._n_total:,} total"
        )

        # ---- Show first step ----
        self._update_display(0)
        self.plotter.view_isometric()
        self.plotter.render()

        gt_info = f" + {self._gt_n_total:,} GT pts, {len(self._gt_steps)} GT steps" if self._has_gt else ""
        print(f"[DisplacementView] Loaded {len(self._steps)} steps, "
              f"{render_n} render pts{gt_info}")

    def clear(self) -> None:
        """Remove all actors and reset."""
        self.plotter.clear()
        self.plotter.set_background("#1a1a2e")
        self.plotter.add_axes(
            xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)',
            line_width=1, color='white',
        )
        self.plotter.show_grid(color='gray', all_edges=True)
        self._cloud_actor = None
        self._steps = []
        self._render_mags = {}
        self._render_U = {}
        self._render_V = {}
        self._render_W = {}
        self._render_deformed = {}
        self._has_gt = False
        self._gt_steps = []
        self._gt_render_mags = {}
        self._gt_render_U = {}
        self._gt_render_V = {}
        self._gt_render_W = {}
        self._gt_render_deformed = {}
        self.slider.setEnabled(False)
        self.step_label.setText("No data")
        self.stat_label.setText("")

    def reset_camera(self) -> None:
        self.plotter.view_isometric()
        self.plotter.render()

    # =================================================================
    # Slots
    # =================================================================

    def _on_controls_changed(self):
        """Source or component selector changed."""
        new_source = self.source_combo.currentData()
        new_comp = self.component_combo.currentData()

        source_changed = (new_source != self._source)
        self._source = new_source
        self._component = new_comp

        if source_changed:
            # Update slider range for the new source
            active_steps = self._steps if self._source == "dic" else self._gt_steps
            self._configure_slider(active_steps)

            # Update point count label
            if self._source == "dic":
                self.point_count_label.setText(
                    f"{len(self._render_ref):,} rendered / {self._n_total:,} total"
                )
            else:
                self.point_count_label.setText(
                    f"{len(self._gt_render_ref):,} rendered / {self._gt_n_total:,} total"
                )

        self._update_display(self.slider.value())

    def _on_slider_changed(self, slider_idx: int):
        """Called when user drags the timeline slider."""
        active_steps = self._steps if self._source == "dic" else self._gt_steps
        if not active_steps or slider_idx >= len(active_steps):
            return
        self._update_display(slider_idx)

    # =================================================================
    # Internal: slider → 3D update
    # =================================================================

    def _update_display(self, slider_idx: int):
        """Update the 3D view for the active source + component."""
        active_steps = self._steps if self._source == "dic" else self._gt_steps
        if not active_steps:
            return

        step = active_steps[slider_idx]

        # ---- Pick data arrays per source ----
        if self._source == "dic":
            deformed = self._render_deformed[step]
            if self._component == "mag":
                scalars = self._render_mags[step]
                title = "|Φ| (mm)"
                clim = [self._global_mag_vmin, self._global_mag_vmax]
            elif self._component == "U":
                scalars = self._render_U[step]
                title = "U (mm)"
                clim = [-self._global_cmp_vmax, self._global_cmp_vmax]
            elif self._component == "V":
                scalars = self._render_V[step]
                title = "V (mm)"
                clim = [-self._global_cmp_vmax, self._global_cmp_vmax]
            else:  # W
                scalars = self._render_W[step]
                title = "W (mm)"
                clim = [-self._global_cmp_vmax, self._global_cmp_vmax]
        else:
            deformed = self._gt_render_deformed[step]
            if self._component == "mag":
                scalars = self._gt_render_mags[step]
                title = "|Φ| GT (mm)"
                clim = [self._gt_mag_vmin, self._gt_mag_vmax]
            elif self._component == "U":
                scalars = self._gt_render_U[step]
                title = "U GT (mm)"
                clim = [-self._gt_cmp_vmax, self._gt_cmp_vmax]
            elif self._component == "V":
                scalars = self._gt_render_V[step]
                title = "V GT (mm)"
                clim = [-self._gt_cmp_vmax, self._gt_cmp_vmax]
            else:  # W
                scalars = self._gt_render_W[step]
                title = "W GT (mm)"
                clim = [-self._gt_cmp_vmax, self._gt_cmp_vmax]

        # Build PolyData
        cloud = pv.PolyData(deformed)
        cloud.point_data[title] = scalars

        # Replace actor
        if self._cloud_actor is not None:
            self.plotter.remove_actor(self._cloud_actor)

        # Colormap: 'turbo' for magnitude (sequential), 'coolwarm' for components (diverging)
        cmap = 'turbo' if self._component == "mag" else 'coolwarm'

        self._cloud_actor = self.plotter.add_mesh(
            cloud,
            scalars=title,
            cmap=cmap,
            point_size=4,
            render_points_as_spheres=True,
            clim=clim,
            scalar_bar_args={
                'title': title,
                'vertical': True,
                'position_x': 0.88,
                'position_y': 0.08,
                'width': 0.05,
                'height': 0.35,
            },
        )

        # Update labels
        mean_val = float(scalars.mean())
        max_val = float(scalars.max())
        source_label = "DIC" if self._source == "dic" else "GT"
        comp_label = {"mag": "|Φ|", "U": "U", "V": "V", "W": "W"}[self._component]
        self.step_label.setText(f"{source_label} Step {step}")
        self.stat_label.setText(
            f"{comp_label} mean: {mean_val:.4f} mm  |  max: {max_val:.4f} mm"
        )

        self.plotter.render()

    # =================================================================
    # Helpers
    # =================================================================

    def _configure_slider(self, steps: list):
        """Configure slider range for a given step list."""
        n = len(steps)
        self.slider.blockSignals(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(n - 1, 0))
        self.slider.setValue(0)
        self.slider.setEnabled(n > 0)
        self.slider.setVisible(n > 1)
        self.slider.blockSignals(False)

        if n > 0:
            self.slider_start_label.setText(f"Step {steps[0]}")
            self.slider_end_label.setText(f"Step {steps[-1]}")
        else:
            self.slider_start_label.setText("Step 0")
            self.slider_end_label.setText("Step 0")

    def _show_placeholder(self, message: str):
        self.plotter.add_text(
            message,
            position='upper_left', font_size=12, color='gray',
        )
        self.plotter.render()
        self.slider.setEnabled(False)
