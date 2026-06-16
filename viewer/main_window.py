"""
MainWindow: top-level QMainWindow for the NDeF-DIC Result Viewer.

Layout:
  ┌─────────────────────────────────────────────────────────┐
  │  Data Dir: [______________] [Browse] [Load]              │
  ├─────────────────────────────────────────────────────────┤
  │  [Sparse] [Dense] [Displacement] [Images]  ← QTabWidget │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │              3D View / Image Grid               │    │
  │  └─────────────────────────────────────────────────┘    │
  │  (timeline slider — displacement tab only)              │
  ├─────────────────────────────────────────────────────────┤
  │  Status: ready                                          │
  └─────────────────────────────────────────────────────────┘
"""

import os
import sys
from typing import Optional, List, Dict

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTabWidget,
    QStatusBar, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt, QSize

from viewer.data_loader import DataLoader, LoadedData
from viewer.widgets.sparse_view import Sparse3DView
from viewer.widgets.dense_view import Dense3DView
from viewer.widgets.displacement_view import DisplacementView
from viewer.widgets.image_viewer import ImageViewer


class MainWindow(QMainWindow):
    """NDeF-DIC Result Viewer main window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NDeF-DIC Result Viewer")
        self.resize(1400, 900)
        self._loaded_data: Optional[LoadedData] = None
        self._setup_ui()

    # =================================================================
    # UI construction
    # =================================================================

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # ---- Top bar: data directory input ----
        self._setup_top_bar(main_layout)

        # ---- Tab widget (4 tabs) ----
        self.tabs = QTabWidget()

        self.sparse_view = Sparse3DView()
        self.dense_view = Dense3DView()
        self.disp_view = DisplacementView()
        self.image_viewer = ImageViewer()

        self.tabs.addTab(self.sparse_view, "🔭 稀疏重建")
        self.tabs.addTab(self.dense_view, "☁️ 稠密重建")
        self.tabs.addTab(self.disp_view, "📐 位移场")
        self.tabs.addTab(self.image_viewer, "📷 原始图像")

        main_layout.addWidget(self.tabs, stretch=1)

        # ---- Status bar ----
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — enter a data directory and click Load")

    def _setup_top_bar(self, parent_layout):
        """Build the top control bar."""
        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)

        top_layout.addWidget(QLabel("Data Directory:"))

        self.data_dir_input = QLineEdit()
        self.data_dir_input.setPlaceholderText("e.g., case/CylinderDIC")
        self.data_dir_input.returnPressed.connect(self._on_load)
        top_layout.addWidget(self.data_dir_input, stretch=1)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse)
        top_layout.addWidget(browse_btn)

        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self._on_load)
        self.load_btn.setMinimumWidth(80)
        self.load_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #2979ff; color: white; "
            "border-radius: 4px; padding: 4px 12px; }"
            "QPushButton:hover { background-color: #448aff; }"
        )
        top_layout.addWidget(self.load_btn)

        parent_layout.addWidget(top)

    # =================================================================
    # Slots
    # =================================================================

    def _on_browse(self):
        """Open file dialog to select data directory."""
        start_dir = self.data_dir_input.text().strip() or os.getcwd()
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Data Directory", start_dir,
        )
        if dir_path:
            self.data_dir_input.setText(dir_path)

    def _on_load(self):
        """Load data from the specified directory."""
        data_dir = self.data_dir_input.text().strip()
        if not data_dir:
            QMessageBox.warning(self, "No Directory", "Please enter a data directory path.")
            return

        if not os.path.isdir(data_dir):
            QMessageBox.warning(
                self, "Invalid Path",
                f"Directory not found:\n{data_dir}",
            )
            return

        self.status_bar.showMessage(f"Loading {data_dir} ...")
        self.load_btn.setEnabled(False)
        self.load_btn.setText("Loading...")

        try:
            loader = DataLoader()
            data = loader.load(data_dir)
            self._loaded_data = data

            # Show warnings if any
            if loader.errors:
                QMessageBox.information(
                    self, "Load Warnings",
                    "Some data could not be loaded:\n\n" +
                    "\n".join(f"• {e}" for e in loader.errors),
                )

            # Distribute data to tabs
            self._distribute_data(data)

            # Update status
            self._update_status(data)

            # Activate best tab based on what was loaded
            self._activate_best_tab(data)

        except Exception as e:
            QMessageBox.critical(
                self, "Load Error",
                f"Failed to load data:\n\n{type(e).__name__}: {e}",
            )
        finally:
            self.load_btn.setEnabled(True)
            self.load_btn.setText("Load")

    # =================================================================
    # Data distribution
    # =================================================================

    def _distribute_data(self, data: LoadedData):
        """Send loaded data to the appropriate tab widgets."""

        # Sparse view
        cameras = self._build_camera_dicts(data)
        self.sparse_view.set_data(data.sparse_points, cameras)

        # Dense view
        self.dense_view.set_data(
            data.dense_points, data.dense_normals, data.dense_vis_mask,
        )

        # Displacement view
        self.disp_view.set_data(
            data.ref_points, data.disp_fields, data.def_points,
            data.gt_ref_points, data.gt_disp_fields,
        )

        # Image viewer
        self.image_viewer.set_data(
            data.cam_names, data.ref_image_paths, data.def_image_paths,
        )

    def _build_camera_dicts(self, data: LoadedData) -> Optional[List[Dict]]:
        """Convert calibration arrays to list-of-dicts for the sparse view."""
        if data.K_list is None or data.num_cameras == 0:
            return None

        cameras = []
        for i in range(data.num_cameras):
            cameras.append({
                "K": data.K_list[i],
                "R": data.R_list[i],
                "t": data.t_list[i],
                "id": i,
                "model": (data.camera_models[i]
                          if data.camera_models and i < len(data.camera_models)
                          else "PINHOLE"),
            })
        return cameras

    def _update_status(self, data: LoadedData):
        """Update status bar with data summary."""
        parts = []

        if data.num_cameras > 0:
            parts.append(f"{data.num_cameras} cameras")

        if data.sparse_points is not None:
            parts.append(f"{len(data.sparse_points):,} sparse pts")

        if data.dense_points is not None:
            parts.append(f"{len(data.dense_points):,} dense pts")

        if data.ref_points is not None:
            parts.append(f"{len(data.ref_points):,} ref pts")
            if data.n_steps > 0:
                parts.append(f"{data.n_steps} disp steps")

        if data.gt_ref_points is not None:
            parts.append(f"{len(data.gt_ref_points):,} GT pts")
            if data.gt_n_steps > 0:
                parts.append(f"{data.gt_n_steps} GT steps")

        if data.cam_names is not None:
            parts.append(f"{len(data.cam_names)} image sets")

        msg = " | ".join(parts) if parts else "No data loaded"
        self.status_bar.showMessage(f"Loaded: {msg}")

    def _activate_best_tab(self, data: LoadedData):
        """Activate the tab that has the most interesting data."""
        if data.ref_points is not None and data.n_steps > 0:
            self.tabs.setCurrentWidget(self.disp_view)
        elif data.dense_points is not None:
            self.tabs.setCurrentWidget(self.dense_view)
        elif data.sparse_points is not None:
            self.tabs.setCurrentWidget(self.sparse_view)
        elif data.cam_names is not None:
            self.tabs.setCurrentWidget(self.image_viewer)

    # =================================================================
    # Public method (for CLI pre-load)
    # =================================================================

    def load_data(self, data_dir: str):
        """Programmatic load (used by main.py --data-dir)."""
        self.data_dir_input.setText(data_dir)
        self._on_load()
