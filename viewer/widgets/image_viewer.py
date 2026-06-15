"""
ImageViewer: scrollable grid of camera image thumbnails.

Displays:
  - QScrollArea with a grid of camera images (reference + deformed)
  - Camera selection via combo box
  - Lazy QPixmap loading from disk paths
"""

import os
import cv2
import numpy as np
from typing import Optional, List, Dict

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QGroupBox, QGridLayout, QComboBox, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage

THUMBNAIL_W = 320
THUMBNAIL_H = 240


class ImageViewer(QWidget):
    """Tab: Camera image grid viewer."""

    camera_selected = Signal(int)  # emitted when user selects a camera

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cam_names: List[str] = []
        self._ref_paths: List[str] = []
        self._def_paths: Dict[int, List[str]] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Image type selector
        control = QHBoxLayout()
        control.addWidget(QLabel("Show:"))
        self.image_type_combo = QComboBox()
        self.image_type_combo.addItem("Reference (undeformed)", "ref")
        self.image_type_combo.currentIndexChanged.connect(self._on_type_changed)
        control.addWidget(self.image_type_combo)
        control.addStretch()

        self.info_label = QLabel("No images loaded")
        control.addWidget(self.info_label)
        layout.addLayout(control)

        # Scrollable grid area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(8)
        self.scroll_area.setWidget(self.grid_widget)

        layout.addWidget(self.scroll_area)

    # =================================================================
    # Public API
    # =================================================================

    def set_data(
        self,
        cam_names: Optional[List[str]],
        ref_image_paths: Optional[List[str]],
        def_image_paths: Optional[Dict[int, List[str]]],
    ) -> None:
        """Load and display camera image thumbnails.

        Args:
            cam_names: list of camera directory names
            ref_image_paths: per-camera reference image paths
            def_image_paths: {step_idx: [per-camera paths]} for deformed images
        """
        self.clear()

        if cam_names is None or ref_image_paths is None:
            self.info_label.setText("No images found")
            return

        self._cam_names = cam_names
        self._ref_paths = ref_image_paths
        self._def_paths = def_image_paths or {}

        # Populate image type combo
        self.image_type_combo.blockSignals(True)
        self.image_type_combo.clear()
        self.image_type_combo.addItem("Reference (undeformed)", "ref")
        for step_idx in sorted(self._def_paths.keys()):
            self.image_type_combo.addItem(f"Deformed Step {step_idx + 1}", f"def_{step_idx}")
        self.image_type_combo.blockSignals(False)

        # Show reference by default
        self._show_images("ref")

        n_cam = len(cam_names)
        n_def = len(self._def_paths)
        self.info_label.setText(f"{n_cam} cameras, {n_def} deformed sets")

        print(f"[ImageViewer] Displayed {n_cam} camera thumbnails")

    def clear(self) -> None:
        """Remove all thumbnails."""
        # Remove all widgets from grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cam_names = []
        self._ref_paths = []
        self._def_paths = {}
        self.info_label.setText("No images loaded")

    # =================================================================
    # Internal
    # =================================================================

    def _show_images(self, mode: str):
        """Load and display thumbnails for the given mode."""
        # Remove existing widgets
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Determine which paths to use
        if mode == "ref":
            paths = self._ref_paths
            label_prefix = "Ref"
        elif mode.startswith("def_"):
            step_idx = int(mode.split("_")[1])
            paths = self._def_paths.get(step_idx, [])
            label_prefix = f"Step {step_idx + 1}"
        else:
            return

        n_cols = 4
        for i, (cam_name, path) in enumerate(zip(self._cam_names, paths)):
            group = self._create_camera_card(cam_name, path, label_prefix, i)
            row, col = i // n_cols, i % n_cols
            self.grid_layout.addWidget(group, row, col)

        # Fill empty cells with spacers
        total = len(self._cam_names)
        last_row = (total - 1) // n_cols
        last_col = (total - 1) % n_cols
        for j in range(last_col + 1, n_cols):
            self.grid_layout.addWidget(QWidget(), last_row, j)

    def _create_camera_card(
        self, cam_name: str, path: Optional[str], label_prefix: str, cam_idx: int,
    ) -> QGroupBox:
        """Create a single camera image card."""
        group = QGroupBox(f"{cam_name}  [{label_prefix}]")
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(4, 4, 4, 4)

        if path and os.path.exists(path):
            # Load image via OpenCV, convert to QPixmap
            pixmap = self._load_thumbnail(path)
            img_label = QLabel()
            img_label.setPixmap(pixmap)
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            img_label.setToolTip(f"Camera {cam_idx}: {os.path.basename(path)}")
            group_layout.addWidget(img_label)
        else:
            no_img = QLabel("(no image)")
            no_img.setAlignment(Qt.AlignCenter)
            no_img.setStyleSheet("color: gray;")
            no_img.setFixedSize(THUMBNAIL_W, THUMBNAIL_H)
            group_layout.addWidget(no_img)

        return group

    def _load_thumbnail(self, path: str) -> QPixmap:
        """Load an image from disk, convert to QPixmap thumbnail."""
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return self._error_pixmap()

            h, w = img.shape[:2]

            if len(img.shape) == 2:
                # Grayscale
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 3:
                # BGR → RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:
                # BGRA → RGBA
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            else:
                img_rgb = img

            bytes_per_line = img_rgb.shape[2] * img_rgb.shape[1]
            qimg = QImage(
                img_rgb.data, img_rgb.shape[1], img_rgb.shape[0],
                bytes_per_line, QImage.Format_RGB888,
            )
            pixmap = QPixmap.fromImage(qimg)
            pixmap = pixmap.scaled(
                THUMBNAIL_W, THUMBNAIL_H,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            return pixmap

        except Exception:
            return self._error_pixmap()

    def _error_pixmap(self) -> QPixmap:
        """Return a placeholder pixmap for broken images."""
        pixmap = QPixmap(THUMBNAIL_W, THUMBNAIL_H)
        pixmap.fill(Qt.darkGray)
        return pixmap

    # =================================================================
    # Slots
    # =================================================================

    def _on_type_changed(self, index: int):
        """Handle image type combo change."""
        mode = self.image_type_combo.currentData()
        if mode:
            self._show_images(mode)
