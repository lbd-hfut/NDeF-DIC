#!/usr/bin/env python
"""
NDeF-DIC Result Viewer — entry point.

Usage:
    python -m viewer.main
    python -m viewer.main --data-dir case/CylinderDIC
"""

import sys
import os
import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description="NDeF-DIC 3D Result Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m viewer.main                              # launch empty
  python -m viewer.main --data-dir case/CylinderDIC  # pre-load data
        """,
    )
    p.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to data directory (e.g., case/CylinderDIC)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ---- CRITICAL: Create QApplication BEFORE any pyvistaqt import ----
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("NDeF-DIC Viewer")
    app.setOrganizationName("NDeF-DIC")

    # Apply dark-ish stylesheet
    app.setStyle("Fusion")

    # Now it's safe to import modules that use pyvistaqt
    from viewer.main_window import MainWindow

    window = MainWindow()
    window.show()

    # Pre-populate from CLI argument
    if args.data_dir:
        data_dir = os.path.abspath(args.data_dir)
        window.load_data(data_dir)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
