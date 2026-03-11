"""
main.py - Application Entry Point
===================================
Launches the Algorithm Visualizer GUI.

Usage:
    python main.py
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore    import Qt
from gui import MainWindow


def main():
    # Enable high-DPI scaling on all platforms
    app = QApplication(sys.argv)
    app.setApplicationName("Algorithm Visualizer")
    app.setApplicationVersion("1.0.0")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
