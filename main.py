#!/usr/bin/env python3
"""
Interactive semantic clustering of text files based on user-specified topics.
GUI version using PyQt5.
"""

import sys
from PyQt5.QtWidgets import QApplication

from app.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("YEEEEEEEEEEEEEEEEEEEEEEEEET")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()