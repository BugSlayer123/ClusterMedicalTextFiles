from PyQt5.QtCore import QEvent, Qt, QElapsedTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QPushButton, QProgressBar, QTabWidget, QLineEdit, QSpinBox, QComboBox, QCheckBox, QTextEdit, \
    QListWidget, QTableWidget, QGroupBox, QLabel, QListView


class StyledButton(QPushButton):
    def __init__(self, text, primary=False):
        super().__init__(text)
        self.primary = primary
        self.setMinimumHeight(40)
        self.setFont(QFont("Segoe UI", 10, QFont.Medium))
        self.apply_style()

    def apply_style(self):
        if self.primary:
            style = """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4A90E2, stop:1 #357ABD);
                    color: white;
                    border: 2px solid #357ABD;
                    border-radius: 8px;
                    padding: 8px 16px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #5BA0F2, stop:1 #4A90E2);
                    border: 2px solid #4A90E2;
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #357ABD, stop:1 #2E6B9E);
                }
                QPushButton:disabled {
                    background: #CCCCCC;
                    color: #666666;
                    border: 2px solid #BBBBBB;
                }
            """
        else:
            style = """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #F8F9FA, stop:1 #E9ECEF);
                    color: #495057;
                    border: 2px solid #DEE2E6;
                    border-radius: 8px;
                    padding: 8px 16px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #FFFFFF, stop:1 #F8F9FA);
                    border: 2px solid #ADB5BD;
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #E9ECEF, stop:1 #DEE2E6);
                }
                QPushButton:disabled {
                    background: #F8F9FA;
                    color: #ADB5BD;
                    border: 2px solid #DEE2E6;
                }
            """
        self.setStyleSheet(style)


class StyledGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.setFont(QFont("Segoe UI", 11, QFont.DemiBold))
        self.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                color: #2C3E50;
                border: 2px solid #BDC3C7;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255,255,255,0.9), stop:1 rgba(248,249,250,0.9));
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 12px;
                background: #FFFFFF;
                border: 2px solid #BDC3C7;
                border-radius: 8px;
                margin-left: 8px;
                color: #2C3E50;
            }
        """)


class StyledProgressBar(QProgressBar):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(8)
        self.setMaximumHeight(8)
        self.setRange(0, 0)
        self.setStyleSheet("""
            QProgressBar {
                background-color: #ECF0F1;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498DB, stop:0.5 #5DADE2, stop:1 #85C1E9);
                border-radius: 4px;
            }
        """)


class StyledTabWidget(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #BDC3C7;
                border-radius: 8px;
                background: white;
                margin-top: 8px;
            }
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F8F9FA, stop:1 #E9ECEF);
                border: 2px solid #DEE2E6;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 10px 20px;
                margin-right: 2px;
                font-weight: 500;
                color: #495057;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFFFFF, stop:1 #F8F9FA);
                border-color: #BDC3C7;
                color: #2C3E50;
                font-weight: 600;
            }
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFFFFF, stop:1 #F1F3F4);
                color: #2C3E50;
            }
        """)


class StyledLineEdit(QLineEdit):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(36)
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet("""
            QLineEdit {
                border: 2px solid #DEE2E6;
                border-radius: 8px;
                padding: 8px 12px;
                background: white;
                color: #495057;
                font-size: 10pt;
            }
            QLineEdit:focus {
                border: 2px solid #4A90E2;
                background: #FFFFFF;
            }
            QLineEdit:hover {
                border: 2px solid #ADB5BD;
            }
        """)


class StyledSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(36)
        self.setFont(QFont("Segoe UI", 10))

        self.setStyleSheet("""
            QSpinBox {
                border: 2px solid #DEE2E6;
                border-radius: 4px;
                padding: 6px 24px 6px 12px; /* leave room for arrows */
                background: white;
                color: #495057;
            }

            QSpinBox:focus {
                border: 2px solid #4A90E2;
            }

            QSpinBox:hover {
                border-color: #ADB5BD;
            }

            /* stepper buttons */
            QSpinBox::up-button, QSpinBox::down-button {
                width: 20px;                 /* small enough to keep arrows normal size */
                subcontrol-origin: padding;  /* native arrows drawn correctly */
                subcontrol-position: top right;
            }

            QSpinBox::down-button {
                subcontrol-position: bottom right;
            }

            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background: rgba(0,0,0,0.03);
            }

            QSpinBox:disabled {
                background: #F8F9FA;
                color: #ADB5BD;
                border-color: #E9ECEF;
            }
        """)


class StyledComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Appearance
        self.setMinimumHeight(36)
        self.setFont(QFont("Segoe UI", 10))
        self.setView(QListView())

        self._open_elapsed = QElapsedTimer()
        self._filter_installed = False

        self.setStyleSheet("""
            QComboBox {
                border: 2px solid #DEE2E6;
                border-radius: 8px;
                padding: 6px 32px 6px 12px; /* right padding leaves room for arrow */
                background: white;
                color: #495057;
            }
            QComboBox:focus {
                border: 2px solid #4A90E2;
            }
            QComboBox:hover {
                border-color: #ADB5BD;
            }

            QComboBox::drop-down {
                subcontrol-position: top right;
                width: 28px;
                border-radius: 8px;
                border-left: none;
                background: transparent;
            }

            QComboBox::down-arrow { image: none; }

            /*QComboBox QAbstractItemView {
                border: 2px solid #DEE2E6;
                border-radius: 8px;
                background: white;
                selection-background-color: #E3F2FD;
                selection-color: #1976D2;
                padding: 4px;
            }*/
        """)

    # ---------- Input handling to avoid "open & immediate select" ----------
    # We accept press but do NOT call the default QComboBox mousePressEvent because
    # on Wayland/X11 this can cause the click to propagate into the popup immediately.
    def mousePressEvent(self, event):
        # record that press happened — do NOT call super() here
        # this avoids QComboBox internals that sometimes open popup on press and propagate events
        if event.button() == Qt.LeftButton:
            event.accept()
            self._pressed_pos = event.pos()
            self._pressed = True
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if getattr(self, "_pressed", False) and event.button() == Qt.LeftButton:
            self._pressed = False
            if self.rect().contains(event.pos()):
                self.showPopup()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def showPopup(self):
        """
        Show the popup but install a short-lived event-filter on the view that
        consumes very-early MouseButtonPress events. This prevents the same
        click that opened the popup from immediately selecting an item.
        """
        super().showPopup()
        self._open_elapsed.restart()

        view = self.view()
        if not self._filter_installed:
            view.installEventFilter(self)
            self._filter_installed = True

    def eventFilter(self, obj, event):
        if obj is self.view():
            if event.type() == QEvent.MouseButtonPress:
                if self._open_elapsed.isValid() and self._open_elapsed.elapsed() < 250:
                    return True
        return super().eventFilter(obj, event)


class StyledCheckBox(QCheckBox):
    def __init__(self, text):
        super().__init__(text)
        self.setFont(QFont("Segoe UI", 10, QFont.Medium))
        # Use a simpler style that works across platforms
        self.setStyleSheet("""
            QCheckBox {
                color: #2C3E50;
                font-weight: 500;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #BDC3C7;
                border-radius: 4px;
                background: white;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #4A90E2;
            }
            QCheckBox::indicator:checked {
                background: #4A90E2;
                border: 2px solid #357ABD;
            }
            QCheckBox::indicator:checked:hover {
                background: #5BA0F2;
                border: 2px solid #4A90E2;
            }
        """)


class StyledLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setFont(QFont("Segoe UI", 10, QFont.Medium))
        self.setStyleSheet("""
            QLabel {
                color: #2C3E50;
                font-weight: 500;
                margin-right: 8px;
            }
        """)


class StyledTextEdit(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QTextEdit {
                border: 1px solid #DEE2E6;
                border-radius: 8px;
                background: white;
                color: #2C3E50;
                padding: 12px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 10pt;
                line-height: 1.5;
            }
            QTextEdit:focus {
                border: 2px solid #4A90E2;
            }
        """)


class StyledListWidget(QListWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QListWidget {
                border: 1px solid #DEE2E6;
                border-radius: 8px;
                background: white;
                alternate-background-color: #F8F9FA;
                font-family: 'Segoe UI', sans-serif;
                font-size: 10pt;
                padding: 4px;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid #ECF0F1;
                border-radius: 4px;
                margin: 2px;
            }
            QListWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #E3F2FD, stop:1 #BBDEFB);
                color: #1976D2;
                border: 1px solid #2196F3;
            }
            QListWidget::item:hover:!selected {
                background: #F5F5F5;
                border: 1px solid #E0E0E0;
            }
        """)


class StyledTableWidget(QTableWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QTableWidget {
                border: 1px solid #DEE2E6;
                border-radius: 8px;
                background: white;
                gridline-color: #ECF0F1;
                font-family: 'Segoe UI', sans-serif;
                font-size: 9pt;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #F1F3F4;
            }
            QTableWidget::item:selected {
                background: #E3F2FD;
                color: #1976D2;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F8F9FA, stop:1 #E9ECEF);
                color: #495057;
                border: 1px solid #DEE2E6;
                padding: 10px;
                font-weight: 600;
                font-size: 10pt;
            }
        """)