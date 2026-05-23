from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ui.brand_background import draw_cvlab_background


class BaseScreen(QWidget):
    back_requested = pyqtSignal()

    def paintEvent(self, event) -> None:
        draw_cvlab_background(self)

    def make_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("panelTitle")
        return label

    def make_status(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("statusLine")
        label.setWordWrap(True)
        return label

    def add_back_button(self, layout: QVBoxLayout) -> None:
        button = QPushButton("< BACK")
        button.clicked.connect(self.back_requested.emit)
        layout.addWidget(button)
