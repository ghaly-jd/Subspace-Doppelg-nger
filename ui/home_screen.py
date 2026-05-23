from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QWidget

from ui.brand_background import draw_cvlab_background


class HomeScreen(QWidget):
    register_requested = pyqtSignal()
    match_requested = pyqtSignal()
    archive_requested = pyqtSignal()
    explain_requested = pyqtSignal()
    live_pose_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(20)

        title = QLabel("THE GRASSMANN MIRROR")
        title.setObjectName("titleLabel")
        subtitle = QLabel("Motion Ritual: Find Your Lab Twin")
        subtitle.setObjectName("subtitleLabel")

        panel = QFrame()
        panel.setObjectName("terminalPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(24, 24, 24, 24)
        panel_layout.setSpacing(12)

        register_button = QPushButton("[1] REGISTER LAB RITUAL")
        match_button = QPushButton("[2] FIND YOUR LAB TWIN")
        archive_button = QPushButton("[3] VIEW MOTION ARCHIVE")
        explain_button = QPushButton("[4] EXPLAIN THE MIRROR")
        live_pose_button = QPushButton("[5] LIVE SKELETON PREVIEW")

        register_button.clicked.connect(self.register_requested.emit)
        match_button.clicked.connect(self.match_requested.emit)
        archive_button.clicked.connect(self.archive_requested.emit)
        explain_button.clicked.connect(self.explain_requested.emit)
        live_pose_button.clicked.connect(self.live_pose_requested.emit)

        panel_layout.addWidget(register_button)
        panel_layout.addWidget(match_button)
        panel_layout.addWidget(archive_button)
        panel_layout.addWidget(explain_button)
        panel_layout.addWidget(live_pose_button)
        panel_layout.addStretch(1)

        root.addWidget(title)
        root.addWidget(subtitle)
        root.addWidget(panel, 1)

    def paintEvent(self, event) -> None:
        draw_cvlab_background(self)
