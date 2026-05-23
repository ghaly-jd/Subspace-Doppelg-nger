from __future__ import annotations

from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout

from ui.base_screen import BaseScreen


class ExplanationScreen(BaseScreen):
    def __init__(self) -> None:
        super().__init__()

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(18)

        panel = QFrame()
        panel.setObjectName("terminalPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(24, 24, 24, 24)
        panel_layout.setSpacing(16)

        body = QLabel(
            "1. Skeleton points form a motion matrix.\n"
            "2. PCA/SVD compresses motion into a subspace.\n"
            "3. Canonical angles compare visitor and archive subspaces.\n"
            "4. Smaller angles mean more similar motion style."
        )
        body.setObjectName("statusLine")

        panel_layout.addWidget(self.make_title("WHAT IS SUBSPACE MATCHING?"))
        panel_layout.addWidget(body)
        panel_layout.addStretch(1)

        root.addWidget(panel, 1)
        self.add_back_button(root)

