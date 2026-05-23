from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from shared.archive_storage import delete_registration, load_archive_entries
from shared.pose_schema import KEYPOINT_NAMES, SKELETON_EDGES
from shared.ritual_processor import normalize_sequence_for_playback
from ui.base_screen import BaseScreen


class SkeletonArchiveCard(QWidget):
    selected = pyqtSignal(str)

    def __init__(self, entry: dict[str, Any], sequence: np.ndarray | None) -> None:
        super().__init__()
        self.entry = entry
        self.sequence = sequence
        self.frame_index = 0
        self.is_selected = False
        self.setMinimumSize(220, 190)
        self.setMaximumHeight(220)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_selected(self, selected: bool) -> None:
        self.is_selected = selected
        self.update()

    def advance(self) -> None:
        if self.sequence is None or self.sequence.shape[0] == 0:
            return
        self.frame_index = (self.frame_index + 1) % int(self.sequence.shape[0])
        self.update()

    def mousePressEvent(self, event) -> None:
        self.selected.emit(str(self.entry.get("id", "")))
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        rect = QRectF(0, 0, self.width(), self.height()).adjusted(6, 6, -6, -6)
        border = QColor("#7dffe8") if self.is_selected else QColor("#1a5963")
        background = QColor(6, 16, 27, 224)
        painter.setPen(QPen(border, 2 if self.is_selected else 1))
        painter.setBrush(background)
        painter.drawRoundedRect(rect, 8, 8)

        painter.setFont(QFont("Menlo", 11, QFont.Weight.Bold))
        painter.setPen(QColor("#f1fffb"))
        title = str(self.entry.get("nickname", "Unknown"))[:22]
        painter.drawText(rect.adjusted(12, 10, -12, -12), Qt.AlignmentFlag.AlignLeft, title)

        painter.setFont(QFont("Menlo", 9, QFont.Weight.Normal))
        painter.setPen(QColor("#8aa4b8"))
        subtitle = f"{self.entry.get('motion_type', 'Motion')} | {self._frame_text()}"
        painter.drawText(rect.adjusted(12, 30, -12, -12), Qt.AlignmentFlag.AlignLeft, subtitle[:34])

        skeleton_rect = rect.adjusted(14, 48, -14, -18)
        if self.sequence is None or self.sequence.size == 0:
            painter.setPen(QColor("#8aa4b8"))
            painter.drawText(skeleton_rect, Qt.AlignmentFlag.AlignCenter, "No skeleton preview")
            painter.end()
            return

        self._draw_motion_trail(painter, skeleton_rect)
        self._draw_skeleton(painter, skeleton_rect)
        painter.end()

    def _draw_skeleton(self, painter: QPainter, rect: QRectF) -> None:
        frame = np.asarray(self.sequence[self.frame_index], dtype=np.float64)
        mapped = self._mapped_points(frame, rect)
        glow = QColor("#7dffe8")
        glow.setAlpha(80)
        painter.setPen(QPen(glow, 7))
        for start, end in SKELETON_EDGES:
            if start in mapped and end in mapped:
                painter.drawLine(mapped[start], mapped[end])
        painter.setPen(QPen(QColor("#d8fff7"), 3))
        for start, end in SKELETON_EDGES:
            if start in mapped and end in mapped:
                painter.drawLine(mapped[start], mapped[end])
        painter.setBrush(QColor("#ffd166"))
        painter.setPen(QPen(QColor("#030810"), 1))
        for name, point in mapped.items():
            radius = 4 if name.endswith("wrist") or name.endswith("ankle") else 3
            painter.drawEllipse(point, radius, radius)

    def _draw_motion_trail(self, painter: QPainter, rect: QRectF) -> None:
        if self.sequence is None or self.sequence.shape[0] < 2:
            return
        trail_indices = [
            index
            for index, name in enumerate(KEYPOINT_NAMES)
            if name in {"left_wrist", "right_wrist", "left_ankle", "right_ankle"}
        ]
        start = max(0, self.frame_index - 12)
        for step, index in enumerate(range(start, self.frame_index)):
            alpha = int(28 + 95 * ((step + 1) / max(1, self.frame_index - start)))
            color = QColor("#ff7aa8")
            color.setAlpha(alpha)
            painter.setPen(QPen(color, 2))
            mapped_a = self._mapped_points(self.sequence[index], rect)
            mapped_b = self._mapped_points(self.sequence[index + 1], rect)
            for keypoint_index in trail_indices:
                name = KEYPOINT_NAMES[keypoint_index]
                if name in mapped_a and name in mapped_b:
                    painter.drawLine(mapped_a[name], mapped_b[name])

    def _mapped_points(self, frame: np.ndarray, rect: QRectF) -> dict[str, QPointF]:
        points = np.asarray(frame[:, :2], dtype=np.float64)
        finite = points[np.isfinite(points).all(axis=1)]
        if finite.size == 0:
            return {}
        center = np.median(finite, axis=0)
        centered = points - center[None, :]
        finite_centered = centered[np.isfinite(centered).all(axis=1)]
        span = finite_centered.max(axis=0) - finite_centered.min(axis=0)
        scale = min(rect.width() * 0.62, rect.height() * 0.72) / max(float(span.max()), 0.2)
        panel_center = QPointF(rect.center().x(), rect.center().y() + rect.height() * 0.08)
        mapped = {}
        for name, point in zip(KEYPOINT_NAMES, centered):
            if not np.isfinite(point).all():
                continue
            mapped[name] = QPointF(
                panel_center.x() + float(point[0]) * scale,
                panel_center.y() + float(point[1]) * scale,
            )
        return mapped

    def _frame_text(self) -> str:
        quality = self.entry.get("quality", {})
        if isinstance(quality, dict) and quality.get("total_frames") is not None:
            return f"{quality.get('total_frames')} frames"
        if self.sequence is not None:
            return f"{self.sequence.shape[0]} frames"
        return "no frames"


class ArchiveScreen(BaseScreen):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.entries: list[dict[str, Any]] = []
        self.cards: dict[str, SkeletonArchiveCard] = {}
        self.selected_registration_id = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(18)

        panel = QFrame()
        panel.setObjectName("terminalPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(24, 24, 24, 24)
        panel_layout.setSpacing(16)

        panel_layout.addWidget(self.make_title("MOTION ARCHIVE"))

        self.gallery_scroll = QScrollArea()
        self.gallery_scroll.setWidgetResizable(True)
        self.gallery_scroll.setMinimumHeight(300)
        self.gallery_container = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_container)
        self.gallery_layout.setContentsMargins(0, 0, 0, 0)
        self.gallery_layout.setSpacing(12)
        self.gallery_scroll.setWidget(self.gallery_container)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Nickname", "Mode", "Avatar", "Frames", "Quality", "Created", "ID"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setColumnHidden(6, True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._handle_table_selection)

        controls = QHBoxLayout()
        self.refresh_button = QPushButton("[ REFRESH ARCHIVE ]")
        self.refresh_button.clicked.connect(self.refresh)
        self.delete_button = QPushButton("[ DELETE SELECTED ]")
        self.delete_button.clicked.connect(self.delete_selected)
        controls.addWidget(self.refresh_button)
        controls.addWidget(self.delete_button)

        self.status_label = self.make_status("")

        panel_layout.addLayout(controls)
        panel_layout.addWidget(self.gallery_scroll, 2)
        panel_layout.addWidget(self.table, 1)
        panel_layout.addWidget(self.status_label)

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._advance_previews)
        self.preview_timer.start(66)

        root.addWidget(panel, 1)
        self.add_back_button(root)
        self.refresh()

    def refresh(self) -> None:
        self.entries = load_archive_entries(self.config)
        self.table.setRowCount(len(self.entries))
        self._clear_gallery()
        self.cards = {}
        for row, entry in enumerate(self.entries):
            self._set_row(row, entry)
            self._add_gallery_card(row, entry)
        self.table.resizeColumnsToContents()
        self._sync_selection_from_table()
        self.status_label.setText(f"{len(self.entries)} registered motion signature(s).")

    def delete_selected(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.entries):
            self.status_label.setText("Select a registration before deleting.")
            return

        entry = self.entries[row]
        nickname = str(entry.get("nickname", "Unknown"))
        motion_type = str(entry.get("motion_type", "Unknown"))
        registration_id = str(entry.get("id", ""))
        reply = QMessageBox.question(
            self,
            "Delete Registration",
            f"Delete {nickname} / {motion_type} from the archive?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        result = delete_registration(self.config, registration_id)
        if not result.get("deleted"):
            self.status_label.setText(str(result.get("reason", "Delete failed.")))
            return

        deleted_count = len(result.get("files_deleted", []))
        self.refresh()
        self.status_label.setText(
            f"Deleted {nickname} / {motion_type}. Removed {deleted_count} artifact file(s)."
        )

    def _set_row(self, row: int, entry: dict[str, Any]) -> None:
        self.table.setItem(row, 0, QTableWidgetItem(str(entry.get("nickname", "Unknown"))))
        self.table.setItem(row, 1, QTableWidgetItem(str(entry.get("motion_type", "Unknown"))))
        self.table.setItem(row, 2, QTableWidgetItem(str(entry.get("avatar", ""))))
        self.table.setItem(row, 3, QTableWidgetItem(self._frame_text(entry)))
        self.table.setItem(row, 4, QTableWidgetItem(self._quality_text(entry)))
        self.table.setItem(row, 5, QTableWidgetItem(self._timestamp_text(entry)))
        self.table.setItem(row, 6, QTableWidgetItem(str(entry.get("id", ""))))

    def _add_gallery_card(self, row: int, entry: dict[str, Any]) -> None:
        sequence = self._preview_sequence(entry)
        card = SkeletonArchiveCard(entry, sequence)
        card.selected.connect(self._select_registration)
        registration_id = str(entry.get("id", ""))
        self.cards[registration_id] = card
        column_count = 3
        self.gallery_layout.addWidget(card, row // column_count, row % column_count)

    def _clear_gallery(self) -> None:
        while self.gallery_layout.count():
            item = self.gallery_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _select_registration(self, registration_id: str) -> None:
        self.selected_registration_id = registration_id
        for row, entry in enumerate(self.entries):
            if str(entry.get("id", "")) == registration_id:
                self.table.blockSignals(True)
                self.table.selectRow(row)
                self.table.blockSignals(False)
                break
        for card_id, card in self.cards.items():
            card.set_selected(card_id == registration_id)

    def _handle_table_selection(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self.entries):
            return
        self._select_registration(str(self.entries[row].get("id", "")))

    def _sync_selection_from_table(self) -> None:
        if not self.entries:
            self.selected_registration_id = ""
            return
        current_row = self.table.currentRow()
        if current_row < 0 or current_row >= len(self.entries):
            current_row = 0
            self.table.selectRow(current_row)
        self._select_registration(str(self.entries[current_row].get("id", "")))

    def _advance_previews(self) -> None:
        for card in self.cards.values():
            card.advance()

    def _preview_sequence(self, entry: dict[str, Any]) -> np.ndarray | None:
        path = self._preview_path(entry)
        if not path or not Path(path).exists():
            return None
        try:
            sequence = np.load(path)
            if sequence.ndim == 3 and sequence.shape[2] >= 3:
                return normalize_sequence_for_playback(sequence)
            if sequence.ndim == 3 and sequence.shape[2] >= 2:
                return sequence[:, :, :2].astype(np.float32)
        except Exception:
            return None
        return None

    def _preview_path(self, entry: dict[str, Any]) -> str:
        ritual = entry.get("ritual", {})
        if isinstance(ritual, dict):
            full = ritual.get("full", {})
            if isinstance(full, dict):
                for key in ("sequence_path", "playback_sequence_path", "normalized_sequence_path"):
                    value = str(full.get(key, ""))
                    if value and Path(value).exists():
                        return value
        metadata_path = str(entry.get("metadata_path", ""))
        if metadata_path and Path(metadata_path).exists():
            try:
                metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
                ritual = metadata.get("ritual", {})
                full = ritual.get("full", {}) if isinstance(ritual, dict) else {}
                if isinstance(full, dict):
                    for key in ("sequence_path", "playback_sequence_path", "normalized_sequence_path"):
                        value = str(full.get(key, ""))
                        if value and Path(value).exists():
                            return value
                value = str(metadata.get("sequence_path", ""))
                if value and Path(value).exists():
                    return value
            except Exception:
                pass
        value = str(entry.get("sequence_path", ""))
        return value if value and Path(value).exists() else ""

    def _frame_text(self, entry: dict[str, Any]) -> str:
        quality = entry.get("quality", {})
        if isinstance(quality, dict) and quality.get("total_frames") is not None:
            return str(quality.get("total_frames"))
        ritual = entry.get("ritual", {})
        if isinstance(ritual, dict):
            frames = ritual.get("full", {}).get("quality", {}).get("frames")
            if frames is not None:
                return str(int(float(frames)))
        return "-"

    def _quality_text(self, entry: dict[str, Any]) -> str:
        quality = entry.get("quality", {})
        if isinstance(quality, dict):
            tracked = quality.get("tracked_frames")
            total = quality.get("total_frames")
            if tracked is not None and total:
                return f"{int(tracked)}/{int(total)} tracked"
        ritual = entry.get("ritual", {})
        if isinstance(ritual, dict):
            confidence = ritual.get("full", {}).get("quality", {}).get("confidence")
            if confidence is not None:
                return f"{float(confidence) * 100.0:.0f}% confidence"
        return "-"

    def _timestamp_text(self, entry: dict[str, Any]) -> str:
        timestamp = str(entry.get("timestamp_utc", ""))
        if not timestamp:
            return "-"
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return parsed.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return timestamp[:16]
