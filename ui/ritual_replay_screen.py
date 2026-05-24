from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
)

from shared.motion_profile import FULL_BODY_PROFILE, WAVE_PROFILE, MotionProfile
from shared.pose_schema import KEYPOINT_NAMES, SKELETON_EDGES
from shared.ritual_processor import normalize_sequence_for_playback
from ui.base_screen import BaseScreen


REPLAY_COLORS = (
    QColor("#7dffe8"),
    QColor("#ffd166"),
    QColor("#ff7aa8"),
)


class RitualReplayScreen(BaseScreen):
    def __init__(self) -> None:
        super().__init__()
        self.result: dict[str, Any] | None = None
        self.replay_items: list[dict[str, Any]] = []
        self.keypoint_names: tuple[str, ...] = KEYPOINT_NAMES
        self.frame_index = 0
        self.sequence_scale = 1.0

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(18)

        panel = QFrame()
        panel.setObjectName("terminalPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(24, 24, 24, 24)
        panel_layout.setSpacing(14)

        self.replay_select = QComboBox()
        self.replay_select.currentIndexChanged.connect(self._load_selected_replay)
        self.compare_select = QComboBox()
        self.compare_select.addItem("2 skeletons", 2)
        self.compare_select.addItem("3 skeletons", 3)
        self.compare_select.currentIndexChanged.connect(self._load_selected_replay)
        self.canvas = QLabel("No replay loaded")
        self.canvas.setMinimumSize(520, 280)
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.play_button = QPushButton("[ PLAY / PAUSE ]")
        self.play_button.clicked.connect(self._toggle_playback)
        self.reset_button = QPushButton("[ RESET ]")
        self.reset_button.clicked.connect(self._reset_playback)
        self.speed_select = QComboBox()
        self.speed_select.addItem("0.5x", 66)
        self.speed_select.addItem("1x", 33)
        self.speed_select.addItem("2x", 16)
        self.speed_select.setCurrentIndex(1)
        self.speed_select.currentIndexChanged.connect(self._update_timer_speed)
        self.status_label = self.make_status("Select an overall or phase replay.")
        self.detail_label = self.make_status("")

        controls = QHBoxLayout()
        controls.setSpacing(10)
        controls.addWidget(self.play_button, 2)
        controls.addWidget(self.reset_button, 1)
        controls.addWidget(self.speed_select, 1)

        panel_layout.addWidget(self.make_title("SKELETON RITUAL REPLAY"))
        panel_layout.addWidget(self.replay_select)
        panel_layout.addWidget(self.compare_select)
        panel_layout.addWidget(self.canvas, 1)
        panel_layout.addLayout(controls)
        panel_layout.addWidget(self.status_label)
        panel_layout.addWidget(self.detail_label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_frame)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(panel)

        root.addWidget(scroll, 1)
        self.add_back_button(root)

    def set_result(self, result: dict[str, Any], *, autoplay: bool = False) -> None:
        self.stop()
        self.result = result
        self.replay_select.clear()
        self._add_replay_options(result)
        self._load_selected_replay()
        if autoplay and self.replay_items:
            self.timer.start(self._timer_interval_ms())

    def stop(self) -> None:
        self.timer.stop()

    def _add_replay_options(self, result: dict[str, Any]) -> None:
        overall = result.get("overall_match") or {}
        query_ritual = result.get("query", {}).get("ritual", {})
        if overall and query_ritual.get("full"):
            self.replay_select.addItem(
                f"Overall: You vs top match(es)",
                {
                    "kind": "overall",
                    "label": "Overall",
                    "query_path": self._playback_path(
                        query_ritual["full"],
                        fallback_sequence_path=result.get("query", {}).get("sequence_path", ""),
                    ),
                    "matches": self._overall_matches(result),
                    "candidate_phase_id": None,
                    "profile": "full_body",
                },
            )

        for phase_id, phase_match in result.get("phase_matches", {}).items():
            query_phase = self._query_phase(result, phase_id)
            if not query_phase:
                continue
            self.replay_select.addItem(
                f"{phase_match.get('label', phase_id)}: You vs phase match(es)",
                {
                    "kind": "phase",
                    "label": phase_match.get("label", phase_id),
                    "query_path": self._playback_path(query_phase),
                    "matches": self._phase_matches_for_replay(result, phase_id, phase_match),
                    "candidate_phase_id": phase_id,
                    "profile": query_phase.get("profile", "full_body"),
                },
            )

    def _load_selected_replay(self) -> None:
        was_playing = self.timer.isActive()
        self.stop()
        option = self.replay_select.currentData()
        if not isinstance(option, dict):
            self.status_label.setText("No replay is available yet.")
            return

        query_path = option.get("query_path", "")
        if not query_path:
            self.status_label.setText("Replay files are missing for this match.")
            return

        try:
            self.keypoint_names = self._keypoint_names_for_profile(str(option.get("profile", "full_body")))
            profile = self._profile_object(str(option.get("profile", "full_body")))
            query_sequence = self._load_playback_sequence(query_path, profile=profile)
            self.replay_items = [
                {
                    "label": "YOU",
                    "sequence": query_sequence,
                    "color": REPLAY_COLORS[0],
                    "match": {},
                }
            ]
            compare_count = int(self.compare_select.currentData() or 2)
            for index, match in enumerate(option.get("matches", [])[: max(1, compare_count - 1)], start=1):
                candidate_path = self._candidate_sequence_path(option, match)
                if not candidate_path:
                    continue
                self.replay_items.append(
                    {
                        "label": f"#{index} {match.get('nickname', 'MATCH')}",
                        "sequence": self._load_playback_sequence(candidate_path, profile=profile),
                        "color": REPLAY_COLORS[min(index, len(REPLAY_COLORS) - 1)],
                        "match": match,
                    }
                )
            if len(self.replay_items) < 2:
                self.status_label.setText("Replay files are missing for this match.")
                self.detail_label.setText("")
                self.replay_items = []
                return
            self.sequence_scale = self._shared_sequence_scale(
                [item["sequence"] for item in self.replay_items]
            )
            self.frame_index = 0
            frame_count = min(item["sequence"].shape[0] for item in self.replay_items)
            self.status_label.setText(
                f"{option.get('label', 'Replay')} - {frame_count} playback frames"
            )
            self.detail_label.setText(self._detail_text(option, self.replay_items[1:]))
            self._draw_frame()
            if was_playing:
                self.timer.start(self._timer_interval_ms())
        except Exception as exc:
            self.status_label.setText(f"Replay failed: {exc}")
            self.detail_label.setText("")

    def _candidate_sequence_path(self, option: dict[str, Any], match: dict[str, Any]) -> str:
        metadata_path = match.get("metadata_path", "")
        if not metadata_path or not Path(metadata_path).exists():
            return ""
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        ritual = metadata.get("ritual", {})
        if option.get("kind") == "overall":
            return self._playback_path(
                ritual.get("full", {}),
                fallback_sequence_path=metadata.get("sequence_path", ""),
            )

        phase_id = option.get("candidate_phase_id")
        for phase in ritual.get("phase_segments", []):
            if phase.get("phase_id") == phase_id:
                return self._playback_path(phase)
        return ""

    def _query_phase(self, result: dict[str, Any], phase_id: str) -> dict[str, Any] | None:
        for phase in result.get("query", {}).get("ritual", {}).get("phase_segments", []):
            if phase.get("phase_id") == phase_id:
                return phase
        return None

    def _playback_path(
        self,
        artifact: dict[str, Any],
        *,
        fallback_sequence_path: str = "",
    ) -> str:
        return str(
            artifact.get("sequence_path")
            or fallback_sequence_path
            or artifact.get("playback_sequence_path")
            or artifact.get("normalized_sequence_path")
            or ""
        )

    def _load_playback_sequence(self, path: str, *, profile: MotionProfile) -> np.ndarray:
        sequence = np.load(path)
        if sequence.ndim == 3 and sequence.shape[2] >= 3:
            return normalize_sequence_for_playback(
                sequence,
                profile=profile,
            )
        return sequence

    def _toggle_playback(self) -> None:
        if self.timer.isActive():
            self.stop()
        else:
            self.timer.start(self._timer_interval_ms())

    def _reset_playback(self) -> None:
        self.frame_index = 0
        self._draw_frame()

    def _update_timer_speed(self) -> None:
        if self.timer.isActive():
            self.timer.start(self._timer_interval_ms())

    def _timer_interval_ms(self) -> int:
        value = self.speed_select.currentData()
        return int(value) if value is not None else 33

    def _advance_frame(self) -> None:
        if len(self.replay_items) < 2:
            self.stop()
            return
        frame_count = min(item["sequence"].shape[0] for item in self.replay_items)
        if frame_count <= 0:
            self.stop()
            return
        self.frame_index = (self.frame_index + 1) % frame_count
        self._draw_frame()

    def _draw_frame(self) -> None:
        if len(self.replay_items) < 2:
            return

        width = max(360, self.canvas.width())
        height = max(240, self.canvas.height())
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor("#030810"))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont("Menlo", 13, QFont.Weight.Bold))

        frame_count = min(item["sequence"].shape[0] for item in self.replay_items)
        frame_index = min(self.frame_index, max(0, frame_count - 1))

        margin = 18
        gap = 18
        panel_count = len(self.replay_items)
        panel_width = (width - margin * 2 - gap * (panel_count - 1)) / panel_count
        panel_height = height - 74
        for index, item in enumerate(self.replay_items):
            rect = QRectF(margin + index * (panel_width + gap), margin, panel_width, panel_height)
            self._draw_panel(
                painter,
                rect,
                label=item["label"],
                color=item["color"],
            )
            self._draw_skeleton_panel(
                painter,
                item["sequence"],
                frame_index=frame_index,
                rect=rect,
                color=item["color"],
            )
        self._draw_progress(painter, width=width, height=height, frame_index=frame_index, frame_count=frame_count)
        painter.end()
        self.canvas.setPixmap(pixmap)

    def _draw_panel(
        self,
        painter: QPainter,
        rect: QRectF,
        *,
        label: str,
        color: QColor,
    ) -> None:
        painter.setPen(QPen(QColor("#164c55"), 1))
        painter.setBrush(QColor("#07121c"))
        painter.drawRoundedRect(rect, 8, 8)

        painter.setPen(QPen(QColor("#103340"), 1))
        grid_step = 48
        x = rect.left() + grid_step
        while x < rect.right():
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += grid_step
        y = rect.top() + grid_step
        while y < rect.bottom():
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += grid_step

        painter.setPen(QPen(color, 2))
        painter.drawText(rect.adjusted(16, 16, -16, -16), Qt.AlignmentFlag.AlignLeft, label)

    def _draw_skeleton_panel(
        self,
        painter: QPainter,
        sequence: np.ndarray,
        *,
        frame_index: int,
        rect: QRectF,
        color: QColor,
    ) -> None:
        if sequence.size == 0:
            return

        current_points = np.asarray(sequence[frame_index], dtype=np.float64)
        center = self._sequence_center(sequence)
        mapped = self._map_points(current_points, rect=rect, center=center)

        self._draw_motion_trails(
            painter,
            sequence,
            frame_index=frame_index,
            rect=rect,
            center=center,
            color=color,
        )

        glow = QColor(color)
        glow.setAlpha(70)
        painter.setPen(QPen(glow, 11))
        for start, end in SKELETON_EDGES:
            if start in mapped and end in mapped:
                painter.drawLine(mapped[start], mapped[end])

        painter.setPen(QPen(color, 5))
        for start, end in SKELETON_EDGES:
            if start in mapped and end in mapped:
                painter.drawLine(mapped[start], mapped[end])

        painter.setPen(QPen(QColor("#eaffff"), 2))
        painter.setBrush(color)
        for name, point in mapped.items():
            radius = 7 if name.endswith("wrist") or name.endswith("ankle") else 5
            painter.drawEllipse(point, radius, radius)

    def _draw_motion_trails(
        self,
        painter: QPainter,
        sequence: np.ndarray,
        *,
        frame_index: int,
        rect: QRectF,
        center: np.ndarray,
        color: QColor,
    ) -> None:
        trail_names = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
        trail_indices = [
            index for index, name in enumerate(self.keypoint_names) if name in trail_names
        ]
        if not trail_indices:
            return
        start = max(0, frame_index - 18)
        for step, index in enumerate(range(start, frame_index)):
            alpha = int(25 + 120 * ((step + 1) / max(1, frame_index - start)))
            trail_color = QColor(color)
            trail_color.setAlpha(alpha)
            painter.setPen(QPen(trail_color, 3))
            mapped_a = self._map_points(sequence[index], rect=rect, center=center)
            mapped_b = self._map_points(sequence[index + 1], rect=rect, center=center)
            for keypoint_index in trail_indices:
                name = self.keypoint_names[keypoint_index]
                if name in mapped_a and name in mapped_b:
                    painter.drawLine(mapped_a[name], mapped_b[name])

    def _draw_progress(
        self,
        painter: QPainter,
        *,
        width: int,
        height: int,
        frame_index: int,
        frame_count: int,
    ) -> None:
        if frame_count <= 0:
            return
        left = 22
        top = height - 42
        bar_width = width - 44
        progress = (frame_index + 1) / frame_count
        painter.setPen(QPen(QColor("#164c55"), 1))
        painter.setBrush(QColor("#07121c"))
        painter.drawRoundedRect(QRectF(left, top, bar_width, 10), 5, 5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#7dffe8"))
        painter.drawRoundedRect(QRectF(left, top, bar_width * progress, 10), 5, 5)
        painter.setPen(QPen(QColor("#8aa4b8"), 1))
        painter.drawText(left, top + 30, f"Frame {frame_index + 1}/{frame_count}")

    def _map_points(
        self,
        points: np.ndarray,
        *,
        rect: QRectF,
        center: np.ndarray,
    ) -> dict[str, QPointF]:
        mapped = {}
        panel_center = QPointF(rect.center().x(), rect.center().y() + rect.height() * 0.08)
        for name, point in zip(self.keypoint_names, points):
            if not np.isfinite(point).all():
                continue
            mapped[name] = QPointF(
                panel_center.x() + float(point[0] - center[0]) * self.sequence_scale,
                panel_center.y() + float(point[1] - center[1]) * self.sequence_scale,
            )
        return mapped

    def _shared_sequence_scale(self, sequences: list[np.ndarray]) -> float:
        combined = np.concatenate([self._centered_points(sequence) for sequence in sequences], axis=0)
        finite = combined[np.isfinite(combined).all(axis=1)]
        if finite.size == 0:
            return 150.0
        span = finite.max(axis=0) - finite.min(axis=0)
        max_span = max(float(span.max()), 0.2)
        panel_width = max(160.0, self.canvas.width() / max(2, len(sequences)) - 64.0)
        panel_height = max(180.0, self.canvas.height() - 120.0)
        return min(panel_width * 0.68, panel_height * 0.72) / max_span

    def _overall_matches(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "nickname": match.get("nickname", "Unknown"),
                "metadata_path": match.get("metadata_path", ""),
                "similarity": match.get("similarity", 0.0),
                "coordinate_similarity": match.get("full_coordinate_similarity", 0.0),
                "velocity_similarity": match.get("full_velocity_similarity", 0.0),
                "mean_angle_degrees": match.get("mean_angle_degrees", 0.0),
                "canonical_angles_degrees": match.get("canonical_angles_degrees", []),
                "velocity_mean_angle_degrees": match.get("full_velocity_mean_angle_degrees", 0.0),
                "velocity_canonical_angles_degrees": match.get("full_velocity_canonical_angles_degrees", []),
                "dtw_similarity": self._average_phase_dtw(match),
            }
            for match in result.get("top_matches", [])
        ]

    def _phase_matches_for_replay(
        self,
        result: dict[str, Any],
        phase_id: str,
        phase_match: dict[str, Any],
    ) -> list[dict[str, Any]]:
        matches = []
        for item in result.get("top_matches", []):
            phase_scores = item.get("phase_scores", {})
            if phase_id not in phase_scores:
                continue
            phase = phase_scores[phase_id]
            matches.append(
                {
                    "nickname": item.get("nickname", "Unknown"),
                    "metadata_path": item.get("metadata_path", ""),
                    "similarity": phase.get("similarity", 0.0),
                    "coordinate_similarity": phase.get("coordinate_similarity", 0.0),
                    "velocity_similarity": phase.get("velocity_similarity", 0.0),
                    "dtw_similarity": phase.get("speed_dtw_similarity", 0.0),
                    "mean_angle_degrees": phase.get("mean_angle_degrees", 0.0),
                    "canonical_angles_degrees": phase.get("canonical_angles_degrees", []),
                    "gate_passed": phase.get("gate_passed", True),
                    "gate_reason": phase.get("gate_reason", ""),
                }
            )
        matches.sort(key=lambda match: float(match.get("similarity", 0.0)), reverse=True)
        if matches:
            return matches
        return [
            {
                "nickname": phase_match.get("nickname", "Unknown"),
                "metadata_path": phase_match.get("metadata_path", ""),
                "similarity": phase_match.get("similarity", 0.0),
                "coordinate_similarity": phase_match.get("coordinate_similarity", 0.0),
                "velocity_similarity": phase_match.get("velocity_similarity", 0.0),
                "dtw_similarity": phase_match.get("speed_dtw_similarity", 0.0),
                "mean_angle_degrees": phase_match.get("mean_angle_degrees", 0.0),
                "canonical_angles_degrees": [],
            }
        ]

    def _detail_text(self, option: dict[str, Any], matches: list[dict[str, Any]]) -> str:
        lines = [f"{option.get('label', 'Replay')} details:"]
        for index, item in enumerate(matches, start=1):
            match = item.get("match", {})
            angles = self._angle_text(match.get("canonical_angles_degrees", []))
            velocity_angles = self._angle_text(match.get("velocity_canonical_angles_degrees", []))
            lines.append(
                f"#{index} {match.get('nickname', 'Unknown')} | "
                f"score {float(match.get('similarity', 0.0)):.1f}% | "
                f"coord {float(match.get('coordinate_similarity', 0.0)):.1f}% | "
                f"velocity {float(match.get('velocity_similarity', 0.0)):.1f}% | "
                f"dtw {float(match.get('dtw_similarity', 0.0)):.1f}%"
            )
            lines.append(
                f"canonical mean {float(match.get('mean_angle_degrees', 0.0)):.1f} deg"
                f" [{angles}]"
                + (
                    f" | velocity angles [{velocity_angles}]"
                    if velocity_angles != "n/a"
                    else ""
                )
            )
            if not match.get("gate_passed", True):
                lines.append(f"gate failed: {match.get('gate_reason', 'phase quality')}")
        return "\n".join(lines)

    def _angle_text(self, angles: Any) -> str:
        if not isinstance(angles, list) or not angles:
            return "n/a"
        return ", ".join(f"{float(angle):.1f}" for angle in angles[:5])

    def _average_phase_dtw(self, match: dict[str, Any]) -> float:
        phase_scores = match.get("phase_scores", {})
        if not isinstance(phase_scores, dict) or not phase_scores:
            return 0.0
        values = [
            float(phase.get("speed_dtw_similarity", 0.0))
            for phase in phase_scores.values()
            if isinstance(phase, dict)
        ]
        return float(np.mean(values)) if values else 0.0

    def _centered_points(self, sequence: np.ndarray) -> np.ndarray:
        points = np.asarray(sequence, dtype=np.float64).reshape(-1, 2)
        finite = points[np.isfinite(points).all(axis=1)]
        if finite.size == 0:
            return points
        center = np.median(finite, axis=0)
        return points - center[None, :]

    def _sequence_center(self, sequence: np.ndarray) -> np.ndarray:
        points = np.asarray(sequence, dtype=np.float64).reshape(-1, 2)
        finite = points[np.isfinite(points).all(axis=1)]
        if finite.size == 0:
            return np.zeros(2, dtype=np.float64)
        return np.median(finite, axis=0)

    def _keypoint_names_for_profile(self, profile: str) -> tuple[str, ...]:
        normalized = "".join(character for character in profile.lower() if character.isalnum())
        if normalized in {"upperbodywave", "upperbody", "wave"}:
            return WAVE_PROFILE.keypoint_names
        return KEYPOINT_NAMES

    def _profile_object(self, profile: str) -> MotionProfile:
        normalized = "".join(character for character in profile.lower() if character.isalnum())
        if normalized in {"upperbodywave", "upperbody", "wave"}:
            return WAVE_PROFILE
        return FULL_BODY_PROFILE
