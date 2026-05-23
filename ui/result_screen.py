from __future__ import annotations

from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout

from ui.base_screen import BaseScreen


class ResultScreen(BaseScreen):
    replay_requested = pyqtSignal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.result: dict[str, Any] | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(18)

        panel = QFrame()
        panel.setObjectName("terminalPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(24, 24, 24, 24)
        panel_layout.setSpacing(16)

        self.match_label = QLabel("No match result yet.")
        self.match_label.setObjectName("panelTitle")
        self.detail_label = self.make_status("Run a match scan to compute canonical angles.")
        self.rank_label = self.make_status("")

        self.replay_button = QPushButton("[ PLAY SKELETON COMPARISON ]")
        self.replay_button.setEnabled(False)
        self.replay_button.clicked.connect(self._request_replay)
        self.explain_button = QPushButton("[ EXPLAIN MATCH ]")
        self.explain_button.setEnabled(False)

        panel_layout.addWidget(self.make_title("YOUR LAB MOTION TWIN"))
        panel_layout.addWidget(self.match_label)
        panel_layout.addWidget(self.detail_label)
        panel_layout.addWidget(self.rank_label)
        panel_layout.addWidget(self.replay_button)
        panel_layout.addWidget(self.explain_button)
        panel_layout.addStretch(1)

        root.addWidget(panel, 1)
        self.add_back_button(root)

    def set_result(self, result: dict[str, Any]) -> None:
        self.result = result
        if result.get("ritual_result"):
            self._set_ritual_result(result)
            return

        matches = result.get("top_matches", [])
        if not matches:
            self.match_label.setText("No match found.")
            self.detail_label.setText("No comparable registrations were available.")
            self.rank_label.setText("")
            self.replay_button.setEnabled(False)
            return

        best = matches[0]
        self.replay_button.setEnabled(False)
        self.match_label.setText(str(best.get("nickname", "Unknown")))
        angles = best.get("canonical_angles_degrees", [])
        angle_text = ", ".join(f"{float(angle):.1f}" for angle in angles[:5])
        self.detail_label.setText(
            f"Motion: {result.get('motion_type', 'Unknown')}\n"
            f"Similarity: {float(best.get('similarity', 0.0)):.1f}%\n"
            f"Mean canonical angle: {float(best.get('mean_angle_degrees', 0.0)):.1f} deg\n"
            f"Projection distance: {float(best.get('projection_distance', 0.0)):.3f}\n"
            f"Canonical angles: {angle_text}"
        )
        self.rank_label.setText(self._rank_text(matches))

    def _set_ritual_result(self, result: dict[str, Any]) -> None:
        best = result.get("overall_match")
        if not best:
            self.match_label.setText("No lab twin found.")
            self.detail_label.setText("No comparable Motion Ritual registrations were available.")
            self.rank_label.setText("")
            self.replay_button.setEnabled(False)
            return

        self.replay_button.setEnabled(True)
        self.match_label.setText(
            f"OVERALL MOTION TWIN: {best.get('nickname', 'Unknown')} "
            f"- {float(best.get('similarity', 0.0)):.1f}%"
        )
        prefix = ""
        source = result.get("registration_source")
        if isinstance(source, dict):
            prefix = f"REGISTERED: {source.get('nickname', 'New ritual')}\n"
        if not result.get("accepted", True):
            prefix += f"RITUAL QUALITY LOW\n{result.get('no_match_reason', '')}\n"
        angle_text = self._angle_text(best.get("canonical_angles_degrees", []), limit=5)
        velocity_angle_text = self._angle_text(
            best.get("full_velocity_canonical_angles_degrees", []),
            limit=5,
        )
        self.detail_label.setText(
            f"{prefix}"
            f"Weighted overall score: {float(best.get('similarity', 0.0)):.1f}%\n"
            f"Coordinate subspace: {float(best.get('full_coordinate_similarity', 0.0)):.1f}% | "
            f"mean angle {float(best.get('mean_angle_degrees', 0.0)):.1f} deg | "
            f"projection {float(best.get('projection_distance', 0.0)):.3f}\n"
            f"Canonical angles: {angle_text}\n"
            f"Velocity subspace: {float(best.get('full_velocity_similarity', 0.0)):.1f}% | "
            f"mean angle {float(best.get('full_velocity_mean_angle_degrees', 0.0)):.1f} deg | "
            f"projection {float(best.get('full_velocity_projection_distance', 0.0)):.3f}\n"
            f"Velocity angles: {velocity_angle_text}\n"
            f"Average phase score: {float(best.get('average_phase_similarity', 0.0)):.1f}% | "
            f"Rhythm: {float(best.get('rhythm_similarity', 0.0)):.1f}% | "
            f"Joint angles: {float(best.get('joint_angle_similarity', 0.0)):.1f}% | "
            f"Energy/balance: {float(best.get('energy_balance_similarity', 0.0)):.1f}%"
        )
        self.rank_label.setText(self._ritual_rank_text(result))

    def _ritual_rank_text(self, result: dict[str, Any]) -> str:
        lines = ["Phase matches:"]
        phase_matches = result.get("phase_matches", {})
        for phase in phase_matches.values():
            lines.append(
                f"{phase.get('label', 'Phase')} -> "
                f"{phase.get('nickname', 'Unknown')} - "
                f"{float(phase.get('similarity', 0.0)):.1f}%"
                f" | coord {float(phase.get('coordinate_similarity', 0.0)):.1f}%"
                f", angle {float(phase.get('mean_angle_degrees', 0.0)):.1f} deg"
                f", vel {float(phase.get('velocity_similarity', 0.0)):.1f}%"
                f", dtw {float(phase.get('speed_dtw_similarity', 0.0)):.1f}%"
                f", q {float(phase.get('quality_weight', 1.0)):.2f}"
                f"{'' if phase.get('gate_passed', True) else ' | gate failed'}"
            )

        style_matches = result.get("style_matches", {})
        if style_matches:
            lines.append("")
            lines.append("Style matches:")
            for style_name in ("velocity", "rhythm", "energy", "balance", "gesture"):
                style = style_matches.get(style_name)
                if not style:
                    continue
                lines.append(
                    f"{style_name.title()} -> {style.get('nickname', 'Unknown')} "
                    f"- {float(style.get('similarity', 0.0)):.1f}%"
                )

        top_matches = result.get("top_matches", [])
        if top_matches:
            lines.append("")
            lines.append("Overall ranking:")
            for index, match in enumerate(top_matches, start=1):
                lines.append(
                    f"{index}. {match.get('nickname', 'Unknown')} - "
                    f"{float(match.get('similarity', 0.0)):.1f}%"
                )
        return "\n".join(lines)

    def _angle_text(self, angles: Any, *, limit: int) -> str:
        if not isinstance(angles, list) or not angles:
            return "n/a"
        return ", ".join(f"{float(angle):.1f}" for angle in angles[:limit])

    def _request_replay(self) -> None:
        if self.result:
            self.replay_requested.emit(self.result)

    def _rank_text(self, matches: list[dict[str, Any]]) -> str:
        lines = ["Top matches:"]
        for index, match in enumerate(matches, start=1):
            lines.append(
                f"{index}. {match.get('nickname', 'Unknown')} - "
                f"{float(match.get('similarity', 0.0)):.1f}%"
            )
        return "\n".join(lines)
