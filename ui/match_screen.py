from __future__ import annotations

import time
from typing import Any

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QComboBox, QFrame, QLabel, QPushButton, QVBoxLayout

from shared.camera_manager import CameraManager
from shared.config import get_nested
from shared.framing_guidance import camera_framing_guidance
from shared.hand_pose_engine import create_hand_engine
from shared.keypoint_smoother import KeypointSmoother
from shared.match_engine import find_motion_matches, load_registered_candidates
from shared.pose_engine import create_pose_engine
from shared.pose_schema import HandKeypoints, PoseKeypoints
from shared.ritual_match_engine import load_ritual_candidates
from shared.ritual_schema import (
    is_ritual_motion,
    ritual_motion_label,
    ritual_phases_from_config,
    ritual_total_seconds,
)
from shared.skeleton_renderer import draw_hands, draw_skeleton
from ui.base_screen import BaseScreen


class MatchScreen(BaseScreen):
    live_pose_requested = pyqtSignal()
    match_completed = pyqtSignal(dict)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.camera: CameraManager | None = None
        self.pose_engine = None
        self.hand_engine = None
        self.smoother = KeypointSmoother()
        self.captured_frames: list[PoseKeypoints] = []
        self.captured_hand_frames: list[HandKeypoints] = []
        self.capture_started_at = 0.0
        self.countdown_started_at = 0.0
        self.state = "idle"

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(18)

        panel = QFrame()
        panel.setObjectName("terminalPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(24, 24, 24, 24)
        panel_layout.setSpacing(14)

        self.motion_select = QComboBox()
        self.motion_select.addItems([ritual_motion_label(config), "Wave", "Squat", "Walk", "Free Motion"])
        self.motion_select.currentTextChanged.connect(
            lambda _text: self._update_candidate_status()
        )

        self.preview = QLabel("Camera preview inactive")
        self.preview.setMinimumSize(640, 300)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setScaledContents(False)

        self.phase_title = QLabel("READY")
        self.phase_title.setObjectName("phasePromptTitle")
        self.phase_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase_body = QLabel("Stand where you can still read this screen.")
        self.phase_body.setObjectName("phasePromptBody")
        self.phase_body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase_body.setWordWrap(True)

        self.start_button = QPushButton("[ START MATCH SCAN ]")
        self.start_button.clicked.connect(self.start_capture)
        self.cancel_button = QPushButton("[ CANCEL SCAN ]")
        self.cancel_button.clicked.connect(self.cancel_capture)
        self.cancel_button.setEnabled(False)
        preview_button = QPushButton("[ OPEN LIVE SKELETON PREVIEW ]")
        preview_button.clicked.connect(self.live_pose_requested.emit)

        self.status_label = self.make_status("")

        panel_layout.addWidget(self.make_title("FIND YOUR LAB TWIN"))
        panel_layout.addWidget(self.motion_select)
        panel_layout.addWidget(self.phase_title)
        panel_layout.addWidget(self.phase_body)
        panel_layout.addWidget(self.preview, 1)
        panel_layout.addWidget(self.start_button)
        panel_layout.addWidget(self.cancel_button)
        panel_layout.addWidget(preview_button)
        panel_layout.addWidget(self.status_label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_capture_frame)

        root.addWidget(panel, 1)
        self.add_back_button(root)
        self._update_candidate_status()

    def start_capture(self) -> None:
        motion_type = self.motion_select.currentText()
        candidate_count = self._candidate_count(motion_type)
        if candidate_count == 0:
            self.status_label.setText(
                f"No registered {motion_type} signature(s) yet. Register one first."
            )
            return

        self.stop_capture(reset_preview=False)
        self.captured_frames = []
        self.captured_hand_frames = []
        self.countdown_started_at = time.monotonic()
        self.capture_started_at = 0.0
        self.state = "countdown"

        smoothing_alpha = float(get_nested(self.config, "pose", "smoothing_alpha", default=0.25))
        self.smoother = KeypointSmoother(alpha=smoothing_alpha)

        try:
            camera_config = get_nested(self.config, "camera", default={})
            self.camera = CameraManager.from_dict(camera_config)
            self.camera.open()
            pose_config = get_nested(self.config, "pose", default={})
            self.pose_engine = create_pose_engine(pose_config)
            hand_config = get_nested(self.config, "hands", default={})
            self.hand_engine = create_hand_engine(hand_config)
        except Exception as exc:
            self.status_label.setText(f"Match scan setup failed: {exc}")
            self.stop_capture(reset_preview=False)
            return

        fps = int(get_nested(self.config, "camera", "fps", default=30))
        interval_ms = max(1, int(1000 / max(1, fps)))
        self.timer.start(interval_ms)
        self._set_controls_for_capture(active=True)
        self._set_phase_display("FRAME CHECK", "Stand where your full body is visible.")
        self.status_label.setText("Checking camera framing before match scan...")

    def cancel_capture(self) -> None:
        self.stop_capture(reset_preview=True)
        self._set_phase_display("READY", "Match scan cancelled. Ready to try again.")
        self._update_candidate_status(prefix="Match scan cancelled. ")

    def stop_capture(self, *, reset_preview: bool = True) -> None:
        self.timer.stop()
        if self.pose_engine is not None:
            self.pose_engine.close()
            self.pose_engine = None
        if self.hand_engine is not None:
            self.hand_engine.close()
            self.hand_engine = None
        if self.camera is not None:
            self.camera.close()
            self.camera = None
        self.smoother.reset()
        self.state = "idle"
        self._set_controls_for_capture(active=False)
        if reset_preview:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Camera preview inactive")
            self._set_phase_display("READY", "Stand where you can still read this screen.")

    def update_capture_frame(self) -> None:
        if self.camera is None or self.pose_engine is None:
            return

        try:
            frame = self.camera.read()
            if frame is None:
                self.status_label.setText("No camera frame received.")
                return

            raw_keypoints = self.pose_engine.detect(frame)
            hand_keypoints = self.hand_engine.detect(frame) if self.hand_engine is not None else {}
            keypoints = self.smoother.smooth(raw_keypoints)
            display_frame = self.pose_engine.draw_full_landmarks(frame)
            display_frame = draw_skeleton(
                display_frame,
                keypoints,
                min_visibility=float(get_nested(self.config, "pose", "min_visibility", default=0.1)),
                force_draw=bool(get_nested(self.config, "pose", "force_draw", default=True)),
            )
            display_frame = draw_hands(
                display_frame,
                hand_keypoints,
                min_visibility=float(get_nested(self.config, "pose", "min_visibility", default=0.1)),
            )
            self.preview.setPixmap(self._frame_to_pixmap(display_frame))

            if self.state == "countdown":
                self._update_countdown(keypoints)
            elif self.state == "recording":
                self._record_frame(keypoints, hand_keypoints)
        except Exception as exc:
            self.status_label.setText(f"Match scan failed: {exc}")
            self.stop_capture(reset_preview=False)

    def _update_countdown(self, keypoints: PoseKeypoints) -> None:
        countdown_seconds = float(get_nested(self.config, "capture", "countdown_seconds", default=3.0))
        elapsed = time.monotonic() - self.countdown_started_at
        remaining = max(0, int(countdown_seconds - elapsed) + 1)
        guidance = camera_framing_guidance(
            keypoints,
            min_visibility=float(get_nested(self.config, "pose", "min_visibility", default=0.1)),
        )
        if elapsed >= countdown_seconds:
            self.state = "recording"
            self.capture_started_at = time.monotonic()
            self._set_phase_display("SCANNING", "Follow each large prompt.")
            self.status_label.setText("Scanning motion...")
            return
        self._set_phase_display(guidance.title, f"{guidance.message}\nMatch scan begins in {remaining}.")
        self.status_label.setText(
            f"{guidance.message}\n"
            f"Visible keypoints: {guidance.visible_keypoints}/12. "
            f"Match scan begins in {remaining}..."
        )

    def _record_frame(self, keypoints: PoseKeypoints, hand_keypoints: HandKeypoints) -> None:
        self.captured_frames.append(keypoints)
        self.captured_hand_frames.append(hand_keypoints)
        motion_type = self.motion_select.currentText()
        duration = self._capture_duration(motion_type)
        elapsed = time.monotonic() - self.capture_started_at
        tracked_frames = sum(1 for frame in self.captured_frames if frame)
        phase_title, phase_body = self._phase_prompt(elapsed, motion_type)
        self._set_phase_display(phase_title, phase_body)
        self.status_label.setText(
            f"{phase_title}\n{phase_body}\n"
            f"Scanning motion... {elapsed:.1f}/{duration:.1f}s "
            f"({tracked_frames} tracked frames)"
        )
        if elapsed >= duration:
            self._finish_capture()

    def _finish_capture(self) -> None:
        frames = list(self.captured_frames)
        hand_frames = list(self.captured_hand_frames)
        motion_type = self.motion_select.currentText()
        self.stop_capture(reset_preview=False)

        min_tracked_frames = self._min_tracked_frames(motion_type)
        tracked_frames = sum(1 for frame in frames if frame)
        if tracked_frames < min_tracked_frames:
            self.status_label.setText(
                f"Retry needed: only {tracked_frames} tracked frames were captured. "
                "Step back so the full body is visible."
            )
            self._set_phase_display("RETRY NEEDED", "Step back so the full body is visible.")
            return

        self._show_processing_stage(
            "SKELETON SEQUENCE",
            "Converting webcam pose tracking into a skeleton time-series.",
            "Skeleton frames captured.\nPreparing normalization...",
        )
        self._show_processing_stage(
            "MOTION MATRIX",
            "Centering, scaling, smoothing, and stacking joint coordinates.",
            "Building the T x D skeleton motion matrix...",
        )
        self._show_processing_stage(
            "PCA / SVD",
            "Compressing the ritual into principal motion directions.",
            "Applying SVD and building coordinate and velocity subspaces...",
        )
        self._show_processing_stage(
            "GRASSMANN SEARCH",
            "Comparing your subspace with the lab archive.",
            "Loading registered rituals and phase artifacts...",
        )
        self._show_processing_stage(
            "CANONICAL ANGLES",
            "Small angles mean the motion subspaces are close.",
            "Calculating canonical angles and projection distances...",
        )
        self._show_processing_stage(
            "STYLE FUSION",
            "Blending phase, velocity, rhythm, joint-angle, and energy cues.",
            "Ranking closest lab motion twins...",
        )
        try:
            result = find_motion_matches(
                self.config,
                motion_type=motion_type,
                frames=frames,
                top_k=3,
                hand_frames=hand_frames if any(hand_frames) else None,
            )
        except Exception as exc:
            self.status_label.setText(f"Match computation failed: {exc}")
            return

        if not result["top_matches"]:
            self.status_label.setText(f"No comparable {motion_type} signatures found.")
            self._set_phase_display("NO MATCH DATA", f"No comparable {motion_type} signatures found.")
            return
        if not result.get("accepted", True):
            best = result["top_matches"][0]
            score = float(best.get("similarity", 0.0))
            self.status_label.setText(
                f"{result.get('no_match_reason', 'No clear match detected')}\n"
                f"Closest weak match: {best['nickname']} ({score:.1f}%)."
            )
            self._set_phase_display("WEAK MATCH", f"Closest: {best['nickname']} ({score:.1f}%).")
            if result.get("ritual_result"):
                self.match_completed.emit(result)
            return

        best = result["top_matches"][0]
        self.status_label.setText(
            f"Lab twin found: {best['nickname']} "
            f"({best['similarity']:.1f}% similarity)."
        )
        self._set_phase_display("LAB TWIN FOUND", f"{best['nickname']} - {best['similarity']:.1f}%")
        self.match_completed.emit(result)

    def _update_candidate_status(self, prefix: str = "") -> None:
        motion_type = self.motion_select.currentText()
        candidate_count = self._candidate_count(motion_type)
        if candidate_count == 0:
            self.status_label.setText(
                f"{prefix}No registered {motion_type} signatures yet."
            )
        else:
            self.status_label.setText(
                f"{prefix}{candidate_count} registered {motion_type} signature(s) ready."
            )

    def _set_controls_for_capture(self, *, active: bool) -> None:
        self.motion_select.setEnabled(not active)
        self.start_button.setEnabled(not active)
        self.cancel_button.setEnabled(active)

    def _candidate_count(self, motion_type: str) -> int:
        if is_ritual_motion(motion_type):
            return len(load_ritual_candidates(self.config))
        return len(load_registered_candidates(self.config, motion_type=motion_type))

    def _capture_duration(self, motion_type: str) -> float:
        if is_ritual_motion(motion_type):
            return ritual_total_seconds(self.config)
        return float(get_nested(self.config, "capture", "query_seconds", default=6.0))

    def _min_tracked_frames(self, motion_type: str) -> int:
        if is_ritual_motion(motion_type):
            return int(get_nested(self.config, "ritual", "min_tracked_frames", default=300))
        return int(get_nested(self.config, "capture", "min_tracked_frames", default=10))

    def _phase_prompt(self, elapsed: float, motion_type: str) -> tuple[str, str]:
        if not is_ritual_motion(motion_type):
            return "SCANNING", "Perform the selected motion."
        cursor = 0.0
        for index, phase in enumerate(ritual_phases_from_config(self.config), start=1):
            cursor += phase.duration_seconds
            if elapsed <= cursor:
                return f"{phase.label.upper()} {index:02d}", phase.prompt
        return "FINALIZING", "Capturing individual motion signature..."

    def _set_phase_display(self, title: str, body: str) -> None:
        self.phase_title.setText(title)
        self.phase_body.setText(body)

    def _show_processing_stage(self, title: str, body: str, status: str) -> None:
        self._set_phase_display(title, body)
        self.status_label.setText(status)
        QApplication.processEvents()
        time.sleep(0.16)

    def closeEvent(self, event) -> None:
        self.stop_capture()
        super().closeEvent(event)

    def _frame_to_pixmap(self, bgr_frame) -> QPixmap:
        import cv2

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        target_size = self.preview.size()
        pixmap = QPixmap.fromImage(image)
        if target_size.width() <= 1 or target_size.height() <= 1:
            return pixmap
        return pixmap.scaled(
            target_size,
            aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
            transformMode=Qt.TransformationMode.SmoothTransformation,
        )
