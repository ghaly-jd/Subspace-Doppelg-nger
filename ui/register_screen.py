from __future__ import annotations

import time
from typing import Any

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QApplication,
    QPushButton,
    QVBoxLayout,
)

from shared.camera_manager import CameraManager
from shared.config import get_nested
from shared.framing_guidance import camera_framing_guidance
from shared.hand_pose_engine import create_hand_engine
from shared.keypoint_smoother import KeypointSmoother
from shared.pose_engine import create_pose_engine
from shared.pose_schema import HandKeypoints, PoseKeypoints
from shared.registration_storage import save_registration
from shared.ritual_match_engine import find_ritual_matches
from shared.ritual_schema import (
    is_ritual_motion,
    ritual_motion_label,
    ritual_phases_from_config,
    ritual_total_seconds,
)
from shared.skeleton_renderer import draw_hands, draw_skeleton
from ui.base_screen import BaseScreen


class RegisterScreen(BaseScreen):
    live_pose_requested = pyqtSignal()
    registration_completed = pyqtSignal(dict)

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

        form = QFormLayout()
        self.nickname_input = QLineEdit()
        self.nickname_input.setPlaceholderText("Student B")
        self.nickname_input.textChanged.connect(self._refresh_start_button)
        self.avatar_select = QComboBox()
        self.avatar_select.addItems(["Blue Star", "Lab Ghost", "Crane", "Dragon"])
        self.motion_select = QComboBox()
        self.motion_select.addItems([ritual_motion_label(config), "Wave", "Squat", "Walk", "Free Motion"])
        form.addRow("Nickname", self.nickname_input)
        form.addRow("Avatar", self.avatar_select)
        form.addRow("Mode", self.motion_select)

        self.preview = QLabel("Camera preview inactive")
        self.preview.setMinimumSize(640, 300)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setScaledContents(False)

        self.phase_title = QLabel("READY")
        self.phase_title.setObjectName("phasePromptTitle")
        self.phase_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase_body = QLabel("Enter a nickname, then stand where you can still read this screen.")
        self.phase_body.setObjectName("phasePromptBody")
        self.phase_body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase_body.setWordWrap(True)

        self.start_button = QPushButton("[ START CAPTURE ]")
        self.start_button.clicked.connect(self.start_capture)
        self.cancel_button = QPushButton("[ CANCEL CAPTURE ]")
        self.cancel_button.clicked.connect(self.cancel_capture)
        self.cancel_button.setEnabled(False)
        preview_button = QPushButton("[ OPEN LIVE SKELETON PREVIEW ]")
        preview_button.clicked.connect(self.live_pose_requested.emit)

        self.status_label = self.make_status(
            "Enter a nickname, then capture the guided skeleton-only motion ritual."
        )

        panel_layout.addWidget(self.make_title("REGISTER LAB RITUAL"))
        panel_layout.addLayout(form)
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
        self._refresh_start_button()

    def start_capture(self) -> None:
        nickname = self.nickname_input.text().strip()
        if not nickname:
            self.status_label.setText("Nickname required before capture.")
            self._refresh_start_button()
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
            self.status_label.setText(f"Capture setup failed: {exc}")
            self.stop_capture(reset_preview=False)
            return

        fps = int(get_nested(self.config, "camera", "fps", default=30))
        interval_ms = max(1, int(1000 / max(1, fps)))
        self.timer.start(interval_ms)
        self._set_controls_for_capture(active=True)
        self._set_phase_display("FRAME CHECK", "Stand where your full body is visible.")
        self.status_label.setText("Checking camera framing before recording...")

    def cancel_capture(self) -> None:
        self.stop_capture(reset_preview=True)
        self._set_phase_display("READY", "Capture cancelled. Ready to try again.")
        self.status_label.setText("Capture cancelled. Ready to try again.")

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
            self._set_phase_display("READY", "Enter a nickname, then stand where you can still read this screen.")

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
            self.status_label.setText(f"Capture failed: {exc}")
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
            self._set_phase_display("CAPTURING", "Follow each large prompt.")
            self.status_label.setText("Recording skeleton motion...")
            return
        self._set_phase_display(guidance.title, f"{guidance.message}\nRecording begins in {remaining}.")
        self.status_label.setText(
            f"{guidance.message}\n"
            f"Visible keypoints: {guidance.visible_keypoints}/12. "
            f"Recording begins in {remaining}..."
        )

    def _record_frame(self, keypoints: PoseKeypoints, hand_keypoints: HandKeypoints) -> None:
        self.captured_frames.append(keypoints)
        self.captured_hand_frames.append(hand_keypoints)
        motion_type = self.motion_select.currentText()
        duration = self._capture_duration(motion_type)
        elapsed = time.monotonic() - self.capture_started_at
        visible_frames = sum(1 for frame in self.captured_frames if frame)
        phase_title, phase_body = self._phase_prompt(elapsed, motion_type)
        self._set_phase_display(phase_title, phase_body)
        self.status_label.setText(
            f"{phase_title}\n{phase_body}\n"
            f"Recording skeleton motion... {elapsed:.1f}/{duration:.1f}s "
            f"({visible_frames} tracked frames)"
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
            "Locking tracked joints into a clean time-series.",
            "Skeleton frames captured.\nPreparing matrix construction...",
        )
        self._show_processing_stage(
            "MOTION MATRIX",
            "Centering, scaling, smoothing, and stacking joint coordinates.",
            "Building the T x D skeleton motion matrix...",
        )
        self._show_processing_stage(
            "PCA / SVD",
            "Compressing the ritual into principal motion directions.",
            "Applying SVD and keeping the strongest subspace basis...",
        )
        self._show_processing_stage(
            "GRASSMANN SIGNATURE",
            "Saving full-ritual and per-phase coordinate and velocity subspaces.",
            "Writing skeleton-only archive artifacts...",
        )
        quality = self._quality_summary(frames)
        metadata = save_registration(
            self.config,
            nickname=self.nickname_input.text().strip(),
            avatar=self.avatar_select.currentText(),
            motion_type=motion_type,
            frames=frames,
            quality=quality,
            hand_frames=hand_frames if any(hand_frames) else None,
        )
        self.status_label.setText(
            "Registration saved: "
            f"{metadata['nickname']} / {metadata['motion_type']} / "
            f"{quality['tracked_frames']} tracked frames."
        )
        self._set_phase_display("SAVED", "Registration complete.")
        if is_ritual_motion(motion_type):
            self._show_registration_matches(frames, hand_frames, metadata)

    def _show_registration_matches(
        self,
        frames: list[PoseKeypoints],
        hand_frames: list[HandKeypoints],
        metadata: dict[str, Any],
    ) -> None:
        self._show_processing_stage(
            "ARCHIVE SEARCH",
            "Comparing this new ritual against existing lab signatures.",
            "Registration saved.\nLoading comparable archive entries...",
        )
        self._show_processing_stage(
            "CANONICAL ANGLES",
            "Measuring how close the motion subspaces are on the Grassmann manifold.",
            "Calculating canonical angles, projection distances, and phase scores...",
        )
        self._show_processing_stage(
            "SCORE FUSION",
            "Blending full ritual, phase, velocity, rhythm, joint-angle, and energy cues.",
            "Ranking closest lab motion twins...",
        )
        try:
            result = find_ritual_matches(
                self.config,
                frames=frames,
                top_k=3,
                exclude_ids={str(metadata.get("id", ""))},
                hand_frames=hand_frames if any(hand_frames) else None,
            )
        except Exception as exc:
            self.status_label.setText(f"Registration saved, but comparison failed: {exc}")
            return
        result["registration_source"] = {
            "id": metadata.get("id", ""),
            "nickname": metadata.get("nickname", ""),
            "metadata_path": metadata.get("metadata_path", ""),
        }
        self.registration_completed.emit(result)

    def _quality_summary(self, frames: list[PoseKeypoints]) -> dict[str, Any]:
        min_visibility = float(get_nested(self.config, "pose", "min_visibility", default=0.1))
        tracked_frames = 0
        visible_keypoint_counts: list[int] = []
        for frame in frames:
            if frame:
                tracked_frames += 1
            visible_keypoint_counts.append(
                sum(1 for keypoint in frame.values() if keypoint.visibility >= min_visibility)
            )

        average_visible_keypoints = 0.0
        if visible_keypoint_counts:
            average_visible_keypoints = sum(visible_keypoint_counts) / len(visible_keypoint_counts)

        return {
            "total_frames": len(frames),
            "tracked_frames": tracked_frames,
            "average_visible_keypoints": round(average_visible_keypoints, 2),
            "min_visibility": min_visibility,
        }

    def _refresh_start_button(self) -> None:
        enabled = bool(self.nickname_input.text().strip()) and self.state == "idle"
        self.start_button.setEnabled(enabled)

    def _set_controls_for_capture(self, *, active: bool) -> None:
        self.nickname_input.setEnabled(not active)
        self.avatar_select.setEnabled(not active)
        self.motion_select.setEnabled(not active)
        self.start_button.setEnabled(not active and bool(self.nickname_input.text().strip()))
        self.cancel_button.setEnabled(active)

    def _capture_duration(self, motion_type: str) -> float:
        if is_ritual_motion(motion_type):
            return ritual_total_seconds(self.config)
        return float(get_nested(self.config, "capture", "registration_seconds", default=6.0))

    def _min_tracked_frames(self, motion_type: str) -> int:
        if is_ritual_motion(motion_type):
            return int(get_nested(self.config, "ritual", "min_tracked_frames", default=300))
        return int(get_nested(self.config, "capture", "min_tracked_frames", default=10))

    def _phase_prompt(self, elapsed: float, motion_type: str) -> tuple[str, str]:
        if not is_ritual_motion(motion_type):
            return "RECORDING", "Perform the selected motion."
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
