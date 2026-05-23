from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout

from shared.camera_manager import CameraManager
from shared.config import get_nested
from shared.keypoint_smoother import KeypointSmoother
from shared.pose_engine import create_pose_engine
from shared.skeleton_renderer import draw_skeleton
from ui.base_screen import BaseScreen


class LivePoseScreen(BaseScreen):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.camera: CameraManager | None = None
        self.pose_engine = None
        self.frame_count = 0
        self.pose_failed = False
        self.pose_failure_message = ""
        self.no_pose_frames = 0
        self.smoother = KeypointSmoother()

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.setSpacing(18)

        panel = QFrame()
        panel.setObjectName("terminalPanel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(24, 24, 24, 24)
        panel_layout.setSpacing(14)

        self.preview = QLabel("Camera preview inactive")
        self.preview.setMinimumSize(640, 360)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setScaledContents(False)
        self.status_label = self.make_status("Press START to open camera and run pose tracking.")

        start_button = QPushButton("[ START LIVE SKELETON PREVIEW ]")
        stop_button = QPushButton("[ STOP PREVIEW ]")
        start_button.clicked.connect(self.start)
        stop_button.clicked.connect(self.stop)

        panel_layout.addWidget(self.make_title("LIVE SKELETON PREVIEW"))
        panel_layout.addWidget(self.preview, 1)
        panel_layout.addWidget(self.status_label)
        panel_layout.addWidget(start_button)
        panel_layout.addWidget(stop_button)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        root.addWidget(panel, 1)
        self.add_back_button(root)

    def start(self) -> None:
        self.stop()
        self.frame_count = 0
        self.pose_failed = False
        self.pose_failure_message = ""
        self.no_pose_frames = 0
        smoothing_alpha = float(get_nested(self.config, "pose", "smoothing_alpha", default=0.25))
        self.smoother = KeypointSmoother(alpha=smoothing_alpha)

        try:
            camera_config = get_nested(self.config, "camera", default={})
            self.camera = CameraManager.from_dict(camera_config)
            self.camera.open()
        except Exception as exc:
            self.status_label.setText(f"Preview failed: {exc}")
            self.stop()
            return

        try:
            pose_config = get_nested(self.config, "pose", default={})
            self.pose_engine = create_pose_engine(pose_config)
        except Exception as exc:
            self.pose_failed = True
            self.pose_failure_message = str(exc)
            self.pose_engine = None
            self.status_label.setText(f"Camera running. Pose unavailable: {exc}")

        fps = int(get_nested(self.config, "camera", "fps", default=30))
        interval_ms = max(1, int(1000 / max(1, fps)))
        self.timer.start(interval_ms)
        if not self.pose_failed:
            self.status_label.setText("Camera running. Waiting for skeleton...")

    def stop(self) -> None:
        self.timer.stop()
        if self.pose_engine is not None:
            self.pose_engine.close()
            self.pose_engine = None
        self.smoother.reset()
        if self.camera is not None:
            self.camera.close()
            self.camera = None
        self.status_label.setText("Live skeleton preview stopped.")

    def update_frame(self) -> None:
        if self.camera is None:
            return

        try:
            frame = self.camera.read()
            if frame is None:
                self.status_label.setText("No camera frame received.")
                return

            keypoints = {}
            display_frame = frame
            if self.pose_engine is not None and not self.pose_failed:
                try:
                    min_visibility = float(
                        get_nested(self.config, "pose", "min_visibility", default=0.1)
                    )
                    force_draw = bool(
                        get_nested(self.config, "pose", "force_draw", default=True)
                    )
                    debug_full_landmarks = bool(
                        get_nested(self.config, "pose", "debug_full_landmarks", default=True)
                    )
                    keypoints = self.pose_engine.detect(frame)
                    smoothed_keypoints = self.smoother.smooth(keypoints)
                    if debug_full_landmarks:
                        display_frame = self.pose_engine.draw_full_landmarks(frame)
                    display_frame = draw_skeleton(
                        display_frame,
                        smoothed_keypoints,
                        min_visibility=min_visibility,
                        force_draw=force_draw,
                    )
                    keypoints = smoothed_keypoints
                except Exception as exc:
                    self.pose_failed = True
                    self.pose_failure_message = str(exc)
                    self.status_label.setText(f"Pose failed; showing raw camera. {exc}")

            self.preview.setPixmap(self._frame_to_pixmap(display_frame))
            self.frame_count += 1
            if self.pose_failed or self.pose_engine is None:
                detail = self.pose_failure_message or "pose engine unavailable"
                self.status_label.setText(
                    f"Raw camera preview running. Pose inactive: {detail}. "
                    f"Frames: {self.frame_count}"
                )
            else:
                visible_count = sum(
                    1
                    for keypoint in keypoints.values()
                    if keypoint.visibility
                    >= float(get_nested(self.config, "pose", "min_visibility", default=0.1))
                )
                if len(keypoints) == 0:
                    self.no_pose_frames += 1
                    self.status_label.setText(
                        "Pose tracking running, no landmarks yet. "
                        f"Frames: {self.frame_count}. Step back and show full body."
                    )
                else:
                    self.no_pose_frames = 0
                    avg_visibility = self.pose_engine.last_average_visibility
                    self.status_label.setText(
                        f"Tracked keypoints: {visible_count}/12 visible, "
                        f"{len(keypoints)}/12 selected. "
                        f"Avg visibility: {avg_visibility:.2f}. "
                        f"Frames: {self.frame_count}"
                    )
        except Exception as exc:
            self.status_label.setText(f"Frame update failed: {exc}")
            self.stop()

    def closeEvent(self, event) -> None:
        self.stop()
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
