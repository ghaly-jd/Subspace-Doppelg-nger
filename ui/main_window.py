from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from shared.config import get_nested
from ui.archive_screen import ArchiveScreen
from ui.explanation_screen import ExplanationScreen
from ui.home_screen import HomeScreen
from ui.live_pose_screen import LivePoseScreen
from ui.match_screen import MatchScreen
from ui.register_screen import RegisterScreen
from ui.result_screen import ResultScreen
from ui.ritual_replay_screen import RitualReplayScreen
from ui.subspace_galaxy_screen import SubspaceGalaxyScreen


class MainWindow(QMainWindow):
    def __init__(self, config: dict[str, Any], config_path: str) -> None:
        super().__init__()
        self.config = config
        self.config_path = config_path

        self.setWindowTitle("The Grassmann Mirror")
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home_screen = HomeScreen()
        self.register_screen = RegisterScreen(config)
        self.match_screen = MatchScreen(config)
        self.archive_screen = ArchiveScreen(config)
        self.result_screen = ResultScreen()
        self.replay_screen = RitualReplayScreen()
        self.galaxy_screen = SubspaceGalaxyScreen()
        self.explanation_screen = ExplanationScreen()
        self.live_pose_screen = LivePoseScreen(config)

        for screen in (
            self.home_screen,
            self.register_screen,
            self.match_screen,
            self.archive_screen,
            self.result_screen,
            self.replay_screen,
            self.galaxy_screen,
            self.explanation_screen,
            self.live_pose_screen,
        ):
            self.stack.addWidget(screen)

        self._connect_navigation()
        self._apply_initial_geometry()

    def show_for_runtime(self) -> None:
        self._apply_initial_geometry()
        mode = self._display_mode()
        if mode == "fullscreen":
            self.showFullScreen()
        elif mode == "maximized":
            self.showMaximized()
        else:
            self.show()

    def _connect_navigation(self) -> None:
        self.home_screen.register_requested.connect(self.show_register)
        self.home_screen.match_requested.connect(self.show_match)
        self.home_screen.archive_requested.connect(self.show_archive)
        self.home_screen.explain_requested.connect(self.show_explanation)
        self.home_screen.live_pose_requested.connect(self.show_live_pose)
        self.register_screen.live_pose_requested.connect(self.show_live_pose)
        self.register_screen.registration_completed.connect(self.show_match_result)
        self.match_screen.live_pose_requested.connect(self.show_live_pose)
        self.match_screen.match_completed.connect(self.show_match_result)
        self.result_screen.replay_requested.connect(self.show_replay_result)
        self.result_screen.galaxy_requested.connect(self.show_subspace_galaxy)
        self.replay_screen.galaxy_requested.connect(self.show_subspace_galaxy)

        for screen in (
            self.register_screen,
            self.match_screen,
            self.archive_screen,
            self.result_screen,
            self.replay_screen,
            self.galaxy_screen,
            self.explanation_screen,
            self.live_pose_screen,
        ):
            screen.back_requested.connect(self.show_home)

    def _apply_initial_geometry(self) -> None:
        screen_geometry = self._target_screen_geometry()
        requested_width = int(get_nested(self.config, "ui", "screen_width", default=1280))
        requested_height = int(get_nested(self.config, "ui", "screen_height", default=720))
        window_scale = float(get_nested(self.config, "ui", "window_scale", default=0.95))
        window_scale = max(0.5, min(1.0, window_scale))

        max_width = int(screen_geometry.width() * window_scale)
        max_height = int(screen_geometry.height() * window_scale)
        width = min(max(800, min(requested_width, max_width)), screen_geometry.width())
        height = min(max(540, min(requested_height, max_height)), screen_geometry.height())
        self.resize(width, height)
        self.move(self._centered_top_left(screen_geometry, width, height))

    def _display_mode(self) -> str:
        configured = str(get_nested(self.config, "runtime", "display_mode", default="")).strip().lower()
        if configured in {"windowed", "maximized", "fullscreen"}:
            return configured
        if bool(get_nested(self.config, "runtime", "fullscreen", default=False)):
            return "fullscreen"
        return "windowed"

    def _target_screen_geometry(self) -> QRect:
        screens = QApplication.screens()
        screen_index = int(get_nested(self.config, "runtime", "screen_index", default=-1))
        if 0 <= screen_index < len(screens):
            return screens[screen_index].availableGeometry()

        primary = QApplication.primaryScreen()
        if primary is not None:
            return primary.availableGeometry()
        return QRect(0, 0, 1280, 720)

    def _centered_top_left(self, geometry: QRect, width: int, height: int) -> QPoint:
        x = geometry.x() + max(0, (geometry.width() - width) // 2)
        y = geometry.y() + max(0, (geometry.height() - height) // 2)
        return QPoint(x, y)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showNormal()
                self._apply_initial_geometry()
            else:
                self.showFullScreen()
            return
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.showNormal()
            self._apply_initial_geometry()
            return
        super().keyPressEvent(event)

    def show_home(self) -> None:
        self.register_screen.stop_capture()
        self.match_screen.stop_capture()
        self.replay_screen.stop()
        self.live_pose_screen.stop()
        self.stack.setCurrentWidget(self.home_screen)

    def show_register(self) -> None:
        self.stack.setCurrentWidget(self.register_screen)

    def show_match(self) -> None:
        self.stack.setCurrentWidget(self.match_screen)

    def show_archive(self) -> None:
        self.archive_screen.refresh()
        self.stack.setCurrentWidget(self.archive_screen)

    def show_result(self) -> None:
        self.stack.setCurrentWidget(self.result_screen)

    def show_match_result(self, result: dict[str, Any]) -> None:
        self.result_screen.set_result(result)
        if result.get("ritual_result") and result.get("overall_match"):
            self.show_replay_result(result, autoplay=True)
            return
        self.show_result()

    def show_replay_result(self, result: dict[str, Any], *, autoplay: bool = False) -> None:
        self.replay_screen.set_result(result, autoplay=autoplay)
        self.stack.setCurrentWidget(self.replay_screen)

    def show_subspace_galaxy(self, result: dict[str, Any]) -> None:
        self.replay_screen.stop()
        self.galaxy_screen.set_result(result)
        self.stack.setCurrentWidget(self.galaxy_screen)

    def show_explanation(self) -> None:
        self.stack.setCurrentWidget(self.explanation_screen)

    def show_live_pose(self) -> None:
        self.stack.setCurrentWidget(self.live_pose_screen)
