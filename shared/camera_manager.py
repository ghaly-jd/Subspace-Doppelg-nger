from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    source: int = 0
    backend: str = "auto"
    width: int = 640
    height: int = 480
    fps: int = 30


class CameraManager:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture = None

    @classmethod
    def from_dict(cls, config: dict) -> "CameraManager":
        camera_config = CameraConfig(
            source=int(config.get("source", 0)),
            backend=str(config.get("backend", "auto")),
            width=int(config.get("width", 640)),
            height=int(config.get("height", 480)),
            fps=int(config.get("fps", 30)),
        )
        return cls(camera_config)

    def open(self) -> None:
        import cv2

        if self._capture is not None:
            return

        backend = self._opencv_backend(cv2)
        if backend is None:
            capture = cv2.VideoCapture(self.config.source)
        else:
            capture = cv2.VideoCapture(self.config.source, backend)

        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open camera source {self.config.source} "
                f"with backend {self.config.backend}"
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        self._capture = capture

    def read(self):
        if self._capture is None:
            self.open()

        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _opencv_backend(self, cv2):
        backend = self.config.backend.lower()
        if backend == "auto":
            return None
        if backend == "avfoundation":
            return cv2.CAP_AVFOUNDATION
        if backend == "dshow":
            return cv2.CAP_DSHOW
        if backend == "msmf":
            return cv2.CAP_MSMF
        return None

    def __enter__(self) -> "CameraManager":
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
