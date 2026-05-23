from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from shared.pose_schema import KEYPOINT_NAMES, Keypoint, PoseKeypoints

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/sessions/matplotlib").resolve()))


@dataclass(frozen=True)
class PoseEngineConfig:
    model_path: str = "models/mediapipe/pose_landmarker_lite.task"
    delegate: str = "cpu"
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5


class MediaPipePoseEngine:
    landmark_indices = {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }

    def __init__(self, config: PoseEngineConfig) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe pose model not found: {model_path}. "
                "Run `python tools/download_models.py pose-landmarker-lite`."
            )

        self._mp = mp
        self._vision = vision
        delegate = python.BaseOptions.Delegate.CPU
        if config.delegate.lower() == "gpu":
            delegate = python.BaseOptions.Delegate.GPU

        base_options = python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=delegate,
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.min_detection_confidence,
            min_pose_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        self.last_landmark_count = 0
        self.last_selected_count = 0
        self.last_average_visibility = 0.0
        self.last_pose_landmarks = []

    @classmethod
    def from_dict(cls, config: dict) -> "MediaPipePoseEngine":
        engine_config = PoseEngineConfig(
            model_path=str(
                config.get("model_path") or "models/mediapipe/pose_landmarker_lite.task"
            ),
            delegate=str(config.get("delegate", "cpu")),
            min_detection_confidence=float(config.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(config.get("min_tracking_confidence", 0.5)),
            min_pose_presence_confidence=float(
                config.get("min_pose_presence_confidence", 0.5)
            ),
        )
        return cls(engine_config)

    def detect(self, bgr_frame) -> PoseKeypoints:
        import cv2

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.pose_landmarks:
            self.last_landmark_count = 0
            self.last_selected_count = 0
            self.last_average_visibility = 0.0
            self.last_pose_landmarks = []
            return {}

        landmarks = result.pose_landmarks[0]
        self.last_pose_landmarks = landmarks
        self.last_landmark_count = len(landmarks)
        keypoints: PoseKeypoints = {}

        for name in KEYPOINT_NAMES:
            landmark_id = self.landmark_indices[name]
            landmark = landmarks[landmark_id]
            keypoints[name] = Keypoint(
                name=name,
                x=float(landmark.x),
                y=float(landmark.y),
                visibility=float(getattr(landmark, "visibility", 1.0)),
            )

        self.last_selected_count = len(keypoints)
        self.last_average_visibility = sum(
            keypoint.visibility for keypoint in keypoints.values()
        ) / max(1, len(keypoints))
        return keypoints

    def draw_full_landmarks(self, frame):
        if not self.last_pose_landmarks:
            return frame

        import cv2

        connections = (
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
            (11, 23),
            (12, 24),
            (23, 24),
            (23, 25),
            (25, 27),
            (24, 26),
            (26, 28),
            (27, 29),
            (29, 31),
            (28, 30),
            (30, 32),
        )
        output = frame.copy()
        height, width = output.shape[:2]

        def point(index: int):
            landmark = self.last_pose_landmarks[index]
            if not 0.0 <= landmark.x <= 1.0 or not 0.0 <= landmark.y <= 1.0:
                return None
            return int(landmark.x * width), int(landmark.y * height)

        for start_index, end_index in connections:
            start = point(start_index)
            end = point(end_index)
            if start is None or end is None:
                continue
            cv2.line(output, start, end, (255, 120, 80), 1, cv2.LINE_AA)

        for index in range(len(self.last_pose_landmarks)):
            landmark_point = point(index)
            if landmark_point is None:
                continue
            cv2.circle(output, landmark_point, 2, (255, 255, 255), -1, cv2.LINE_AA)

        return output

    def close(self) -> None:
        self._landmarker.close()


class YoloPoseEngine:
    keypoint_indices = {
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        from ultralytics import YOLO

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"YOLO pose model not found: {path}. "
                "Run `python tools/download_models.py yolo11n-pose`."
            )

        self.model = YOLO(str(path))
        self.device = device
        self.last_landmark_count = 0
        self.last_selected_count = 0
        self.last_average_visibility = 0.0
        self.last_pose_landmarks = []

    @classmethod
    def from_dict(cls, config: dict) -> "YoloPoseEngine":
        return cls(
            model_path=str(config.get("model_path") or "models/yolo/yolo11n-pose.pt"),
            device=str(config.get("delegate", "cpu")),
        )

    def detect(self, bgr_frame) -> PoseKeypoints:
        results = self.model.predict(
            bgr_frame,
            imgsz=640,
            device=self.device,
            verbose=False,
        )
        if not results:
            return self._empty()

        result = results[0]
        if result.keypoints is None or result.keypoints.xyn is None:
            return self._empty()

        normalized = result.keypoints.xyn
        confidence = result.keypoints.conf
        if len(normalized) == 0:
            return self._empty()

        person_points = normalized[0].cpu().numpy()
        if confidence is None:
            person_conf = [1.0] * len(person_points)
        else:
            person_conf = confidence[0].cpu().numpy()

        self.last_pose_landmarks = person_points
        self.last_landmark_count = len(person_points)
        keypoints: PoseKeypoints = {}

        for name in KEYPOINT_NAMES:
            index = self.keypoint_indices[name]
            x, y = person_points[index]
            visibility = float(person_conf[index]) if index < len(person_conf) else 1.0
            keypoints[name] = Keypoint(
                name=name,
                x=float(x),
                y=float(y),
                visibility=visibility,
            )

        self.last_selected_count = len(keypoints)
        self.last_average_visibility = sum(
            keypoint.visibility for keypoint in keypoints.values()
        ) / max(1, len(keypoints))
        return keypoints

    def draw_full_landmarks(self, frame):
        if len(self.last_pose_landmarks) == 0:
            return frame

        import cv2

        connections = (
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
        )
        output = frame.copy()
        height, width = output.shape[:2]

        def point(index: int):
            x, y = self.last_pose_landmarks[index]
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                return None
            return int(x * width), int(y * height)

        for start_index, end_index in connections:
            start = point(start_index)
            end = point(end_index)
            if start is None or end is None:
                continue
            cv2.line(output, start, end, (255, 120, 80), 1, cv2.LINE_AA)

        for index in range(len(self.last_pose_landmarks)):
            landmark_point = point(index)
            if landmark_point is None:
                continue
            cv2.circle(output, landmark_point, 2, (255, 255, 255), -1, cv2.LINE_AA)

        return output

    def close(self) -> None:
        return

    def _empty(self) -> PoseKeypoints:
        self.last_landmark_count = 0
        self.last_selected_count = 0
        self.last_average_visibility = 0.0
        self.last_pose_landmarks = []
        return {}


def create_pose_engine(config: dict):
    engine = str(config.get("engine", "yolo")).lower()
    if engine == "mediapipe":
        return MediaPipePoseEngine.from_dict(config)
    if engine == "yolo":
        return YoloPoseEngine.from_dict(config)
    raise ValueError(f"Unsupported pose engine: {engine}")
