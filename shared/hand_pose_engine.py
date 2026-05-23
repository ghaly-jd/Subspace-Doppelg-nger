from __future__ import annotations

from dataclasses import dataclass

from shared.pose_schema import HAND_LANDMARK_NAMES, HandKeypoints, Keypoint


@dataclass(frozen=True)
class HandEngineConfig:
    enabled: bool = False
    engine: str = "none"
    max_num_hands: int = 2
    min_detection_confidence: float = 0.45
    min_tracking_confidence: float = 0.45


class NoOpHandEngine:
    def detect(self, _bgr_frame) -> HandKeypoints:
        return {}

    def close(self) -> None:
        return


class MediaPipeHandsEngine:
    def __init__(self, config: HandEngineConfig) -> None:
        import mediapipe as mp

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_num_hands,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

    def detect(self, bgr_frame) -> HandKeypoints:
        import cv2

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb_frame)
        if not result.multi_hand_landmarks:
            return {}

        output: HandKeypoints = {}
        handedness = result.multi_handedness or []
        for hand_index, landmarks in enumerate(result.multi_hand_landmarks):
            side = self._side_for_hand(handedness, hand_index)
            for landmark_index, name in enumerate(HAND_LANDMARK_NAMES):
                landmark = landmarks.landmark[landmark_index]
                output[f"{side}_{name}"] = Keypoint(
                    name=f"{side}_{name}",
                    x=float(landmark.x),
                    y=float(landmark.y),
                    visibility=float(getattr(landmark, "visibility", 1.0) or 1.0),
                )
        return output

    def close(self) -> None:
        self._hands.close()

    def _side_for_hand(self, handedness, hand_index: int) -> str:
        if hand_index >= len(handedness):
            return "left" if hand_index == 0 else "right"
        classifications = getattr(handedness[hand_index], "classification", [])
        if not classifications:
            return "left" if hand_index == 0 else "right"
        label = str(classifications[0].label).lower()
        return "left" if label == "left" else "right"


def create_hand_engine(config: dict):
    engine_config = HandEngineConfig(
        enabled=bool(config.get("enabled", False)),
        engine=str(config.get("engine", "none")).lower(),
        max_num_hands=int(config.get("max_num_hands", 2)),
        min_detection_confidence=float(config.get("min_detection_confidence", 0.45)),
        min_tracking_confidence=float(config.get("min_tracking_confidence", 0.45)),
    )
    if not engine_config.enabled:
        return NoOpHandEngine()
    if engine_config.engine == "mediapipe":
        return MediaPipeHandsEngine(engine_config)
    if engine_config.engine == "none":
        return NoOpHandEngine()
    raise ValueError(f"Unsupported hand engine: {engine_config.engine}")
