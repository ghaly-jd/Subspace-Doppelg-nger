from __future__ import annotations

from shared.pose_schema import Keypoint, PoseKeypoints


class KeypointSmoother:
    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = max(0.0, min(1.0, alpha))
        self._previous: PoseKeypoints = {}

    def reset(self) -> None:
        self._previous = {}

    def smooth(self, keypoints: PoseKeypoints) -> PoseKeypoints:
        if not keypoints:
            return {}

        smoothed: PoseKeypoints = {}
        for name, current in keypoints.items():
            previous = self._previous.get(name)
            if previous is None:
                smoothed[name] = current
                continue

            smoothed[name] = Keypoint(
                name=name,
                x=self._blend(previous.x, current.x),
                y=self._blend(previous.y, current.y),
                visibility=current.visibility,
            )

        self._previous = smoothed
        return smoothed

    def _blend(self, previous: float, current: float) -> float:
        return self.alpha * current + (1.0 - self.alpha) * previous

