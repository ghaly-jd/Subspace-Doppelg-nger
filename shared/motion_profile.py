from __future__ import annotations

from dataclasses import dataclass

from shared.pose_schema import KEYPOINT_NAMES


@dataclass(frozen=True)
class MotionProfile:
    name: str
    keypoint_names: tuple[str, ...]
    center_keypoint_names: tuple[str, ...]
    scale_from_keypoint_names: tuple[str, ...]
    scale_to_keypoint_names: tuple[str, ...]


FULL_BODY_PROFILE = MotionProfile(
    name="full_body",
    keypoint_names=KEYPOINT_NAMES,
    center_keypoint_names=("left_hip", "right_hip"),
    scale_from_keypoint_names=("left_shoulder", "right_shoulder"),
    scale_to_keypoint_names=("left_hip", "right_hip"),
)

WAVE_PROFILE = MotionProfile(
    name="upper_body_wave",
    keypoint_names=(
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ),
    center_keypoint_names=("left_shoulder", "right_shoulder"),
    scale_from_keypoint_names=("left_shoulder",),
    scale_to_keypoint_names=("right_shoulder",),
)


def profile_for_motion(motion_type: str) -> MotionProfile:
    normalized = "".join(character for character in motion_type.lower() if character.isalnum())
    if normalized == "wave":
        return WAVE_PROFILE
    return FULL_BODY_PROFILE
