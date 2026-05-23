from __future__ import annotations

from dataclasses import dataclass

from shared.pose_schema import KEYPOINT_NAMES, PoseKeypoints


@dataclass(frozen=True)
class FramingGuidance:
    title: str
    message: str
    ready: bool
    visible_keypoints: int


def camera_framing_guidance(
    keypoints: PoseKeypoints,
    *,
    min_visibility: float = 0.1,
) -> FramingGuidance:
    visible = {
        name: keypoint
        for name, keypoint in keypoints.items()
        if keypoint.visibility >= min_visibility
    }
    visible_count = sum(1 for name in KEYPOINT_NAMES if name in visible)
    if visible_count < 8:
        return FramingGuidance(
            title="ADJUST FRAME",
            message="Step into view so shoulders, hips, knees, and ankles are visible.",
            ready=False,
            visible_keypoints=visible_count,
        )

    xs = [visible[name].x for name in KEYPOINT_NAMES if name in visible]
    ys = [visible[name].y for name in KEYPOINT_NAMES if name in visible]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    body_width = max_x - min_x
    body_height = max_y - min_y
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    if min_x < 0.04 or max_x > 0.96 or min_y < 0.03 or max_y > 0.98:
        return FramingGuidance(
            title="ADJUST FRAME",
            message="Move toward the center so no limbs touch the edge of the camera view.",
            ready=False,
            visible_keypoints=visible_count,
        )
    if body_height < 0.42:
        return FramingGuidance(
            title="STEP CLOSER",
            message="Your skeleton is small in frame. Step closer while keeping the full body visible.",
            ready=False,
            visible_keypoints=visible_count,
        )
    if body_height > 0.90 or body_width > 0.78:
        return FramingGuidance(
            title="STEP BACK",
            message="Your skeleton is too large in frame. Step back so arms and feet fit.",
            ready=False,
            visible_keypoints=visible_count,
        )
    if abs(center_x - 0.5) > 0.22 or abs(center_y - 0.52) > 0.24:
        return FramingGuidance(
            title="CENTER UP",
            message="Move toward the middle of the preview before the scan starts.",
            ready=False,
            visible_keypoints=visible_count,
        )

    return FramingGuidance(
        title="FRAME READY",
        message="Full body visible. Keep this distance and follow the prompts.",
        ready=True,
        visible_keypoints=visible_count,
    )
