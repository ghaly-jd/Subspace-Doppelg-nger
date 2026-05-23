from __future__ import annotations

from shared.pose_schema import HAND_SKELETON_EDGES, HandKeypoints, SKELETON_EDGES, PoseKeypoints


def draw_skeleton(
    frame,
    keypoints: PoseKeypoints,
    min_visibility: float = 0.1,
    force_draw: bool = False,
):
    import cv2

    height, width = frame.shape[:2]
    output = frame.copy()

    def to_pixel(name: str):
        keypoint = keypoints.get(name)
        if keypoint is None:
            return None
        if not force_draw and keypoint.visibility < min_visibility:
            return None
        if not 0.0 <= keypoint.x <= 1.0 or not 0.0 <= keypoint.y <= 1.0:
            return None
        return int(keypoint.x * width), int(keypoint.y * height)

    for start_name, end_name in SKELETON_EDGES:
        start = to_pixel(start_name)
        end = to_pixel(end_name)
        if start is None or end is None:
            continue
        cv2.line(output, start, end, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.line(output, start, end, (0, 255, 255), 3, cv2.LINE_AA)

    for name in keypoints:
        point = to_pixel(name)
        if point is None:
            continue
        cv2.circle(output, point, 9, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(output, point, 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(output, point, 10, (255, 255, 255), 1, cv2.LINE_AA)

    return output


def draw_hands(
    frame,
    keypoints: HandKeypoints,
    min_visibility: float = 0.1,
):
    import cv2

    height, width = frame.shape[:2]
    output = frame.copy()

    def to_pixel(name: str):
        keypoint = keypoints.get(name)
        if keypoint is None or keypoint.visibility < min_visibility:
            return None
        if not 0.0 <= keypoint.x <= 1.0 or not 0.0 <= keypoint.y <= 1.0:
            return None
        return int(keypoint.x * width), int(keypoint.y * height)

    for start_name, end_name in HAND_SKELETON_EDGES:
        start = to_pixel(start_name)
        end = to_pixel(end_name)
        if start is None or end is None:
            continue
        cv2.line(output, start, end, (30, 30, 30), 5, cv2.LINE_AA)
        cv2.line(output, start, end, (255, 230, 80), 2, cv2.LINE_AA)

    for name in keypoints:
        point = to_pixel(name)
        if point is None:
            continue
        radius = 5 if name.endswith("_tip") else 3
        cv2.circle(output, point, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(output, point, radius, (255, 230, 80), -1, cv2.LINE_AA)

    return output
