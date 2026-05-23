from __future__ import annotations

from dataclasses import dataclass


KEYPOINT_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

SKELETON_EDGES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float
    y: float
    visibility: float


PoseKeypoints = dict[str, Keypoint]


HAND_LANDMARK_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

HAND_KEYPOINT_NAMES = tuple(
    f"{side}_{name}" for side in ("left", "right") for name in HAND_LANDMARK_NAMES
)

HAND_SKELETON_EDGES = tuple(
    (f"{side}_{start}", f"{side}_{end}")
    for side in ("left", "right")
    for start, end in (
        ("wrist", "thumb_cmc"),
        ("thumb_cmc", "thumb_mcp"),
        ("thumb_mcp", "thumb_ip"),
        ("thumb_ip", "thumb_tip"),
        ("wrist", "index_mcp"),
        ("index_mcp", "index_pip"),
        ("index_pip", "index_dip"),
        ("index_dip", "index_tip"),
        ("wrist", "middle_mcp"),
        ("middle_mcp", "middle_pip"),
        ("middle_pip", "middle_dip"),
        ("middle_dip", "middle_tip"),
        ("wrist", "ring_mcp"),
        ("ring_mcp", "ring_pip"),
        ("ring_pip", "ring_dip"),
        ("ring_dip", "ring_tip"),
        ("wrist", "pinky_mcp"),
        ("pinky_mcp", "pinky_pip"),
        ("pinky_pip", "pinky_dip"),
        ("pinky_dip", "pinky_tip"),
    )
)

HandKeypoints = dict[str, Keypoint]
