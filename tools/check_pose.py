from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.camera_manager import CameraManager
from shared.config import load_config
from shared.pose_engine import create_pose_engine
from shared.skeleton_renderer import draw_skeleton


def main() -> int:
    parser = argparse.ArgumentParser(description="Check MediaPipe pose detection from camera.")
    parser.add_argument(
        "--config",
        default="configs/debug.yaml",
        help="Path to runtime config YAML file.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=90,
        help="Number of camera frames to test.",
    )
    parser.add_argument(
        "--save-frame",
        default="data/sessions/pose_debug_frame.jpg",
        help="Where to save the last debug overlay frame.",
    )
    args = parser.parse_args()

    import cv2

    config = load_config(Path(args.config))
    camera = CameraManager.from_dict(config.get("camera", {}))
    pose_engine = create_pose_engine(config.get("pose", {}))
    save_path = Path(args.save_frame)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    detected_frames = 0
    last_keypoint_count = 0
    last_visibility = 0.0
    last_debug_frame = None

    try:
        camera.open()
        for frame_index in range(args.frames):
            frame = camera.read()
            if frame is None:
                print(f"Frame {frame_index}: no camera frame")
                continue

            keypoints = pose_engine.detect(frame)
            if keypoints:
                detected_frames += 1
                last_keypoint_count = len(keypoints)
                last_visibility = pose_engine.last_average_visibility
                debug_frame = pose_engine.draw_full_landmarks(frame)
                last_debug_frame = draw_skeleton(debug_frame, keypoints, force_draw=True)

            if frame_index % 15 == 0:
                print(
                    f"Frame {frame_index}: selected={len(keypoints)}/12, "
                    f"mediapipe_landmarks={pose_engine.last_landmark_count}, "
                    f"avg_visibility={pose_engine.last_average_visibility:.2f}"
                )
    finally:
        camera.close()
        pose_engine.close()

    if last_debug_frame is not None:
        cv2.imwrite(str(save_path), last_debug_frame)
        print(f"Saved debug overlay frame: {save_path}")

    print(f"Detected pose frames: {detected_frames}/{args.frames}")
    print(f"Last selected keypoints: {last_keypoint_count}/12")
    print(f"Last average visibility: {last_visibility:.2f}")

    if detected_frames == 0:
        print("No pose detected. Stand farther back with full body visible and good lighting.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
