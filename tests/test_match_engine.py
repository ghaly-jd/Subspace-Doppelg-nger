from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from shared.match_engine import find_motion_matches
from shared.pose_schema import KEYPOINT_NAMES, Keypoint
from shared.registration_storage import save_registration


class MatchEngineTests(unittest.TestCase):
    def test_query_matches_saved_registration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._config(root)
            frames = self._frames(frame_count=48)
            save_registration(
                config,
                nickname="Student B",
                avatar="Blue Star",
                motion_type="Wave",
                frames=frames,
                quality={"total_frames": len(frames), "tracked_frames": len(frames)},
            )

            result = find_motion_matches(config, motion_type="Wave", frames=frames)

            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(result["top_matches"][0]["nickname"], "Student B")
            self.assertGreater(result["top_matches"][0]["similarity"], 99.0)
            self.assertTrue(result["accepted"])

    def test_wave_ignores_missing_lower_body(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._config(root)
            registration_frames = self._frames(frame_count=48)
            query_frames = self._frames(frame_count=48, lower_body_visibility=0.0)
            save_registration(
                config,
                nickname="Upper Body Wave",
                avatar="Blue Star",
                motion_type="Wave",
                frames=registration_frames,
                quality={"total_frames": len(registration_frames), "tracked_frames": len(registration_frames)},
            )

            result = find_motion_matches(config, motion_type="Wave", frames=query_frames)

            self.assertEqual(result["top_matches"][0]["nickname"], "Upper Body Wave")
            self.assertGreater(result["top_matches"][0]["similarity"], 99.0)
            self.assertEqual(result["query"]["subspace"]["basis_shape"], [12, 3])
            self.assertTrue(result["accepted"])

    def test_wave_rejects_tiny_unreliable_upper_body(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._config(root)
            registration_frames = self._frames(frame_count=48)
            query_frames = self._frames(frame_count=48, tiny_upper_body=True)
            save_registration(
                config,
                nickname="Student B",
                avatar="Blue Star",
                motion_type="Wave",
                frames=registration_frames,
                quality={"total_frames": len(registration_frames), "tracked_frames": len(registration_frames)},
            )

            result = find_motion_matches(config, motion_type="Wave", frames=query_frames)

            self.assertFalse(result["accepted"])
            self.assertLess(result["top_matches"][0]["similarity"], 70.0)

    def _config(self, root: Path) -> dict:
        return {
            "pose": {"min_visibility": 0.1},
            "subspace": {
                "num_frames": 120,
                "dimension_k": 5,
                "similarity_alpha": 1.5,
                "smooth_window": 5,
            },
            "motion_profiles": {
                "wave": {
                    "dimension_k": 3,
                    "similarity_alpha": 0.35,
                    "min_accept_similarity": 70,
                }
            },
            "data": {
                "registrations_dir": str(root / "registrations"),
                "sessions_dir": str(root / "sessions"),
                "database_path": str(root / "database.json"),
            },
        }

    def _frames(
        self,
        *,
        frame_count: int,
        lower_body_visibility: float = 0.9,
        tiny_upper_body: bool = False,
    ) -> list[dict[str, Keypoint]]:
        frames = []
        base_points = {
            "left_shoulder": (0.42, 0.35),
            "right_shoulder": (0.58, 0.35),
            "left_elbow": (0.36, 0.46),
            "right_elbow": (0.64, 0.46),
            "left_wrist": (0.32, 0.58),
            "right_wrist": (0.68, 0.58),
            "left_hip": (0.45, 0.65),
            "right_hip": (0.55, 0.65),
            "left_knee": (0.44, 0.82),
            "right_knee": (0.56, 0.82),
            "left_ankle": (0.43, 0.95),
            "right_ankle": (0.57, 0.95),
        }
        for frame_index in range(frame_count):
            phase = frame_index / 7.0
            frame = {}
            for keypoint_index, name in enumerate(KEYPOINT_NAMES):
                visibility = 0.9
                base_x, base_y = base_points[name]
                x = base_x
                y = base_y
                if name == "right_wrist":
                    x += 0.06 * math.sin(phase)
                    y += 0.03 * math.cos(phase)
                if name == "right_elbow":
                    x += 0.025 * math.sin(phase)
                    y += 0.015 * math.cos(phase)
                if name in ("left_wrist", "left_elbow"):
                    x += 0.01 * math.sin(phase + 0.5)
                if tiny_upper_body and name in (
                    "left_shoulder",
                    "right_shoulder",
                    "left_elbow",
                    "right_elbow",
                    "left_wrist",
                    "right_wrist",
                ):
                    x = 0.5 + 0.004 * math.sin(phase + keypoint_index * 0.2)
                    y = 0.5 + 0.004 * math.cos(phase + keypoint_index * 0.2)
                if name in (
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle",
                ):
                    visibility = lower_body_visibility
                frame[name] = Keypoint(
                    name=name,
                    x=x,
                    y=y,
                    visibility=visibility,
                )
            frames.append(frame)
        return frames


if __name__ == "__main__":
    unittest.main()
