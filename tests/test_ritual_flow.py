from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from shared.dtw_tools import speed_curve_dtw_similarity
from shared.match_engine import find_motion_matches
from shared.pose_schema import HAND_KEYPOINT_NAMES, KEYPOINT_NAMES, Keypoint
from shared.registration_storage import save_registration
from shared.ritual_match_engine import find_ritual_matches, load_ritual_candidates
from shared.ritual_schema import RITUAL_MOTION_TYPE, ritual_phases_from_config
from shared.ritual_segmenter import segment_ritual_sequence


class RitualFlowTests(unittest.TestCase):
    def test_ritual_registration_and_query_returns_phase_and_style_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._config(root)
            frames = self._frames(frame_count=180)
            save_registration(
                config,
                nickname="Student B",
                avatar="Blue Star",
                motion_type=RITUAL_MOTION_TYPE,
                frames=frames,
                quality={"total_frames": len(frames), "tracked_frames": len(frames)},
            )

            result = find_motion_matches(config, motion_type=RITUAL_MOTION_TYPE, frames=frames)

            self.assertTrue(result["ritual_result"])
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(result["overall_match"]["nickname"], "Student B")
            self.assertGreater(result["overall_match"]["similarity"], 95.0)
            self.assertEqual(len(result["phase_matches"]), 9)
            self.assertIn("velocity", result["style_matches"])
            self.assertTrue(result["accepted"])
            playback_path = result["query"]["ritual"]["full"]["playback_sequence_path"]
            math_path = result["query"]["ritual"]["full"]["normalized_sequence_path"]
            playback_sequence = np.load(playback_path)
            self.assertEqual(playback_sequence.shape[0], len(frames))
            self.assertEqual(np.load(math_path).shape[0], 120)
            center_x = playback_sequence.mean(axis=1)[:, 0]
            self.assertGreater(float(center_x.max() - center_x.min()), 0.1)

    def test_ritual_candidate_loader_ignores_legacy_wave_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._config(root)
            frames = self._frames(frame_count=90)
            save_registration(
                config,
                nickname="Wave Only",
                avatar="Blue Star",
                motion_type="Wave",
                frames=frames,
                quality={"total_frames": len(frames), "tracked_frames": len(frames)},
            )

            self.assertEqual(load_ritual_candidates(config), [])

    def test_registration_comparison_can_exclude_new_self_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._config(root)
            first_frames = self._frames(frame_count=160)
            second_frames = self._frames(frame_count=160, phase_offset=0.35)
            save_registration(
                config,
                nickname="Student A",
                avatar="Blue Star",
                motion_type=RITUAL_MOTION_TYPE,
                frames=first_frames,
                quality={"total_frames": len(first_frames), "tracked_frames": len(first_frames)},
            )
            metadata = save_registration(
                config,
                nickname="Student B",
                avatar="Lab Ghost",
                motion_type=RITUAL_MOTION_TYPE,
                frames=second_frames,
                quality={"total_frames": len(second_frames), "tracked_frames": len(second_frames)},
            )

            result = find_ritual_matches(
                config,
                frames=second_frames,
                exclude_ids={metadata["id"]},
            )

            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(result["overall_match"]["nickname"], "Student A")
            self.assertNotEqual(result["overall_match"]["id"], metadata["id"])

    def test_ritual_registration_can_optionally_store_hand_landmarks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._config(root)
            frames = self._frames(frame_count=60)
            hand_frames = self._hand_frames(frame_count=60)

            metadata = save_registration(
                config,
                nickname="Optional Hands",
                avatar="Blue Star",
                motion_type=RITUAL_MOTION_TYPE,
                frames=frames,
                quality={"total_frames": len(frames), "tracked_frames": len(frames)},
                hand_frames=hand_frames,
            )

            self.assertIn("hand_sequence_path", metadata)
            hand_sequence = np.load(metadata["hand_sequence_path"])
            self.assertEqual(hand_sequence.shape, (60, len(HAND_KEYPOINT_NAMES), 3))

    def test_segmenter_creates_one_segment_per_phase(self) -> None:
        frames = self._frames(frame_count=90)
        sequence = self._sequence_from_frames(frames)
        phases = ritual_phases_from_config(self._config(Path("/tmp")))

        segments = segment_ritual_sequence(sequence, phases)

        self.assertEqual(len(segments), 9)
        self.assertEqual(segments[0].start_frame, 0)
        self.assertEqual(segments[-1].end_frame, 90)
        self.assertTrue(all(segment.end_frame > segment.start_frame for segment in segments))

    def test_segmenter_records_event_centers(self) -> None:
        frames = self._frames(frame_count=145)
        sequence = self._sequence_from_frames(frames)
        phases = ritual_phases_from_config(self._config(Path("/tmp")))

        segments = segment_ritual_sequence(sequence, phases, event_based=True)

        self.assertTrue(all(segment.segmentation_method == "event" for segment in segments))
        self.assertTrue(all(segment.event_center_frame is not None for segment in segments))

    def test_speed_curve_dtw_tolerates_tempo_changes(self) -> None:
        slow_curve = [0.0, 0.4, 1.0, 0.4, 0.0, 0.4, 1.0, 0.4, 0.0]
        fast_curve = [0.0, 1.0, 0.0, 1.0, 0.0]
        flat_curve = [0.0 for _ in slow_curve]

        self.assertGreater(speed_curve_dtw_similarity(slow_curve, fast_curve), 70.0)
        self.assertLess(speed_curve_dtw_similarity(slow_curve, flat_curve), 70.0)

    def test_required_phase_gate_can_reject_low_activity_wave(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._config(root)
            config["ritual_quality"] = {
                "full_min_confidence": 0.1,
                "phase_min_confidence": 0.1,
                "enforce_required_phase_gates": True,
                "required_phases": ["wave_hello"],
                "per_phase": {
                    "wave_hello": {
                        "min_wrist_activity": 999.0,
                    },
                },
            }
            frames = self._frames(frame_count=180)
            save_registration(
                config,
                nickname="Still Wave",
                avatar="Blue Star",
                motion_type=RITUAL_MOTION_TYPE,
                frames=frames,
                quality={"total_frames": len(frames), "tracked_frames": len(frames)},
            )

            result = find_motion_matches(config, motion_type=RITUAL_MOTION_TYPE, frames=frames)

            self.assertFalse(result["accepted"])
            self.assertIn("phase", result["no_match_reason"].lower())
            self.assertFalse(result["overall_match"]["phase_scores"]["wave_hello"]["gate_passed"])

    def _config(self, root: Path) -> dict:
        return {
            "pose": {"min_visibility": 0.1},
            "subspace": {
                "num_frames": 120,
                "dimension_k": 5,
                "similarity_alpha": 1.5,
                "smooth_window": 3,
            },
            "ritual": {
                "name": RITUAL_MOTION_TYPE,
                "min_tracked_frames": 20,
                "min_accept_similarity": 55,
            },
            "data": {
                "registrations_dir": str(root / "registrations"),
                "sessions_dir": str(root / "sessions"),
                "database_path": str(root / "database.json"),
            },
        }

    def _sequence_from_frames(self, frames: list[dict[str, Keypoint]]):
        from shared.registration_storage import keypoint_frames_to_array

        return keypoint_frames_to_array(frames)

    def _frames(
        self,
        *,
        frame_count: int,
        phase_offset: float = 0.0,
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
            progress = frame_index / max(1, frame_count - 1)
            phase = frame_index / 6.0 + phase_offset
            frame = {}
            lateral = 0.04 * math.sin(progress * math.tau * 2.0)
            squat = 0.05 * max(0.0, math.sin(progress * math.tau * 3.0))
            for keypoint_index, name in enumerate(KEYPOINT_NAMES):
                base_x, base_y = base_points[name]
                x = base_x + lateral
                y = base_y + squat
                if name in ("left_wrist", "right_wrist"):
                    x += 0.06 * math.sin(phase + keypoint_index)
                    y += 0.04 * math.cos(phase + keypoint_index)
                if name in ("left_elbow", "right_elbow"):
                    y += 0.03 * math.sin(phase)
                frame[name] = Keypoint(name=name, x=x, y=y, visibility=0.95)
            frames.append(frame)
        return frames

    def _hand_frames(self, *, frame_count: int) -> list[dict[str, Keypoint]]:
        frames = []
        for frame_index in range(frame_count):
            phase = frame_index / 10.0
            frame = {}
            for index, name in enumerate(HAND_KEYPOINT_NAMES):
                side_offset = -0.08 if name.startswith("left_") else 0.08
                finger_offset = (index % 21) * 0.002
                frame[name] = Keypoint(
                    name=name,
                    x=0.5 + side_offset + finger_offset,
                    y=0.45 + 0.02 * math.sin(phase + finger_offset),
                    visibility=0.95,
                )
            frames.append(frame)
        return frames


if __name__ == "__main__":
    unittest.main()
