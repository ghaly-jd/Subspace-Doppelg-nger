from __future__ import annotations

import math
import unittest

import numpy as np

from shared.pose_schema import KEYPOINT_NAMES
from shared.skeleton_normalizer import normalize_to_motion_matrix
from shared.subspace import build_motion_subspace, compare_subspaces


class Phase4MathTests(unittest.TestCase):
    def test_normalizer_resamples_to_motion_matrix(self) -> None:
        sequence = self._sequence(frame_count=32)

        matrix, summary = normalize_to_motion_matrix(
            sequence,
            target_frames=120,
            smooth_window=3,
            min_visibility=0.1,
        )

        self.assertEqual(matrix.shape, (120, 24))
        self.assertEqual(summary.input_frames, 32)
        self.assertEqual(summary.output_frames, 120)
        self.assertTrue(np.isfinite(matrix).all())

    def test_identical_subspaces_are_nearly_perfect_match(self) -> None:
        matrix, _ = normalize_to_motion_matrix(self._sequence(frame_count=48))
        model = build_motion_subspace(matrix, dimension_k=5)

        comparison = compare_subspaces(model.basis, model.basis)

        self.assertLess(comparison.mean_angle_degrees, 0.01)
        self.assertAlmostEqual(comparison.similarity, 100.0, places=4)

    def _sequence(self, *, frame_count: int) -> np.ndarray:
        sequence = np.zeros((frame_count, len(KEYPOINT_NAMES), 3), dtype=np.float32)
        for frame_index in range(frame_count):
            phase = frame_index / 6.0
            for keypoint_index, _ in enumerate(KEYPOINT_NAMES):
                side_offset = -0.05 if keypoint_index % 2 == 0 else 0.05
                height_offset = 0.02 * keypoint_index
                sequence[frame_index, keypoint_index] = (
                    0.5 + side_offset + 0.03 * math.sin(phase + keypoint_index),
                    0.3 + height_offset + 0.02 * math.cos(phase),
                    0.9,
                )
        return sequence


if __name__ == "__main__":
    unittest.main()
