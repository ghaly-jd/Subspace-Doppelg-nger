from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.pose_schema import KEYPOINT_NAMES


LEFT_SHOULDER = KEYPOINT_NAMES.index("left_shoulder")
RIGHT_SHOULDER = KEYPOINT_NAMES.index("right_shoulder")
LEFT_HIP = KEYPOINT_NAMES.index("left_hip")
RIGHT_HIP = KEYPOINT_NAMES.index("right_hip")
KEYPOINT_INDEX_BY_NAME = {name: index for index, name in enumerate(KEYPOINT_NAMES)}


@dataclass(frozen=True)
class NormalizationSummary:
    input_frames: int
    output_frames: int
    feature_dimensions: int
    interpolated_values: int
    median_torso_scale: float


def normalize_to_motion_matrix(
    sequence: np.ndarray,
    *,
    target_frames: int = 120,
    smooth_window: int = 5,
    min_visibility: float = 0.0,
    keypoint_names: tuple[str, ...] = KEYPOINT_NAMES,
    center_keypoint_names: tuple[str, ...] = ("left_hip", "right_hip"),
    scale_from_keypoint_names: tuple[str, ...] = ("left_shoulder", "right_shoulder"),
    scale_to_keypoint_names: tuple[str, ...] = ("left_hip", "right_hip"),
) -> tuple[np.ndarray, NormalizationSummary]:
    normalized_sequence, summary = normalize_keypoint_sequence(
        sequence,
        target_frames=target_frames,
        smooth_window=smooth_window,
        min_visibility=min_visibility,
        keypoint_names=keypoint_names,
        center_keypoint_names=center_keypoint_names,
        scale_from_keypoint_names=scale_from_keypoint_names,
        scale_to_keypoint_names=scale_to_keypoint_names,
    )
    matrix = normalized_sequence.reshape(normalized_sequence.shape[0], -1)
    summary = NormalizationSummary(
        input_frames=summary.input_frames,
        output_frames=summary.output_frames,
        feature_dimensions=int(matrix.shape[1]),
        interpolated_values=summary.interpolated_values,
        median_torso_scale=summary.median_torso_scale,
    )
    return matrix.astype(np.float32), summary


def normalize_keypoint_sequence(
    sequence: np.ndarray,
    *,
    target_frames: int = 120,
    smooth_window: int = 5,
    min_visibility: float = 0.0,
    keypoint_names: tuple[str, ...] = KEYPOINT_NAMES,
    center_keypoint_names: tuple[str, ...] = ("left_hip", "right_hip"),
    scale_from_keypoint_names: tuple[str, ...] = ("left_shoulder", "right_shoulder"),
    scale_to_keypoint_names: tuple[str, ...] = ("left_hip", "right_hip"),
) -> tuple[np.ndarray, NormalizationSummary]:
    if sequence.ndim != 3 or sequence.shape[1] != len(KEYPOINT_NAMES) or sequence.shape[2] < 2:
        raise ValueError(
            "Expected keypoint sequence with shape "
            f"(frames, {len(KEYPOINT_NAMES)}, 2 or 3); got {sequence.shape}"
        )
    if sequence.shape[0] < 2:
        raise ValueError("At least two frames are required for motion normalization.")
    if target_frames < 2:
        raise ValueError("target_frames must be at least 2.")

    keypoint_indices = _indices_for_names(keypoint_names)
    center_indices = _indices_for_names(center_keypoint_names)
    scale_from_indices = _indices_for_names(scale_from_keypoint_names)
    scale_to_indices = _indices_for_names(scale_to_keypoint_names)

    coords = np.array(sequence[:, :, :2], dtype=np.float64, copy=True)
    if sequence.shape[2] >= 3 and min_visibility > 0.0:
        visibility = sequence[:, :, 2]
        coords[visibility < min_visibility] = np.nan

    missing_before = int(np.isnan(coords).sum())
    coords = _interpolate_missing(coords)
    interpolated_values = missing_before - int(np.isnan(coords).sum())
    coords = np.nan_to_num(coords, nan=0.0)

    center = coords[:, center_indices].mean(axis=1)
    scale_from = coords[:, scale_from_indices].mean(axis=1)
    scale_to = coords[:, scale_to_indices].mean(axis=1)
    scale = np.linalg.norm(scale_from - scale_to, axis=1)
    valid_scales = scale[np.isfinite(scale) & (scale > 1e-6)]
    median_scale = float(np.median(valid_scales)) if len(valid_scales) else 1.0
    scale = np.where(scale > 1e-6, scale, median_scale)

    selected_coords = coords[:, keypoint_indices]
    centered = selected_coords - center[:, None, :]
    normalized = centered / scale[:, None, None]
    smoothed = _moving_average(normalized, window=smooth_window)
    resampled = _resample_sequence(smoothed, target_frames=target_frames)

    summary = NormalizationSummary(
        input_frames=int(sequence.shape[0]),
        output_frames=int(target_frames),
        feature_dimensions=int(resampled.shape[1] * resampled.shape[2]),
        interpolated_values=int(interpolated_values),
        median_torso_scale=median_scale,
    )
    return resampled.astype(np.float32), summary


def _interpolate_missing(coords: np.ndarray) -> np.ndarray:
    filled = np.array(coords, dtype=np.float64, copy=True)
    frame_positions = np.arange(filled.shape[0], dtype=np.float64)

    for keypoint_index in range(filled.shape[1]):
        for axis in range(filled.shape[2]):
            values = filled[:, keypoint_index, axis]
            valid = np.isfinite(values)
            if valid.all():
                continue
            if valid.sum() == 0:
                filled[:, keypoint_index, axis] = 0.0
                continue
            filled[:, keypoint_index, axis] = np.interp(
                frame_positions,
                frame_positions[valid],
                values[valid],
            )
    return filled


def _moving_average(sequence: np.ndarray, *, window: int) -> np.ndarray:
    if window <= 1:
        return sequence
    window = max(1, int(window))
    left_pad = window // 2
    right_pad = window - 1 - left_pad
    padded = np.pad(sequence, ((left_pad, right_pad), (0, 0), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.empty_like(sequence)
    for keypoint_index in range(sequence.shape[1]):
        for axis in range(sequence.shape[2]):
            smoothed[:, keypoint_index, axis] = np.convolve(
                padded[:, keypoint_index, axis],
                kernel,
                mode="valid",
            )
    return smoothed


def _resample_sequence(sequence: np.ndarray, *, target_frames: int) -> np.ndarray:
    if sequence.shape[0] == target_frames:
        return sequence

    source_positions = np.linspace(0.0, 1.0, sequence.shape[0])
    target_positions = np.linspace(0.0, 1.0, target_frames)
    output = np.empty((target_frames, sequence.shape[1], sequence.shape[2]), dtype=np.float64)
    for keypoint_index in range(sequence.shape[1]):
        for axis in range(sequence.shape[2]):
            output[:, keypoint_index, axis] = np.interp(
                target_positions,
                source_positions,
                sequence[:, keypoint_index, axis],
            )
    return output


def _indices_for_names(names: tuple[str, ...]) -> list[int]:
    try:
        return [KEYPOINT_INDEX_BY_NAME[name] for name in names]
    except KeyError as exc:
        raise ValueError(f"Unknown keypoint name: {exc.args[0]}") from exc
