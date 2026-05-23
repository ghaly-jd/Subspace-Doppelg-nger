from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from shared.config import get_nested
from shared.motion_features import compute_motion_features, velocity_matrix_from_motion_matrix
from shared.motion_profile import FULL_BODY_PROFILE, WAVE_PROFILE, MotionProfile
from shared.pose_schema import KEYPOINT_NAMES
from shared.ritual_schema import ritual_phases_from_config
from shared.ritual_segmenter import segment_ritual_sequence
from shared.skeleton_normalizer import normalize_keypoint_sequence, normalize_to_motion_matrix
from shared.subspace import build_motion_subspace


def build_ritual_artifacts(
    config: dict[str, Any],
    *,
    sequence: np.ndarray,
    base_path: Path,
    artifact_id: str,
    save_arrays: bool = True,
) -> dict[str, Any]:
    target_frames = int(get_nested(config, "subspace", "num_frames", default=120))
    dimension_k = int(get_nested(config, "subspace", "dimension_k", default=5))
    smooth_window = int(get_nested(config, "subspace", "smooth_window", default=5))
    min_visibility = float(get_nested(config, "pose", "min_visibility", default=0.0))
    phases = ritual_phases_from_config(config)
    event_based_segmentation = bool(
        get_nested(config, "ritual_segmentation", "event_based", default=True)
    )
    event_padding_ratio = float(
        get_nested(config, "ritual_segmentation", "event_padding_ratio", default=0.45)
    )

    full = _process_sequence(
        sequence,
        profile=FULL_BODY_PROFILE,
        target_frames=target_frames,
        dimension_k=dimension_k,
        smooth_window=smooth_window,
        min_visibility=min_visibility,
    )
    full_paths = _artifact_paths(base_path, artifact_id, "full")
    if save_arrays:
        _save_processed_arrays(full_paths, full)

    phase_segments = []
    for segment in segment_ritual_sequence(
        sequence,
        phases,
        event_based=event_based_segmentation,
        event_padding_ratio=event_padding_ratio,
        min_visibility=min_visibility,
    ):
        profile = _profile_for_phase(segment.phase.profile)
        phase_target_frames = _phase_target_frames(target_frames, segment.phase.duration_seconds, phases)
        processed = _process_sequence(
            segment.sequence,
            profile=profile,
            target_frames=phase_target_frames,
            dimension_k=dimension_k,
            smooth_window=smooth_window,
            min_visibility=min_visibility,
        )
        phase_paths = _artifact_paths(base_path, artifact_id, segment.phase.id)
        if save_arrays:
            np.save(phase_paths["sequence_path"], segment.sequence)
            _save_processed_arrays(phase_paths, processed)
        phase_segments.append(
            {
                "phase_id": segment.phase.id,
                "label": segment.phase.label,
                "prompt": segment.phase.prompt,
                "profile": profile.name,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "duration_frames": segment.end_frame - segment.start_frame,
                "event_center_frame": segment.event_center_frame,
                "segmentation_method": segment.segmentation_method,
                "sequence_path": str(phase_paths["sequence_path"]),
                "normalized_sequence_path": str(phase_paths["normalized_sequence_path"]),
                "playback_sequence_path": str(phase_paths["playback_sequence_path"]),
                "motion_matrix_path": str(phase_paths["motion_matrix_path"]),
                "coordinate_subspace_basis_path": str(phase_paths["coordinate_subspace_basis_path"]),
                "coordinate_subspace_mean_path": str(phase_paths["coordinate_subspace_mean_path"]),
                "velocity_matrix_path": str(phase_paths["velocity_matrix_path"]),
                "velocity_subspace_basis_path": str(phase_paths["velocity_subspace_basis_path"]),
                "velocity_subspace_mean_path": str(phase_paths["velocity_subspace_mean_path"]),
                "features": processed["features"],
                "quality": _phase_quality(
                    segment.sequence,
                    phase_id=segment.phase.id,
                    min_visibility=min_visibility,
                ),
                "subspace": _subspace_summary(processed),
            }
        )

    segmentation_method = (
        phase_segments[0]["segmentation_method"]
        if phase_segments
        else ("event" if event_based_segmentation else "timer")
    )

    return {
        "ritual_version": 1,
        "segmentation": {
            "method": segmentation_method,
            "requested_method": "event" if event_based_segmentation else "timer",
            "event_padding_ratio": event_padding_ratio,
        },
        "phase_count": len(phase_segments),
        "full": {
            "normalized_sequence_path": str(full_paths["normalized_sequence_path"]),
            "playback_sequence_path": str(full_paths["playback_sequence_path"]),
            "motion_matrix_path": str(full_paths["motion_matrix_path"]),
            "coordinate_subspace_basis_path": str(full_paths["coordinate_subspace_basis_path"]),
            "coordinate_subspace_mean_path": str(full_paths["coordinate_subspace_mean_path"]),
            "velocity_matrix_path": str(full_paths["velocity_matrix_path"]),
            "velocity_subspace_basis_path": str(full_paths["velocity_subspace_basis_path"]),
            "velocity_subspace_mean_path": str(full_paths["velocity_subspace_mean_path"]),
            "features": full["features"],
            "quality": _sequence_quality(sequence, min_visibility=min_visibility),
            "subspace": _subspace_summary(full),
        },
        "phase_segments": phase_segments,
    }


def load_basis(path: str) -> np.ndarray:
    return np.load(path)


def _process_sequence(
    sequence: np.ndarray,
    *,
    profile: MotionProfile,
    target_frames: int,
    dimension_k: int,
    smooth_window: int,
    min_visibility: float,
) -> dict[str, Any]:
    normalized_sequence, normalization_summary = normalize_keypoint_sequence(
        sequence,
        target_frames=target_frames,
        smooth_window=smooth_window,
        min_visibility=min_visibility,
        keypoint_names=profile.keypoint_names,
        center_keypoint_names=profile.center_keypoint_names,
        scale_from_keypoint_names=profile.scale_from_keypoint_names,
        scale_to_keypoint_names=profile.scale_to_keypoint_names,
    )
    playback_sequence = normalize_sequence_for_playback(
        sequence,
        smooth_window=smooth_window,
        min_visibility=min_visibility,
        profile=profile,
    )
    motion_matrix, matrix_summary = normalize_to_motion_matrix(
        sequence,
        target_frames=target_frames,
        smooth_window=smooth_window,
        min_visibility=min_visibility,
        keypoint_names=profile.keypoint_names,
        center_keypoint_names=profile.center_keypoint_names,
        scale_from_keypoint_names=profile.scale_from_keypoint_names,
        scale_to_keypoint_names=profile.scale_to_keypoint_names,
    )
    coordinate_subspace = build_motion_subspace(motion_matrix, dimension_k=dimension_k)
    velocity_matrix = velocity_matrix_from_motion_matrix(motion_matrix)
    velocity_subspace = build_motion_subspace(velocity_matrix, dimension_k=dimension_k)
    return {
        "normalized_sequence": normalized_sequence,
        "playback_sequence": playback_sequence,
        "motion_matrix": motion_matrix,
        "coordinate_subspace": coordinate_subspace,
        "velocity_matrix": velocity_matrix,
        "velocity_subspace": velocity_subspace,
        "features": compute_motion_features(normalized_sequence),
        "normalization": {
            "input_frames": normalization_summary.input_frames,
            "output_frames": normalization_summary.output_frames,
            "feature_dimensions": matrix_summary.feature_dimensions,
            "interpolated_values": normalization_summary.interpolated_values,
            "median_torso_scale": normalization_summary.median_torso_scale,
            "smooth_window": smooth_window,
            "min_visibility": min_visibility,
        },
        "playback": {
            "input_frames": int(sequence.shape[0]),
            "output_frames": int(playback_sequence.shape[0]),
        },
    }


def normalize_sequence_for_playback(
    sequence: np.ndarray,
    *,
    smooth_window: int = 5,
    min_visibility: float = 0.0,
    profile: MotionProfile = FULL_BODY_PROFILE,
) -> np.ndarray:
    if sequence.ndim != 3 or sequence.shape[1] != len(KEYPOINT_NAMES) or sequence.shape[2] < 2:
        raise ValueError(
            "Expected keypoint sequence with shape "
            f"(frames, {len(KEYPOINT_NAMES)}, 2 or 3); got {sequence.shape}"
        )

    keypoint_indices = _indices_for_names(profile.keypoint_names)
    center_indices = _indices_for_names(profile.center_keypoint_names)
    scale_from_indices = _indices_for_names(profile.scale_from_keypoint_names)
    scale_to_indices = _indices_for_names(profile.scale_to_keypoint_names)

    coords = np.array(sequence[:, :, :2], dtype=np.float64, copy=True)
    if sequence.shape[2] >= 3 and min_visibility > 0.0:
        visibility = sequence[:, :, 2]
        coords[visibility < min_visibility] = np.nan
    coords = _interpolate_missing(coords)
    coords = np.nan_to_num(coords, nan=0.0)

    frame_centers = coords[:, center_indices].mean(axis=1)
    global_center = np.median(frame_centers, axis=0)
    scale_from = coords[:, scale_from_indices].mean(axis=1)
    scale_to = coords[:, scale_to_indices].mean(axis=1)
    scale_values = np.linalg.norm(scale_from - scale_to, axis=1)
    valid_scales = scale_values[np.isfinite(scale_values) & (scale_values > 1e-6)]
    median_scale = float(np.median(valid_scales)) if len(valid_scales) else 1.0

    selected = coords[:, keypoint_indices]
    normalized = (selected - global_center[None, None, :]) / median_scale
    return _moving_average(normalized, window=smooth_window).astype(np.float32)


def _save_processed_arrays(paths: dict[str, Path], processed: dict[str, Any]) -> None:
    np.save(paths["normalized_sequence_path"], processed["normalized_sequence"])
    np.save(paths["playback_sequence_path"], processed["playback_sequence"])
    np.save(paths["motion_matrix_path"], processed["motion_matrix"])
    np.save(paths["coordinate_subspace_basis_path"], processed["coordinate_subspace"].basis)
    np.save(paths["coordinate_subspace_mean_path"], processed["coordinate_subspace"].mean)
    np.save(paths["velocity_matrix_path"], processed["velocity_matrix"])
    np.save(paths["velocity_subspace_basis_path"], processed["velocity_subspace"].basis)
    np.save(paths["velocity_subspace_mean_path"], processed["velocity_subspace"].mean)


def _artifact_paths(base_path: Path, artifact_id: str, label: str) -> dict[str, Path]:
    return {
        "sequence_path": base_path.with_name(f"{artifact_id}_{label}_sequence.npy"),
        "normalized_sequence_path": base_path.with_name(f"{artifact_id}_{label}_normalized.npy"),
        "playback_sequence_path": base_path.with_name(f"{artifact_id}_{label}_playback.npy"),
        "motion_matrix_path": base_path.with_name(f"{artifact_id}_{label}_motion_matrix.npy"),
        "coordinate_subspace_basis_path": base_path.with_name(f"{artifact_id}_{label}_coordinate_basis.npy"),
        "coordinate_subspace_mean_path": base_path.with_name(f"{artifact_id}_{label}_coordinate_mean.npy"),
        "velocity_matrix_path": base_path.with_name(f"{artifact_id}_{label}_velocity_matrix.npy"),
        "velocity_subspace_basis_path": base_path.with_name(f"{artifact_id}_{label}_velocity_basis.npy"),
        "velocity_subspace_mean_path": base_path.with_name(f"{artifact_id}_{label}_velocity_mean.npy"),
    }


def _sequence_quality(sequence: np.ndarray, *, min_visibility: float) -> dict[str, float]:
    if sequence.ndim != 3 or sequence.shape[0] == 0 or sequence.shape[2] < 3:
        return {
            "frames": float(sequence.shape[0]) if sequence.ndim else 0.0,
            "visibility_ratio": 0.0,
            "confidence": 0.0,
        }
    visibility = sequence[:, :, 2]
    visibility_ratio = float((visibility >= min_visibility).mean())
    tracked_frame_ratio = float(((visibility >= min_visibility).any(axis=1)).mean())
    confidence = max(0.0, min(1.0, visibility_ratio / 0.7)) * max(
        0.0,
        min(1.0, tracked_frame_ratio / 0.8),
    )
    return {
        "frames": float(sequence.shape[0]),
        "visibility_ratio": visibility_ratio,
        "tracked_frame_ratio": tracked_frame_ratio,
        "confidence": confidence,
    }


def _phase_quality(
    sequence: np.ndarray,
    *,
    phase_id: str,
    min_visibility: float,
) -> dict[str, float]:
    quality = _sequence_quality(sequence, min_visibility=min_visibility)
    metrics = _phase_activity_metrics(sequence, min_visibility=min_visibility)
    quality.update(metrics)
    if phase_id == "wave_hello":
        quality["activity_score"] = metrics["wrist_activity"]
    elif phase_id == "clap_twice":
        quality["activity_score"] = max(metrics["wrist_activity"], metrics["clap_peak_count"] / 2.0)
    elif phase_id == "squat":
        quality["activity_score"] = metrics["hip_drop"]
    elif phase_id == "balance_hold":
        quality["activity_score"] = metrics["knee_lift"]
    else:
        quality["activity_score"] = metrics["body_activity"]
    return quality


def _phase_activity_metrics(sequence: np.ndarray, *, min_visibility: float) -> dict[str, float]:
    if sequence.ndim != 3 or sequence.shape[0] < 2 or sequence.shape[2] < 2:
        return {
            "body_activity": 0.0,
            "wrist_activity": 0.0,
            "hip_drop": 0.0,
            "knee_lift": 0.0,
            "clap_peak_count": 0.0,
            "wrist_distance_min": 0.0,
        }

    coords = np.array(sequence[:, :, :2], dtype=np.float64, copy=True)
    if sequence.shape[2] >= 3 and min_visibility > 0.0:
        coords[sequence[:, :, 2] < min_visibility] = np.nan
    coords = _interpolate_missing(coords)
    coords = np.nan_to_num(coords, nan=0.0)
    scale = _median_body_scale(coords)

    body_activity = float(np.linalg.norm(np.diff(coords, axis=0), axis=2).mean() / scale)
    wrists = coords[:, _indices_for_names(("left_wrist", "right_wrist"))]
    wrist_steps = np.linalg.norm(np.diff(wrists, axis=0), axis=2)
    wrist_activity = float(wrist_steps.sum(axis=0).max() / scale)

    hips = coords[:, _indices_for_names(("left_hip", "right_hip"))].mean(axis=1)
    hip_drop = float((np.max(hips[:, 1]) - np.min(hips[:, 1])) / scale)

    knees = coords[:, _indices_for_names(("left_knee", "right_knee"))]
    knee_lift = float(np.max(np.maximum(hips[:, 1] - knees[:, 0, 1], hips[:, 1] - knees[:, 1, 1])) / scale)
    knee_lift = max(0.0, knee_lift)

    wrist_distance = np.linalg.norm(wrists[:, 0] - wrists[:, 1], axis=1) / scale
    return {
        "body_activity": body_activity,
        "wrist_activity": wrist_activity,
        "hip_drop": hip_drop,
        "knee_lift": knee_lift,
        "clap_peak_count": float(_local_minima_count(wrist_distance)),
        "wrist_distance_min": float(np.min(wrist_distance)) if wrist_distance.size else 0.0,
    }


def _median_body_scale(coords: np.ndarray) -> float:
    shoulders = coords[:, _indices_for_names(("left_shoulder", "right_shoulder"))].mean(axis=1)
    hips = coords[:, _indices_for_names(("left_hip", "right_hip"))].mean(axis=1)
    scale = np.linalg.norm(shoulders - hips, axis=1)
    valid = scale[np.isfinite(scale) & (scale > 1e-6)]
    return float(np.median(valid)) if len(valid) else 1.0


def _local_minima_count(values: np.ndarray) -> int:
    if values.size < 3:
        return 0
    threshold = float(np.median(values) - np.std(values) * 0.25)
    return sum(
        1
        for index in range(1, values.size - 1)
        if values[index] <= threshold
        and values[index] <= values[index - 1]
        and values[index] <= values[index + 1]
    )


def _subspace_summary(processed: dict[str, Any]) -> dict[str, Any]:
    coordinate = processed["coordinate_subspace"]
    velocity = processed["velocity_subspace"]
    return {
        "coordinate_basis_shape": list(coordinate.basis.shape),
        "velocity_basis_shape": list(velocity.basis.shape),
        "coordinate_explained_variance_ratio": coordinate.explained_variance_ratio.tolist(),
        "velocity_explained_variance_ratio": velocity.explained_variance_ratio.tolist(),
    }


def _profile_for_phase(profile_name: str) -> MotionProfile:
    normalized = "".join(character for character in profile_name.lower() if character.isalnum())
    if normalized in {"upperbody", "wave", "gesture"}:
        return WAVE_PROFILE
    return FULL_BODY_PROFILE


def _phase_target_frames(
    target_frames: int,
    duration_seconds: float,
    phases,
) -> int:
    total_seconds = float(sum(max(0.001, phase.duration_seconds) for phase in phases))
    proportional = int(round(target_frames * max(0.001, duration_seconds) / total_seconds))
    return max(24, proportional)


def _indices_for_names(names: tuple[str, ...]) -> list[int]:
    index_by_name = {name: index for index, name in enumerate(KEYPOINT_NAMES)}
    try:
        return [index_by_name[name] for name in names]
    except KeyError as exc:
        raise ValueError(f"Unknown keypoint name: {exc.args[0]}") from exc


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
    if window <= 1 or sequence.shape[0] < 2:
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
