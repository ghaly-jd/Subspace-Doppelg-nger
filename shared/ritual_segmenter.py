from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.pose_schema import KEYPOINT_NAMES
from shared.ritual_schema import RitualPhase


@dataclass(frozen=True)
class RitualSegment:
    phase: RitualPhase
    start_frame: int
    end_frame: int
    sequence: np.ndarray
    event_center_frame: int | None = None
    segmentation_method: str = "timer"


def segment_ritual_sequence(
    sequence: np.ndarray,
    phases: tuple[RitualPhase, ...],
    *,
    event_based: bool = True,
    event_padding_ratio: float = 0.45,
    min_visibility: float = 0.0,
) -> list[RitualSegment]:
    if sequence.ndim != 3:
        raise ValueError(f"Expected ritual sequence with 3 dimensions; got {sequence.shape}")
    if sequence.shape[0] < 2:
        raise ValueError("At least two frames are required for ritual segmentation.")
    if not phases:
        raise ValueError("At least one ritual phase is required.")

    frame_count = int(sequence.shape[0])
    boundaries = _timer_boundaries(frame_count, phases)
    method = "timer"
    event_centers: list[int | None] = [None] * len(phases)
    if event_based:
        refined = _event_boundaries(
            sequence,
            phases,
            timer_boundaries=boundaries,
            event_padding_ratio=event_padding_ratio,
            min_visibility=min_visibility,
        )
        if refined is not None:
            boundaries, event_centers = refined
            method = "event"

    return _segments_from_boundaries(
        sequence,
        phases,
        boundaries=boundaries,
        event_centers=event_centers,
        method=method,
    )


def _timer_boundaries(frame_count: int, phases: tuple[RitualPhase, ...]) -> np.ndarray:
    durations = np.array([max(0.001, phase.duration_seconds) for phase in phases], dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(durations) / float(durations.sum())))
    boundaries = np.rint(cumulative * frame_count).astype(int)
    boundaries[0] = 0
    boundaries[-1] = frame_count
    return boundaries


def _segments_from_boundaries(
    sequence: np.ndarray,
    phases: tuple[RitualPhase, ...],
    *,
    boundaries: np.ndarray,
    event_centers: list[int | None],
    method: str,
) -> list[RitualSegment]:
    frame_count = int(sequence.shape[0])

    segments: list[RitualSegment] = []
    previous_end = 0
    for index, phase in enumerate(phases):
        start = int(max(previous_end, boundaries[index]))
        end = int(boundaries[index + 1])
        remaining_phases = len(phases) - index - 1
        min_end = start + 2
        max_end = frame_count - (remaining_phases * 2)
        end = max(min_end, min(end, max_end if remaining_phases else frame_count))
        if end > frame_count:
            end = frame_count
        if end - start < 2:
            start = max(0, min(start, frame_count - 2))
            end = min(frame_count, start + 2)
        segments.append(
            RitualSegment(
                phase=phase,
                start_frame=start,
                end_frame=end,
                sequence=sequence[start:end],
                event_center_frame=event_centers[index],
                segmentation_method=method,
            )
        )
        previous_end = end
    return segments


def _event_boundaries(
    sequence: np.ndarray,
    phases: tuple[RitualPhase, ...],
    *,
    timer_boundaries: np.ndarray,
    event_padding_ratio: float,
    min_visibility: float,
) -> tuple[np.ndarray, list[int | None]] | None:
    frame_count = int(sequence.shape[0])
    coords = _clean_coords(sequence, min_visibility=min_visibility)
    if coords.shape[0] < 4 or not np.isfinite(coords).any():
        return None

    centers: list[int] = []
    for index, phase in enumerate(phases):
        start = int(timer_boundaries[index])
        end = int(timer_boundaries[index + 1])
        duration = max(2, end - start)
        padding = int(round(duration * max(0.0, event_padding_ratio)))
        window_start = max(0, start - padding)
        window_end = min(frame_count, end + padding)
        center = _phase_event_center(phase.id, coords, window_start, window_end)
        center = int(np.clip(center, window_start, max(window_start, window_end - 1)))
        centers.append(center)

    centers = _monotonic_centers(centers, frame_count)
    if len(centers) != len(phases):
        return None

    boundaries = np.zeros(len(phases) + 1, dtype=int)
    boundaries[0] = 0
    boundaries[-1] = frame_count
    for index in range(1, len(phases)):
        boundary = int(round((centers[index - 1] + centers[index]) / 2.0))
        boundary = max(boundaries[index - 1] + 2, min(boundary, frame_count - ((len(phases) - index) * 2)))
        boundaries[index] = boundary
    return boundaries, centers


def _phase_event_center(phase_id: str, coords: np.ndarray, start: int, end: int) -> int:
    window = coords[start:end]
    if window.shape[0] < 2:
        return start

    if phase_id == "calibration_pose":
        activity = _body_activity(window)
        quietest = int(np.argmin(_rolling_mean(activity, window=5))) if activity.size else window.shape[0] // 2
        return start + quietest
    if phase_id == "lateral_step":
        center_x = _body_center(window)[:, 0]
        return start + int(np.argmax(np.abs(center_x - np.nanmedian(center_x))))
    if phase_id == "arm_raise":
        wrists = _points(window, ("left_wrist", "right_wrist"))
        wrist_y = np.nanmean(wrists[:, :, 1], axis=1)
        return start + int(np.nanargmin(wrist_y))
    if phase_id == "squat":
        hip_y = _body_center(window, names=("left_hip", "right_hip"))[:, 1]
        return start + int(np.nanargmax(hip_y))
    if phase_id == "rotation":
        shoulders = _points(window, ("left_shoulder", "right_shoulder"))
        shoulder_width = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
        return start + int(np.argmax(np.abs(shoulder_width - np.nanmedian(shoulder_width))))
    if phase_id == "balance_hold":
        hip_y = _body_center(window, names=("left_hip", "right_hip"))[:, 1]
        knees = _points(window, ("left_knee", "right_knee"))
        knee_lift = np.maximum(hip_y - knees[:, 0, 1], hip_y - knees[:, 1, 1])
        return start + int(np.nanargmax(knee_lift))
    if phase_id == "wave_hello":
        wrist_activity = _wrist_activity(window)
        return start + int(np.argmax(_rolling_mean(wrist_activity, window=3)))
    if phase_id == "clap_twice":
        wrists = _points(window, ("left_wrist", "right_wrist"))
        wrist_distance = np.linalg.norm(wrists[:, 0] - wrists[:, 1], axis=1)
        peaks = _local_minima(wrist_distance)
        if peaks:
            strongest = sorted(peaks, key=lambda item: wrist_distance[item])[:2]
            return start + int(round(float(np.mean(strongest))))
        return start + int(np.nanargmin(wrist_distance))
    if phase_id == "point_forward":
        wrists = _points(window, ("left_wrist", "right_wrist"))
        shoulders = _points(window, ("left_shoulder", "right_shoulder"))
        reach = np.maximum(
            np.linalg.norm(wrists[:, 0] - shoulders[:, 0], axis=1),
            np.linalg.norm(wrists[:, 1] - shoulders[:, 1], axis=1),
        )
        return start + int(np.nanargmax(reach))

    activity = _body_activity(window)
    return start + int(np.argmax(_rolling_mean(activity, window=3))) if activity.size else start


def _monotonic_centers(centers: list[int], frame_count: int) -> list[int]:
    if not centers:
        return centers
    output = [int(np.clip(centers[0], 0, frame_count - 1))]
    for center in centers[1:]:
        output.append(max(output[-1] + 1, int(np.clip(center, 0, frame_count - 1))))
    overflow = output[-1] - (frame_count - 1)
    if overflow > 0:
        output = [center - overflow for center in output]
    for index in range(len(output) - 2, -1, -1):
        output[index] = min(output[index], output[index + 1] - 1)
    return [int(np.clip(center, 0, frame_count - 1)) for center in output]


def _clean_coords(sequence: np.ndarray, *, min_visibility: float) -> np.ndarray:
    coords = np.array(sequence[:, :, :2], dtype=np.float64, copy=True)
    if sequence.shape[2] >= 3 and min_visibility > 0.0:
        visibility = sequence[:, :, 2]
        coords[visibility < min_visibility] = np.nan
    frame_positions = np.arange(coords.shape[0], dtype=np.float64)
    for keypoint_index in range(coords.shape[1]):
        for axis in range(coords.shape[2]):
            values = coords[:, keypoint_index, axis]
            valid = np.isfinite(values)
            if valid.all():
                continue
            if valid.sum() == 0:
                values[:] = 0.0
                continue
            coords[:, keypoint_index, axis] = np.interp(
                frame_positions,
                frame_positions[valid],
                values[valid],
            )
    return np.nan_to_num(coords, nan=0.0)


def _points(window: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    return window[:, [_keypoint_index(name) for name in names], :]


def _body_center(
    window: np.ndarray,
    *,
    names: tuple[str, ...] = ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
) -> np.ndarray:
    return np.nanmean(_points(window, names), axis=1)


def _body_activity(window: np.ndarray) -> np.ndarray:
    if window.shape[0] < 2:
        return np.zeros(1, dtype=np.float64)
    return np.linalg.norm(np.diff(window, axis=0), axis=2).mean(axis=1)


def _wrist_activity(window: np.ndarray) -> np.ndarray:
    wrists = _points(window, ("left_wrist", "right_wrist"))
    if wrists.shape[0] < 2:
        return np.zeros(1, dtype=np.float64)
    return np.linalg.norm(np.diff(wrists, axis=0), axis=2).mean(axis=1)


def _rolling_mean(values: np.ndarray, *, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values
    window = min(int(window), int(values.size))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _local_minima(values: np.ndarray) -> list[int]:
    if values.size < 3:
        return []
    return [
        index
        for index in range(1, values.size - 1)
        if values[index] <= values[index - 1] and values[index] <= values[index + 1]
    ]


def _keypoint_index(name: str) -> int:
    try:
        return KEYPOINT_NAMES.index(name)
    except ValueError as exc:
        raise ValueError(f"Unknown keypoint name: {name}") from exc
