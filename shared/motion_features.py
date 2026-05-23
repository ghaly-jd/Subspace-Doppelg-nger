from __future__ import annotations

import math
from typing import Any

import numpy as np


def velocity_matrix_from_motion_matrix(matrix: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(matrix, dtype=np.float64), nan=0.0)
    if clean.shape[0] < 2:
        return np.zeros_like(clean)
    velocity = np.diff(clean, axis=0)
    if velocity.shape[0] < 2:
        velocity = np.vstack([velocity, velocity])
    return velocity.astype(np.float32)


def compute_motion_features(normalized_sequence: np.ndarray) -> dict[str, dict[str, float]]:
    sequence = np.nan_to_num(np.asarray(normalized_sequence, dtype=np.float64), nan=0.0)
    if sequence.ndim != 3 or sequence.shape[0] < 2:
        return _empty_features()

    velocity = np.diff(sequence, axis=0)
    joint_speed = np.linalg.norm(velocity, axis=2)
    speed_curve = joint_speed.mean(axis=1)
    center_curve = sequence.mean(axis=1)

    rhythm = {
        "mean_speed": _safe_mean(speed_curve),
        "max_speed": _safe_max(speed_curve),
        "speed_std": _safe_std(speed_curve),
        "speed_peak_count": float(_peak_count(speed_curve)),
        "dominant_period": _dominant_period(speed_curve),
    }
    energy = {
        "total_joint_travel": float(joint_speed.sum()),
        "mean_joint_travel": _safe_mean(joint_speed.sum(axis=0)),
        "max_joint_travel": _safe_max(joint_speed.sum(axis=0)),
        "acceleration_energy": _acceleration_energy(velocity),
    }
    balance = {
        "center_lateral_range": _safe_range(center_curve[:, 0]),
        "center_vertical_range": _safe_range(center_curve[:, 1]),
        "center_path_length": _path_length(center_curve),
        "center_stability": 1.0 / (1.0 + _safe_std(center_curve[:, 0]) + _safe_std(center_curve[:, 1])),
    }
    joint_angles = _joint_angle_features(sequence)
    gesture = _gesture_features(sequence)

    return {
        "rhythm": rhythm,
        "energy": energy,
        "balance": balance,
        "joint_angles": joint_angles,
        "gesture": gesture,
        "curves": {
            "speed_curve": speed_curve.astype(float).tolist(),
        },
    }


def feature_similarity(
    features_a: dict[str, Any],
    features_b: dict[str, Any],
    *,
    categories: tuple[str, ...] | None = None,
) -> float:
    values_a = _flatten_numeric(features_a, categories=categories)
    values_b = _flatten_numeric(features_b, categories=categories)
    shared_keys = sorted(set(values_a) & set(values_b))
    if not shared_keys:
        return 0.0

    distances = []
    for key in shared_keys:
        a = values_a[key]
        b = values_b[key]
        scale = max(abs(a), abs(b), 1.0)
        distances.append(abs(a - b) / scale)
    mean_distance = float(np.mean(distances)) if distances else 1.0
    return float(100.0 * math.exp(-1.8 * mean_distance))


def combined_energy_balance_similarity(features_a: dict[str, Any], features_b: dict[str, Any]) -> float:
    return feature_similarity(features_a, features_b, categories=("energy", "balance"))


def _joint_angle_features(sequence: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    angle_specs = {
        "left_elbow_angle": (0, 2, 4),
        "right_elbow_angle": (1, 3, 5),
        "left_knee_angle": (6, 8, 10),
        "right_knee_angle": (7, 9, 11),
        "left_torso_angle": (0, 6, 8),
        "right_torso_angle": (1, 7, 9),
    }
    for name, indices in angle_specs.items():
        if max(indices) >= sequence.shape[1]:
            continue
        angles = _angle_series(sequence[:, indices[0]], sequence[:, indices[1]], sequence[:, indices[2]])
        features[f"{name}_mean"] = _safe_mean(angles)
        features[f"{name}_range"] = _safe_range(angles)
    return features


def _gesture_features(sequence: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    if sequence.shape[1] >= 6:
        left_wrist = sequence[:, 4]
        right_wrist = sequence[:, 5]
        wrist_distance = np.linalg.norm(left_wrist - right_wrist, axis=1)
        features["wrist_distance_min"] = _safe_min(wrist_distance)
        features["wrist_distance_range"] = _safe_range(wrist_distance)
        features["left_wrist_range"] = _point_range(left_wrist)
        features["right_wrist_range"] = _point_range(right_wrist)
        features["right_wrist_travel"] = _path_length(right_wrist)
        features["left_wrist_travel"] = _path_length(left_wrist)
    return features


def _angle_series(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    numerator = np.sum(ba * bc, axis=1)
    denominator = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    cosine = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 1e-9)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _peak_count(values: np.ndarray) -> int:
    if values.size < 3:
        return 0
    threshold = float(values.mean() + values.std() * 0.35)
    peaks = 0
    for index in range(1, values.size - 1):
        if values[index] > threshold and values[index] >= values[index - 1] and values[index] >= values[index + 1]:
            peaks += 1
    return peaks


def _dominant_period(values: np.ndarray) -> float:
    centered = values - values.mean()
    if centered.size < 4 or float(np.abs(centered).sum()) <= 1e-9:
        return 0.0
    correlation = np.correlate(centered, centered, mode="full")[centered.size - 1 :]
    if correlation.size <= 2:
        return 0.0
    lag = int(np.argmax(correlation[1:]) + 1)
    return float(lag)


def _acceleration_energy(velocity: np.ndarray) -> float:
    if velocity.shape[0] < 2:
        return 0.0
    acceleration = np.diff(velocity, axis=0)
    return float(np.linalg.norm(acceleration, axis=2).sum())


def _flatten_numeric(
    value: dict[str, Any],
    *,
    categories: tuple[str, ...] | None,
    prefix: str = "",
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    items = value.items() if categories is None else ((key, value.get(key, {})) for key in categories)
    for key, item in items:
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flattened.update(_flatten_numeric(item, categories=None, prefix=path))
        elif isinstance(item, (int, float)) and math.isfinite(float(item)):
            flattened[path] = float(item)
    return flattened


def _empty_features() -> dict[str, dict[str, float]]:
    return {
        "rhythm": {},
        "energy": {},
        "balance": {},
        "joint_angles": {},
        "gesture": {},
        "curves": {},
    }


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


def _safe_min(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size else 0.0


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def _safe_range(values: np.ndarray) -> float:
    return _safe_max(values) - _safe_min(values)


def _point_range(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def _path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
