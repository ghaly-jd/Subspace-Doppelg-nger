from __future__ import annotations

import math

import numpy as np


def dtw_distance(values_a: list[float] | np.ndarray, values_b: list[float] | np.ndarray) -> float:
    curve_a = _clean_curve(values_a)
    curve_b = _clean_curve(values_b)
    if curve_a.size == 0 or curve_b.size == 0:
        return float("inf")

    costs = np.full((curve_a.size + 1, curve_b.size + 1), np.inf, dtype=np.float64)
    costs[0, 0] = 0.0
    for index_a in range(1, curve_a.size + 1):
        value_a = curve_a[index_a - 1]
        for index_b in range(1, curve_b.size + 1):
            cost = abs(value_a - curve_b[index_b - 1])
            costs[index_a, index_b] = cost + min(
                costs[index_a - 1, index_b],
                costs[index_a, index_b - 1],
                costs[index_a - 1, index_b - 1],
            )
    path_scale = float(curve_a.size + curve_b.size)
    return float(costs[curve_a.size, curve_b.size] / max(1.0, path_scale))


def speed_curve_dtw_similarity(
    values_a: list[float] | np.ndarray,
    values_b: list[float] | np.ndarray,
    *,
    alpha: float = 3.0,
) -> float:
    curve_a = _normalize_curve(_clean_curve(values_a))
    curve_b = _normalize_curve(_clean_curve(values_b))
    distance = dtw_distance(curve_a, curve_b)
    if not math.isfinite(distance):
        return 0.0
    return float(100.0 * math.exp(-alpha * distance))


def _clean_curve(values: list[float] | np.ndarray) -> np.ndarray:
    curve = np.asarray(values, dtype=np.float64).reshape(-1)
    if curve.size == 0:
        return curve
    return np.nan_to_num(curve, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_curve(curve: np.ndarray) -> np.ndarray:
    if curve.size == 0:
        return curve
    scale = float(np.percentile(np.abs(curve), 90))
    if scale <= 1e-9:
        scale = float(np.max(np.abs(curve)))
    if scale <= 1e-9:
        return np.zeros_like(curve)
    return curve / scale
