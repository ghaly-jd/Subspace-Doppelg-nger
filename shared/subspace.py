from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubspaceModel:
    basis: np.ndarray
    mean: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray


@dataclass(frozen=True)
class SubspaceComparison:
    canonical_angles_degrees: np.ndarray
    mean_angle_degrees: float
    projection_distance: float
    similarity: float


def build_motion_subspace(matrix: np.ndarray, *, dimension_k: int = 5) -> SubspaceModel:
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D motion matrix; got {matrix.shape}")
    if matrix.shape[0] < 2:
        raise ValueError("At least two frames are required to build a motion subspace.")
    if dimension_k < 1:
        raise ValueError("dimension_k must be at least 1.")

    clean_matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float64), nan=0.0)
    mean = clean_matrix.mean(axis=0)
    centered = clean_matrix - mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)

    max_k = min(dimension_k, vh.shape[0])
    basis = vh[:max_k].T
    total_energy = float(np.sum(singular_values**2))
    if total_energy <= 0.0:
        explained = np.zeros(max_k, dtype=np.float64)
    else:
        explained = (singular_values[:max_k] ** 2) / total_energy

    return SubspaceModel(
        basis=basis.astype(np.float32),
        mean=mean.astype(np.float32),
        singular_values=singular_values[:max_k].astype(np.float32),
        explained_variance_ratio=explained.astype(np.float32),
    )


def canonical_angles(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    a = _orthonormalize(basis_a)
    b = _orthonormalize(basis_b)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Basis feature dimensions differ: {a.shape[0]} vs {b.shape[0]}")

    singular_values = np.linalg.svd(a.T @ b, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return np.arccos(singular_values)


def projection_distance(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    angles = canonical_angles(basis_a, basis_b)
    return float(np.sqrt(np.sum(np.sin(angles) ** 2)))


def similarity_from_distance(distance: float, *, alpha: float = 1.5) -> float:
    return float(100.0 * np.exp(-alpha * max(0.0, distance)))


def compare_subspaces(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    *,
    similarity_alpha: float = 1.5,
) -> SubspaceComparison:
    angles = canonical_angles(basis_a, basis_b)
    distance = float(np.sqrt(np.sum(np.sin(angles) ** 2)))
    return SubspaceComparison(
        canonical_angles_degrees=np.degrees(angles).astype(np.float32),
        mean_angle_degrees=float(np.degrees(angles).mean()) if len(angles) else 0.0,
        projection_distance=distance,
        similarity=similarity_from_distance(distance, alpha=similarity_alpha),
    )


def _orthonormalize(basis: np.ndarray) -> np.ndarray:
    matrix = np.asarray(basis, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D basis matrix; got {matrix.shape}")
    q, _ = np.linalg.qr(matrix)
    return q
