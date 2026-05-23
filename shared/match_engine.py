from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from shared.config import get_nested
from shared.motion_profile import MotionProfile, profile_for_motion
from shared.pose_schema import KEYPOINT_NAMES, HandKeypoints, PoseKeypoints
from shared.registration_storage import keypoint_frames_to_array
from shared.ritual_match_engine import find_ritual_matches
from shared.ritual_schema import is_ritual_motion
from shared.skeleton_normalizer import normalize_keypoint_sequence, normalize_to_motion_matrix
from shared.subspace import build_motion_subspace, compare_subspaces


def find_motion_matches(
    config: dict[str, Any],
    *,
    motion_type: str,
    frames: list[PoseKeypoints],
    top_k: int = 3,
    hand_frames: list[HandKeypoints] | None = None,
) -> dict[str, Any]:
    if is_ritual_motion(motion_type):
        return find_ritual_matches(config, frames=frames, top_k=top_k, hand_frames=hand_frames)

    query = build_query_artifacts(config, motion_type=motion_type, frames=frames)
    candidates = load_registered_candidates(config, motion_type=motion_type)
    ranked = rank_candidates(
        query["subspace_basis"],
        candidates,
        similarity_alpha=_motion_float(
            config,
            motion_type,
            "similarity_alpha",
            float(get_nested(config, "subspace", "similarity_alpha", default=1.5)),
        ),
    )
    ranked = _apply_query_quality(ranked, query["quality"])
    min_accept_similarity = _motion_float(
        config,
        motion_type,
        "min_accept_similarity",
        0.0,
    )
    accepted = bool(ranked) and ranked[0]["similarity"] >= min_accept_similarity

    result = {
        "motion_type": motion_type,
        "query": _public_query(query),
        "top_matches": ranked[:top_k],
        "candidate_count": len(candidates),
        "accepted": accepted,
        "min_accept_similarity": min_accept_similarity,
        "no_match_reason": "" if accepted else _no_match_reason(motion_type, query["quality"]),
    }
    _write_query_result(query["metadata_path"], result)
    return result


def build_query_artifacts(
    config: dict[str, Any],
    *,
    motion_type: str,
    frames: list[PoseKeypoints],
) -> dict[str, Any]:
    sessions_dir = Path(get_nested(config, "data", "sessions_dir", default="data/sessions"))
    sessions_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)
    query_id = _build_query_id(timestamp, motion_type)
    sequence_path = sessions_dir / f"{query_id}_sequence.npy"
    normalized_path = sessions_dir / f"{query_id}_normalized.npy"
    matrix_path = sessions_dir / f"{query_id}_motion_matrix.npy"
    basis_path = sessions_dir / f"{query_id}_subspace_basis.npy"
    mean_path = sessions_dir / f"{query_id}_subspace_mean.npy"
    metadata_path = sessions_dir / f"{query_id}.json"
    suffix = 2
    while sequence_path.exists() or metadata_path.exists():
        query_id = f"{_build_query_id(timestamp, motion_type)}-{suffix}"
        sequence_path = sessions_dir / f"{query_id}_sequence.npy"
        normalized_path = sessions_dir / f"{query_id}_normalized.npy"
        matrix_path = sessions_dir / f"{query_id}_motion_matrix.npy"
        basis_path = sessions_dir / f"{query_id}_subspace_basis.npy"
        mean_path = sessions_dir / f"{query_id}_subspace_mean.npy"
        metadata_path = sessions_dir / f"{query_id}.json"
        suffix += 1

    sequence = keypoint_frames_to_array(frames)
    profile = profile_for_motion(motion_type)
    target_frames = int(get_nested(config, "subspace", "num_frames", default=120))
    dimension_k = _motion_int(
        config,
        motion_type,
        "dimension_k",
        int(get_nested(config, "subspace", "dimension_k", default=5)),
    )
    smooth_window = int(get_nested(config, "subspace", "smooth_window", default=5))
    min_visibility = float(get_nested(config, "pose", "min_visibility", default=0.0))

    normalized_sequence, motion_matrix, normalization_summary, matrix_summary, subspace_model = (
        _build_profile_subspace(
            sequence,
            profile,
            target_frames=target_frames,
            dimension_k=dimension_k,
            smooth_window=smooth_window,
            min_visibility=min_visibility,
        )
    )
    quality = _query_quality(sequence, normalized_sequence, profile, min_visibility=min_visibility)

    np.save(sequence_path, sequence)
    np.save(normalized_path, normalized_sequence)
    np.save(matrix_path, motion_matrix)
    np.save(basis_path, subspace_model.basis)
    np.save(mean_path, subspace_model.mean)

    metadata = {
        "id": query_id,
        "motion_type": motion_type,
        "timestamp_utc": timestamp.isoformat(),
        "keypoint_names": list(KEYPOINT_NAMES),
        "sequence_shape": list(sequence.shape),
        "sequence_path": str(sequence_path),
        "normalized_sequence_path": str(normalized_path),
        "motion_matrix_path": str(matrix_path),
        "subspace_basis_path": str(basis_path),
        "subspace_mean_path": str(mean_path),
        "metadata_path": str(metadata_path),
        "motion_profile": _profile_metadata(profile),
        "quality": quality,
        "normalization": {
            "input_frames": normalization_summary.input_frames,
            "output_frames": normalization_summary.output_frames,
            "feature_dimensions": matrix_summary.feature_dimensions,
            "interpolated_values": normalization_summary.interpolated_values,
            "median_torso_scale": normalization_summary.median_torso_scale,
            "smooth_window": smooth_window,
            "min_visibility": min_visibility,
        },
        "subspace": {
            "dimension_k": int(subspace_model.basis.shape[1]),
            "basis_shape": list(subspace_model.basis.shape),
            "explained_variance_ratio": subspace_model.explained_variance_ratio.tolist(),
            "singular_values": subspace_model.singular_values.tolist(),
        },
        "raw_video_stored": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "id": query_id,
        "metadata_path": str(metadata_path),
        "sequence_path": str(sequence_path),
        "normalized_sequence_path": str(normalized_path),
        "motion_matrix_path": str(matrix_path),
        "subspace_basis_path": str(basis_path),
        "subspace_basis": subspace_model.basis,
        "motion_profile": metadata["motion_profile"],
        "quality": metadata["quality"],
        "normalization": metadata["normalization"],
        "subspace": metadata["subspace"],
    }


def _build_profile_subspace(
    sequence: np.ndarray,
    profile: MotionProfile,
    *,
    target_frames: int,
    dimension_k: int,
    smooth_window: int,
    min_visibility: float,
):
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
    subspace_model = build_motion_subspace(motion_matrix, dimension_k=dimension_k)
    return normalized_sequence, motion_matrix, normalization_summary, matrix_summary, subspace_model


def load_registered_candidates(config: dict[str, Any], *, motion_type: str) -> list[dict[str, Any]]:
    database_path = Path(get_nested(config, "data", "database_path", default="data/database.json"))
    if not database_path.exists():
        return []

    database = json.loads(database_path.read_text(encoding="utf-8"))
    registrations = database.get("registrations", []) if isinstance(database, dict) else database
    candidates: list[dict[str, Any]] = []
    profile = profile_for_motion(motion_type)
    target_frames = int(get_nested(config, "subspace", "num_frames", default=120))
    dimension_k = _motion_int(
        config,
        motion_type,
        "dimension_k",
        int(get_nested(config, "subspace", "dimension_k", default=5)),
    )
    smooth_window = int(get_nested(config, "subspace", "smooth_window", default=5))
    min_visibility = float(get_nested(config, "pose", "min_visibility", default=0.0))
    for entry in registrations:
        if _normalize_motion_type(entry.get("motion_type", "")) != _normalize_motion_type(motion_type):
            continue
        sequence_path = entry.get("sequence_path")
        if not sequence_path or not Path(sequence_path).exists():
            continue
        sequence = np.load(sequence_path)
        _, _, _, _, subspace_model = _build_profile_subspace(
            sequence,
            profile,
            target_frames=target_frames,
            dimension_k=dimension_k,
            smooth_window=smooth_window,
            min_visibility=min_visibility,
        )
        candidates.append(
            {
                "id": entry.get("id", ""),
                "nickname": entry.get("nickname", "Unknown"),
                "avatar": entry.get("avatar", ""),
                "motion_type": entry.get("motion_type", ""),
                "metadata_path": entry.get("metadata_path", ""),
                "sequence_path": sequence_path,
                "normalized_sequence_path": entry.get("normalized_sequence_path", ""),
                "subspace_basis_path": entry.get("subspace_basis_path", ""),
                "subspace_basis": subspace_model.basis,
                "motion_profile": _profile_metadata(profile),
                "quality": entry.get("quality", {}),
            }
        )
    return candidates


def rank_candidates(
    query_basis: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    similarity_alpha: float = 1.5,
) -> list[dict[str, Any]]:
    ranked = []
    for candidate in candidates:
        comparison = compare_subspaces(
            query_basis,
            candidate["subspace_basis"],
            similarity_alpha=similarity_alpha,
        )
        ranked.append(
            {
                "id": candidate["id"],
                "nickname": candidate["nickname"],
                "avatar": candidate["avatar"],
                "motion_type": candidate["motion_type"],
                "similarity": comparison.similarity,
                "raw_similarity": comparison.similarity,
                "mean_angle_degrees": comparison.mean_angle_degrees,
                "projection_distance": comparison.projection_distance,
                "canonical_angles_degrees": comparison.canonical_angles_degrees.tolist(),
                "metadata_path": candidate["metadata_path"],
                "sequence_path": candidate["sequence_path"],
                "normalized_sequence_path": candidate["normalized_sequence_path"],
                "quality": candidate["quality"],
            }
        )
    ranked.sort(key=lambda item: item["similarity"], reverse=True)
    return ranked


def _public_query(query: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": query["id"],
        "metadata_path": query["metadata_path"],
        "sequence_path": query["sequence_path"],
        "normalized_sequence_path": query["normalized_sequence_path"],
        "motion_matrix_path": query["motion_matrix_path"],
        "subspace_basis_path": query["subspace_basis_path"],
        "motion_profile": query["motion_profile"],
        "quality": query["quality"],
        "normalization": query["normalization"],
        "subspace": query["subspace"],
    }


def _write_query_result(metadata_path: str, result: dict[str, Any]) -> None:
    path = Path(metadata_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["match_result"] = {
        "candidate_count": result["candidate_count"],
        "top_matches": result["top_matches"],
        "accepted": result["accepted"],
        "min_accept_similarity": result["min_accept_similarity"],
        "no_match_reason": result["no_match_reason"],
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _build_query_id(timestamp: datetime, motion_type: str) -> str:
    timestamp_part = timestamp.strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp_part}_query_{_slugify(motion_type) or 'motion'}"


def _normalize_motion_type(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:48]


def _profile_metadata(profile: MotionProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "keypoint_names": list(profile.keypoint_names),
        "center_keypoint_names": list(profile.center_keypoint_names),
        "scale_from_keypoint_names": list(profile.scale_from_keypoint_names),
        "scale_to_keypoint_names": list(profile.scale_to_keypoint_names),
    }


def _query_quality(
    sequence: np.ndarray,
    normalized_sequence: np.ndarray,
    profile: MotionProfile,
    *,
    min_visibility: float,
) -> dict[str, Any]:
    selected_indices = [_keypoint_index(name) for name in profile.keypoint_names]
    selected_visibility = sequence[:, selected_indices, 2]
    visibility_ratio = float((selected_visibility >= min_visibility).mean())

    scale_from_indices = [_keypoint_index(name) for name in profile.scale_from_keypoint_names]
    scale_to_indices = [_keypoint_index(name) for name in profile.scale_to_keypoint_names]
    scale_from = sequence[:, scale_from_indices, :2].mean(axis=1)
    scale_to = sequence[:, scale_to_indices, :2].mean(axis=1)
    scale_values = np.linalg.norm(scale_from - scale_to, axis=1)
    scale_visibility = (
        sequence[:, scale_from_indices, 2].mean(axis=1) >= min_visibility
    ) & (sequence[:, scale_to_indices, 2].mean(axis=1) >= min_visibility)
    median_scale = float(np.median(scale_values[scale_visibility])) if scale_visibility.any() else 0.0

    wrist_ranges: list[float] = []
    wrist_travel: list[float] = []
    profile_names = list(profile.keypoint_names)
    for wrist_name in ("left_wrist", "right_wrist"):
        if wrist_name not in profile_names:
            continue
        wrist_index = profile_names.index(wrist_name)
        wrist_points = normalized_sequence[:, wrist_index, :]
        wrist_ranges.append(
            float(np.linalg.norm(np.nanmax(wrist_points, axis=0) - np.nanmin(wrist_points, axis=0)))
        )
        wrist_travel.append(float(np.linalg.norm(np.diff(wrist_points, axis=0), axis=1).sum()))

    max_wrist_range = max(wrist_ranges, default=0.0)
    max_wrist_travel = max(wrist_travel, default=0.0)
    if profile.name == "upper_body_wave":
        visibility_confidence = _ratio(visibility_ratio, 0.8)
        scale_confidence = _ratio(median_scale, 0.16)
        activity_confidence = _ratio(max_wrist_range, 0.5)
        confidence = visibility_confidence * scale_confidence * activity_confidence
    else:
        visibility_confidence = _ratio(visibility_ratio, 0.7)
        scale_confidence = _ratio(median_scale, 0.08)
        activity_confidence = 1.0
        confidence = visibility_confidence * scale_confidence

    return {
        "profile_name": profile.name,
        "selected_visibility_ratio": visibility_ratio,
        "median_scale": median_scale,
        "max_wrist_range": max_wrist_range,
        "max_wrist_travel": max_wrist_travel,
        "visibility_confidence": visibility_confidence,
        "scale_confidence": scale_confidence,
        "activity_confidence": activity_confidence,
        "confidence": max(0.0, min(1.0, confidence)),
    }


def _apply_query_quality(
    ranked: list[dict[str, Any]],
    quality: dict[str, Any],
) -> list[dict[str, Any]]:
    confidence = float(quality.get("confidence", 1.0))
    adjusted = []
    for item in ranked:
        updated = dict(item)
        raw_similarity = float(updated.get("raw_similarity", updated.get("similarity", 0.0)))
        updated["raw_similarity"] = raw_similarity
        updated["query_quality_confidence"] = confidence
        updated["similarity"] = raw_similarity * confidence
        adjusted.append(updated)
    adjusted.sort(key=lambda item: item["similarity"], reverse=True)
    return adjusted


def _no_match_reason(motion_type: str, quality: dict[str, Any]) -> str:
    if quality.get("profile_name") == "upper_body_wave":
        if float(quality.get("scale_confidence", 1.0)) < 0.75:
            return "Upper body was too small or unstable in frame for a reliable Wave match."
        if float(quality.get("visibility_confidence", 1.0)) < 0.75:
            return "Not enough shoulders, elbows, and wrists were visible for a reliable Wave match."
        if float(quality.get("activity_confidence", 1.0)) < 0.75:
            return "No clear wrist wave motion was detected."
        return "No strong Wave doppelganger match was found."
    return f"No strong {motion_type} doppelganger match was found."


def _ratio(value: float, target: float) -> float:
    if target <= 0:
        return 1.0
    return max(0.0, min(1.0, float(value) / float(target)))


def _keypoint_index(name: str) -> int:
    try:
        return KEYPOINT_NAMES.index(name)
    except ValueError as exc:
        raise ValueError(f"Unknown keypoint name: {name}") from exc


def _motion_int(config: dict[str, Any], motion_type: str, key: str, default: int) -> int:
    value = _motion_config_value(config, motion_type, key, default)
    return int(value)


def _motion_float(config: dict[str, Any], motion_type: str, key: str, default: float) -> float:
    value = _motion_config_value(config, motion_type, key, default)
    return float(value)


def _motion_config_value(config: dict[str, Any], motion_type: str, key: str, default: Any) -> Any:
    profiles = get_nested(config, "motion_profiles", default={})
    if not isinstance(profiles, dict):
        return default
    normalized = _normalize_motion_type(motion_type)
    for profile_name, profile_config in profiles.items():
        if _normalize_motion_type(str(profile_name)) == normalized and isinstance(profile_config, dict):
            return profile_config.get(key, default)
    return default
