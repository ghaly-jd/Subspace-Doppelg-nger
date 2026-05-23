from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from shared.config import get_nested
from shared.pose_schema import HAND_KEYPOINT_NAMES, KEYPOINT_NAMES, HandKeypoints, PoseKeypoints
from shared.ritual_processor import build_ritual_artifacts
from shared.ritual_schema import is_ritual_motion
from shared.skeleton_normalizer import normalize_keypoint_sequence, normalize_to_motion_matrix
from shared.subspace import build_motion_subspace, compare_subspaces


def save_registration(
    config: dict[str, Any],
    *,
    nickname: str,
    avatar: str,
    motion_type: str,
    frames: list[PoseKeypoints],
    quality: dict[str, Any],
    hand_frames: list[HandKeypoints] | None = None,
) -> dict[str, Any]:
    registrations_dir = Path(
        get_nested(config, "data", "registrations_dir", default="data/registrations")
    )
    database_path = Path(
        get_nested(config, "data", "database_path", default="data/database.json")
    )
    registrations_dir.mkdir(parents=True, exist_ok=True)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)
    registration_id = _build_registration_id(timestamp, nickname, motion_type)
    sequence_path = registrations_dir / f"{registration_id}_sequence.npy"
    metadata_path = registrations_dir / f"{registration_id}.json"
    suffix = 2
    while sequence_path.exists() or metadata_path.exists():
        registration_id = f"{_build_registration_id(timestamp, nickname, motion_type)}-{suffix}"
        sequence_path = registrations_dir / f"{registration_id}_sequence.npy"
        metadata_path = registrations_dir / f"{registration_id}.json"
        suffix += 1

    sequence = keypoint_frames_to_array(frames)
    np.save(sequence_path, sequence)
    hand_sequence_path = registrations_dir / f"{registration_id}_hand_sequence.npy"
    hand_sequence = None
    if hand_frames is not None:
        hand_sequence = hand_frames_to_array(hand_frames)
        np.save(hand_sequence_path, hand_sequence)

    metadata = {
        "id": registration_id,
        "nickname": nickname,
        "avatar": avatar,
        "motion_type": motion_type,
        "timestamp_utc": timestamp.isoformat(),
        "keypoint_names": list(KEYPOINT_NAMES),
        "sequence_shape": list(sequence.shape),
        "sequence_path": str(sequence_path),
        "metadata_path": str(metadata_path),
        "quality": quality,
        "raw_video_stored": False,
    }
    if hand_sequence is not None:
        metadata["hand_keypoint_names"] = list(HAND_KEYPOINT_NAMES)
        metadata["hand_sequence_shape"] = list(hand_sequence.shape)
        metadata["hand_sequence_path"] = str(hand_sequence_path)
    if is_ritual_motion(motion_type):
        _add_ritual_artifacts(config, metadata, metadata_path, sequence)
    else:
        _add_subspace_artifacts(config, metadata, metadata_path, sequence)

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _append_database_record(database_path, metadata)
    return metadata


def keypoint_frames_to_array(frames: list[PoseKeypoints]) -> np.ndarray:
    sequence = np.full((len(frames), len(KEYPOINT_NAMES), 3), np.nan, dtype=np.float32)
    for frame_index, keypoints in enumerate(frames):
        for keypoint_index, name in enumerate(KEYPOINT_NAMES):
            keypoint = keypoints.get(name)
            if keypoint is None:
                continue
            sequence[frame_index, keypoint_index] = (
                float(keypoint.x),
                float(keypoint.y),
                float(keypoint.visibility),
            )
    return sequence


def hand_frames_to_array(frames: list[HandKeypoints]) -> np.ndarray:
    sequence = np.full((len(frames), len(HAND_KEYPOINT_NAMES), 3), np.nan, dtype=np.float32)
    for frame_index, keypoints in enumerate(frames):
        for keypoint_index, name in enumerate(HAND_KEYPOINT_NAMES):
            keypoint = keypoints.get(name)
            if keypoint is None:
                continue
            sequence[frame_index, keypoint_index] = (
                float(keypoint.x),
                float(keypoint.y),
                float(keypoint.visibility),
            )
    return sequence


def _append_database_record(database_path: Path, metadata: dict[str, Any]) -> None:
    if database_path.exists():
        database = json.loads(database_path.read_text(encoding="utf-8"))
    else:
        database = {"version": 1, "registrations": []}

    if isinstance(database, list):
        database = {"version": 1, "registrations": database}

    registrations = database.setdefault("registrations", [])
    entry = {
        "id": metadata["id"],
        "nickname": metadata["nickname"],
        "avatar": metadata["avatar"],
        "motion_type": metadata["motion_type"],
        "timestamp_utc": metadata["timestamp_utc"],
        "metadata_path": metadata["metadata_path"],
        "sequence_path": metadata["sequence_path"],
        "quality": metadata["quality"],
    }
    if "hand_sequence_path" in metadata:
        entry["hand_sequence_path"] = metadata["hand_sequence_path"]
        entry["hand_sequence_shape"] = metadata.get("hand_sequence_shape", [])
    for key in (
        "normalized_sequence_path",
        "motion_matrix_path",
        "subspace_basis_path",
        "subspace_mean_path",
        "normalization",
        "subspace",
        "ritual",
        "processing_error",
    ):
        if key in metadata:
            entry[key] = metadata[key]
    registrations.append(entry)
    database_path.write_text(json.dumps(database, indent=2), encoding="utf-8")


def _add_ritual_artifacts(
    config: dict[str, Any],
    metadata: dict[str, Any],
    metadata_path: Path,
    sequence: np.ndarray,
) -> None:
    try:
        metadata["ritual"] = build_ritual_artifacts(
            config,
            sequence=sequence,
            base_path=metadata_path.with_suffix(""),
            artifact_id=metadata["id"],
            save_arrays=True,
        )
    except Exception as exc:
        metadata["processing_error"] = str(exc)


def _add_subspace_artifacts(
    config: dict[str, Any],
    metadata: dict[str, Any],
    metadata_path: Path,
    sequence: np.ndarray,
) -> None:
    try:
        target_frames = int(get_nested(config, "subspace", "num_frames", default=120))
        dimension_k = int(get_nested(config, "subspace", "dimension_k", default=5))
        similarity_alpha = float(get_nested(config, "subspace", "similarity_alpha", default=1.5))
        smooth_window = int(get_nested(config, "subspace", "smooth_window", default=5))
        min_visibility = float(get_nested(config, "pose", "min_visibility", default=0.0))

        normalized_sequence, normalization_summary = normalize_keypoint_sequence(
            sequence,
            target_frames=target_frames,
            smooth_window=smooth_window,
            min_visibility=min_visibility,
        )
        motion_matrix, matrix_summary = normalize_to_motion_matrix(
            sequence,
            target_frames=target_frames,
            smooth_window=smooth_window,
            min_visibility=min_visibility,
        )
        subspace_model = build_motion_subspace(motion_matrix, dimension_k=dimension_k)
        self_comparison = compare_subspaces(
            subspace_model.basis,
            subspace_model.basis,
            similarity_alpha=similarity_alpha,
        )

        base_path = metadata_path.with_suffix("")
        normalized_path = base_path.with_name(f"{metadata['id']}_normalized.npy")
        matrix_path = base_path.with_name(f"{metadata['id']}_motion_matrix.npy")
        basis_path = base_path.with_name(f"{metadata['id']}_subspace_basis.npy")
        mean_path = base_path.with_name(f"{metadata['id']}_subspace_mean.npy")

        np.save(normalized_path, normalized_sequence)
        np.save(matrix_path, motion_matrix)
        np.save(basis_path, subspace_model.basis)
        np.save(mean_path, subspace_model.mean)

        metadata["normalized_sequence_path"] = str(normalized_path)
        metadata["motion_matrix_path"] = str(matrix_path)
        metadata["subspace_basis_path"] = str(basis_path)
        metadata["subspace_mean_path"] = str(mean_path)
        metadata["normalization"] = {
            "input_frames": normalization_summary.input_frames,
            "output_frames": normalization_summary.output_frames,
            "feature_dimensions": matrix_summary.feature_dimensions,
            "interpolated_values": normalization_summary.interpolated_values,
            "median_torso_scale": normalization_summary.median_torso_scale,
            "smooth_window": smooth_window,
            "min_visibility": min_visibility,
        }
        metadata["subspace"] = {
            "dimension_k": int(subspace_model.basis.shape[1]),
            "basis_shape": list(subspace_model.basis.shape),
            "explained_variance_ratio": subspace_model.explained_variance_ratio.tolist(),
            "singular_values": subspace_model.singular_values.tolist(),
            "self_similarity": self_comparison.similarity,
        }
    except Exception as exc:
        metadata["processing_error"] = str(exc)


def _build_registration_id(
    timestamp: datetime,
    nickname: str,
    motion_type: str,
) -> str:
    timestamp_part = timestamp.strftime("%Y%m%dT%H%M%SZ")
    name_part = _slugify(nickname) or "anonymous"
    motion_part = _slugify(motion_type) or "motion"
    return f"{timestamp_part}_{motion_part}_{name_part}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48]
