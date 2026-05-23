from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from shared.config import get_nested
from shared.dtw_tools import speed_curve_dtw_similarity
from shared.motion_features import (
    combined_energy_balance_similarity,
    feature_similarity,
)
from shared.pose_schema import HAND_KEYPOINT_NAMES, KEYPOINT_NAMES, HandKeypoints, PoseKeypoints
from shared.registration_storage import hand_frames_to_array, keypoint_frames_to_array
from shared.ritual_processor import build_ritual_artifacts
from shared.ritual_schema import is_ritual_motion, ritual_motion_label, ritual_scoring_weights
from shared.subspace import compare_subspaces


def find_ritual_matches(
    config: dict[str, Any],
    *,
    frames: list[PoseKeypoints],
    top_k: int = 3,
    exclude_ids: set[str] | None = None,
    hand_frames: list[HandKeypoints] | None = None,
) -> dict[str, Any]:
    query = build_ritual_query_artifacts(config, frames=frames, hand_frames=hand_frames)
    candidates = load_ritual_candidates(config, exclude_ids=exclude_ids)
    ranked = rank_ritual_candidates(config, query=query, candidates=candidates)

    min_accept_similarity = float(get_nested(config, "ritual", "min_accept_similarity", default=55.0))
    query_confidence = float(query["ritual"]["full"]["quality"].get("confidence", 0.0))
    full_min_confidence = float(
        get_nested(config, "ritual_quality", "full_min_confidence", default=0.45)
    )
    phase_gates_passed = _best_phase_gates_passed(config, ranked[0] if ranked else None)
    accepted = (
        bool(ranked)
        and ranked[0]["similarity"] >= min_accept_similarity
        and query_confidence >= full_min_confidence
        and phase_gates_passed
    )

    result = {
        "motion_type": ritual_motion_label(config),
        "ritual_result": True,
        "query": _public_query(query),
        "candidate_count": len(candidates),
        "top_matches": ranked[:top_k],
        "overall_match": ranked[0] if ranked else None,
        "phase_matches": _phase_matches(ranked),
        "style_matches": _style_matches(ranked),
        "accepted": accepted,
        "min_accept_similarity": min_accept_similarity,
        "no_match_reason": ""
        if accepted
        else _no_match_reason(
            query_confidence,
            bool(ranked),
            full_min_confidence=full_min_confidence,
            phase_gates_passed=phase_gates_passed,
        ),
    }
    _write_query_result(query["metadata_path"], result)
    return result


def build_ritual_query_artifacts(
    config: dict[str, Any],
    *,
    frames: list[PoseKeypoints],
    hand_frames: list[HandKeypoints] | None = None,
) -> dict[str, Any]:
    sessions_dir = Path(get_nested(config, "data", "sessions_dir", default="data/sessions"))
    sessions_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)
    query_id = _build_query_id(timestamp)
    sequence_path = sessions_dir / f"{query_id}_sequence.npy"
    metadata_path = sessions_dir / f"{query_id}.json"
    suffix = 2
    while sequence_path.exists() or metadata_path.exists():
        query_id = f"{_build_query_id(timestamp)}-{suffix}"
        sequence_path = sessions_dir / f"{query_id}_sequence.npy"
        metadata_path = sessions_dir / f"{query_id}.json"
        suffix += 1

    sequence = keypoint_frames_to_array(frames)
    np.save(sequence_path, sequence)
    hand_sequence_path = sessions_dir / f"{query_id}_hand_sequence.npy"
    hand_sequence = None
    if hand_frames is not None:
        hand_sequence = hand_frames_to_array(hand_frames)
        np.save(hand_sequence_path, hand_sequence)
    ritual = build_ritual_artifacts(
        config,
        sequence=sequence,
        base_path=metadata_path.with_suffix(""),
        artifact_id=query_id,
        save_arrays=True,
    )
    metadata = {
        "id": query_id,
        "motion_type": ritual_motion_label(config),
        "timestamp_utc": timestamp.isoformat(),
        "keypoint_names": list(KEYPOINT_NAMES),
        "sequence_shape": list(sequence.shape),
        "sequence_path": str(sequence_path),
        "metadata_path": str(metadata_path),
        "ritual": ritual,
        "raw_video_stored": False,
    }
    if hand_sequence is not None:
        metadata["hand_keypoint_names"] = list(HAND_KEYPOINT_NAMES)
        metadata["hand_sequence_shape"] = list(hand_sequence.shape)
        metadata["hand_sequence_path"] = str(hand_sequence_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_ritual_candidates(
    config: dict[str, Any],
    *,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    database_path = Path(get_nested(config, "data", "database_path", default="data/database.json"))
    if not database_path.exists():
        return []

    database = json.loads(database_path.read_text(encoding="utf-8"))
    registrations = database.get("registrations", []) if isinstance(database, dict) else database
    candidates = []
    exclude_ids = exclude_ids or set()
    for entry in registrations:
        if str(entry.get("id", "")) in exclude_ids:
            continue
        if not is_ritual_motion(str(entry.get("motion_type", ""))):
            continue
        ritual = entry.get("ritual")
        if not isinstance(ritual, dict):
            metadata_path = entry.get("metadata_path")
            if metadata_path and Path(metadata_path).exists():
                metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
                ritual = metadata.get("ritual", {})
        if not isinstance(ritual, dict) or not ritual.get("full"):
            continue
        candidates.append(
            {
                "id": entry.get("id", ""),
                "nickname": entry.get("nickname", "Unknown"),
                "avatar": entry.get("avatar", ""),
                "metadata_path": entry.get("metadata_path", ""),
                "sequence_path": entry.get("sequence_path", ""),
                "ritual": ritual,
            }
        )
    return candidates


def rank_ritual_candidates(
    config: dict[str, Any],
    *,
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    weights = ritual_scoring_weights(config)
    ranked = []
    query_ritual = query["ritual"]
    for candidate in candidates:
        candidate_ritual = candidate["ritual"]
        full_coordinate = _subspace_similarity(
            query_ritual["full"]["coordinate_subspace_basis_path"],
            candidate_ritual["full"]["coordinate_subspace_basis_path"],
            config=config,
        )
        full_velocity = _subspace_similarity(
            query_ritual["full"]["velocity_subspace_basis_path"],
            candidate_ritual["full"]["velocity_subspace_basis_path"],
            config=config,
        )
        phase_scores = _compare_phases(config, query_ritual, candidate_ritual)
        average_phase = _weighted_phase_average(phase_scores)
        rhythm = feature_similarity(
            query_ritual["full"]["features"],
            candidate_ritual["full"]["features"],
            categories=("rhythm",),
        )
        joint_angles = feature_similarity(
            query_ritual["full"]["features"],
            candidate_ritual["full"]["features"],
            categories=("joint_angles",),
        )
        energy_balance = combined_energy_balance_similarity(
            query_ritual["full"]["features"],
            candidate_ritual["full"]["features"],
        )
        overall_weights = weights["overall"]
        similarity = (
            overall_weights["full_coordinate_subspace"] * full_coordinate["similarity"]
            + overall_weights["average_phase"] * average_phase
            + overall_weights["full_velocity_subspace"] * full_velocity["similarity"]
            + overall_weights["rhythm"] * rhythm
            + overall_weights["joint_angles"] * joint_angles
            + overall_weights["energy_balance"] * energy_balance
        )
        ranked.append(
            {
                "id": candidate["id"],
                "nickname": candidate["nickname"],
                "avatar": candidate["avatar"],
                "motion_type": ritual_motion_label(config),
                "similarity": similarity,
                "full_coordinate_similarity": full_coordinate["similarity"],
                "full_velocity_similarity": full_velocity["similarity"],
                "full_velocity_mean_angle_degrees": full_velocity["mean_angle_degrees"],
                "full_velocity_projection_distance": full_velocity["projection_distance"],
                "full_velocity_canonical_angles_degrees": full_velocity["canonical_angles_degrees"],
                "average_phase_similarity": average_phase,
                "rhythm_similarity": rhythm,
                "joint_angle_similarity": joint_angles,
                "energy_balance_similarity": energy_balance,
                "mean_angle_degrees": full_coordinate["mean_angle_degrees"],
                "projection_distance": full_coordinate["projection_distance"],
                "canonical_angles_degrees": full_coordinate["canonical_angles_degrees"],
                "metadata_path": candidate["metadata_path"],
                "sequence_path": candidate["sequence_path"],
                "phase_scores": phase_scores,
                "style_scores": {
                    "velocity": full_velocity["similarity"],
                    "rhythm": rhythm,
                    "energy": feature_similarity(
                        query_ritual["full"]["features"],
                        candidate_ritual["full"]["features"],
                        categories=("energy",),
                    ),
                    "balance": feature_similarity(
                        query_ritual["full"]["features"],
                        candidate_ritual["full"]["features"],
                        categories=("balance",),
                    ),
                    "gesture": _gesture_similarity(query_ritual, candidate_ritual),
                },
            }
        )
    ranked.sort(key=lambda item: item["similarity"], reverse=True)
    return ranked


def _compare_phases(
    config: dict[str, Any],
    query_ritual: dict[str, Any],
    candidate_ritual: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    phase_weights = ritual_scoring_weights(config)["phase"]
    query_phases = {phase["phase_id"]: phase for phase in query_ritual.get("phase_segments", [])}
    candidate_phases = {
        phase["phase_id"]: phase for phase in candidate_ritual.get("phase_segments", [])
    }
    scores: dict[str, dict[str, Any]] = {}
    for phase_id, query_phase in query_phases.items():
        candidate_phase = candidate_phases.get(phase_id)
        if not candidate_phase:
            continue
        coordinate = _subspace_similarity(
            query_phase["coordinate_subspace_basis_path"],
            candidate_phase["coordinate_subspace_basis_path"],
            config=config,
        )
        velocity = _subspace_similarity(
            query_phase["velocity_subspace_basis_path"],
            candidate_phase["velocity_subspace_basis_path"],
            config=config,
        )
        rhythm = feature_similarity(
            query_phase["features"],
            candidate_phase["features"],
            categories=("rhythm",),
        )
        joint_angles = feature_similarity(
            query_phase["features"],
            candidate_phase["features"],
            categories=("joint_angles",),
        )
        speed_dtw = speed_curve_dtw_similarity(
            query_phase.get("features", {}).get("curves", {}).get("speed_curve", []),
            candidate_phase.get("features", {}).get("curves", {}).get("speed_curve", []),
            alpha=float(get_nested(config, "ritual_scoring", "phase", "speed_dtw_alpha", default=3.0)),
        )
        raw_similarity = (
            phase_weights["coordinate_subspace"] * coordinate["similarity"]
            + phase_weights["velocity_subspace"] * velocity["similarity"]
            + phase_weights["speed_dtw"] * speed_dtw
            + phase_weights["rhythm"] * rhythm
            + phase_weights["joint_angles"] * joint_angles
        )
        quality_weight = _phase_quality_weight(query_phase, candidate_phase)
        gate_passed, gate_reason = _phase_quality_gate(config, phase_id, query_phase, candidate_phase)
        similarity = raw_similarity * quality_weight if gate_passed else 0.0
        scores[phase_id] = {
            "phase_id": phase_id,
            "label": query_phase.get("label", phase_id),
            "similarity": similarity,
            "raw_similarity": raw_similarity,
            "quality_weight": quality_weight,
            "gate_passed": gate_passed,
            "gate_reason": gate_reason,
            "coordinate_similarity": coordinate["similarity"],
            "velocity_similarity": velocity["similarity"],
            "speed_dtw_similarity": speed_dtw,
            "velocity_mean_angle_degrees": velocity["mean_angle_degrees"],
            "velocity_projection_distance": velocity["projection_distance"],
            "rhythm_similarity": rhythm,
            "joint_angle_similarity": joint_angles,
            "mean_angle_degrees": coordinate["mean_angle_degrees"],
            "projection_distance": coordinate["projection_distance"],
            "canonical_angles_degrees": coordinate["canonical_angles_degrees"],
        }
    return scores


def _phase_matches(ranked: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    phase_ids = sorted({phase_id for item in ranked for phase_id in item.get("phase_scores", {})})
    matches = {}
    for phase_id in phase_ids:
        best = max(
            (item for item in ranked if phase_id in item.get("phase_scores", {})),
            key=lambda item: item["phase_scores"][phase_id]["similarity"],
        )
        phase = best["phase_scores"][phase_id]
        matches[phase_id] = {
            "phase_id": phase_id,
            "label": phase.get("label", phase_id),
            "nickname": best["nickname"],
            "avatar": best["avatar"],
            "similarity": phase["similarity"],
            "raw_similarity": phase.get("raw_similarity", phase["similarity"]),
            "quality_weight": phase.get("quality_weight", 1.0),
            "gate_passed": phase.get("gate_passed", True),
            "gate_reason": phase.get("gate_reason", ""),
            "coordinate_similarity": phase.get("coordinate_similarity", 0.0),
            "velocity_similarity": phase.get("velocity_similarity", 0.0),
            "speed_dtw_similarity": phase.get("speed_dtw_similarity", 0.0),
            "mean_angle_degrees": phase.get("mean_angle_degrees", 0.0),
            "projection_distance": phase.get("projection_distance", 0.0),
            "candidate_id": best["id"],
            "metadata_path": best["metadata_path"],
        }
    return matches


def _style_matches(ranked: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    style_names = ("velocity", "rhythm", "energy", "balance", "gesture")
    matches = {}
    for style_name in style_names:
        best = max(ranked, key=lambda item: item.get("style_scores", {}).get(style_name, 0.0), default=None)
        if best is None:
            continue
        matches[style_name] = {
            "style": style_name,
            "nickname": best["nickname"],
            "avatar": best["avatar"],
            "similarity": best.get("style_scores", {}).get(style_name, 0.0),
            "candidate_id": best["id"],
        }
    return matches


def _weighted_phase_average(phase_scores: dict[str, dict[str, Any]]) -> float:
    if not phase_scores:
        return 0.0
    weighted_sum = 0.0
    weight_sum = 0.0
    for phase in phase_scores.values():
        weight = float(phase.get("quality_weight", 1.0))
        weighted_sum += float(phase.get("similarity", 0.0)) * weight
        weight_sum += weight
    if weight_sum <= 1e-9:
        return 0.0
    return weighted_sum / weight_sum


def _phase_quality_weight(
    query_phase: dict[str, Any],
    candidate_phase: dict[str, Any],
) -> float:
    query_confidence = float(query_phase.get("quality", {}).get("confidence", 1.0))
    candidate_confidence = float(candidate_phase.get("quality", {}).get("confidence", 1.0))
    return max(0.15, min(1.0, float(np.sqrt(max(0.0, query_confidence * candidate_confidence)))))


def _phase_quality_gate(
    config: dict[str, Any],
    phase_id: str,
    query_phase: dict[str, Any],
    candidate_phase: dict[str, Any],
) -> tuple[bool, str]:
    query_quality = query_phase.get("quality", {})
    candidate_quality = candidate_phase.get("quality", {})
    min_confidence = float(get_nested(config, "ritual_quality", "phase_min_confidence", default=0.25))
    min_activity = float(get_nested(config, "ritual_quality", "phase_min_activity", default=0.0))
    per_phase = get_nested(config, "ritual_quality", "per_phase", phase_id, default={})
    if not isinstance(per_phase, dict):
        per_phase = {}

    confidence = min(
        float(query_quality.get("confidence", 1.0)),
        float(candidate_quality.get("confidence", 1.0)),
    )
    if confidence < min_confidence:
        return False, "low tracking confidence"

    phase_min_activity = float(per_phase.get("min_activity", min_activity))
    if not _metric_at_least(query_quality, "activity_score", phase_min_activity):
        return False, "query phase activity too low"
    if not _metric_at_least(candidate_quality, "activity_score", phase_min_activity):
        return False, "candidate phase activity too low"

    for metric_name, threshold_key in (
        ("wrist_activity", "min_wrist_activity"),
        ("hip_drop", "min_hip_drop"),
        ("knee_lift", "min_knee_lift"),
        ("clap_peak_count", "min_clap_peaks"),
    ):
        if threshold_key not in per_phase:
            continue
        threshold = float(per_phase[threshold_key])
        if not _metric_at_least(query_quality, metric_name, threshold):
            return False, f"query {metric_name} below gate"
        if not _metric_at_least(candidate_quality, metric_name, threshold):
            return False, f"candidate {metric_name} below gate"

    if "max_wrist_distance_min" in per_phase:
        threshold = float(per_phase["max_wrist_distance_min"])
        if not _metric_at_most(query_quality, "wrist_distance_min", threshold):
            return False, "query clap contact not clear"
        if not _metric_at_most(candidate_quality, "wrist_distance_min", threshold):
            return False, "candidate clap contact not clear"

    return True, ""


def _metric_at_least(quality: dict[str, Any], key: str, threshold: float) -> bool:
    if threshold <= 0.0 or key not in quality:
        return True
    return float(quality.get(key, 0.0)) >= threshold


def _metric_at_most(quality: dict[str, Any], key: str, threshold: float) -> bool:
    if threshold <= 0.0 or key not in quality:
        return True
    return float(quality.get(key, threshold + 1.0)) <= threshold


def _best_phase_gates_passed(config: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return False
    if not bool(get_nested(config, "ritual_quality", "enforce_required_phase_gates", default=True)):
        return True
    required = get_nested(config, "ritual_quality", "required_phases", default=[])
    required_phase_ids = set(required) if isinstance(required, list) else set()
    if not required_phase_ids:
        required_phase_ids = set(best.get("phase_scores", {}).keys())
    for phase_id, phase in best.get("phase_scores", {}).items():
        if phase_id in required_phase_ids and not bool(phase.get("gate_passed", True)):
            return False
    return True


def _gesture_similarity(query_ritual: dict[str, Any], candidate_ritual: dict[str, Any]) -> float:
    gesture_phase_ids = ("wave_hello", "clap_twice", "point_forward")
    query_phases = {phase["phase_id"]: phase for phase in query_ritual.get("phase_segments", [])}
    candidate_phases = {
        phase["phase_id"]: phase for phase in candidate_ritual.get("phase_segments", [])
    }
    scores = []
    for phase_id in gesture_phase_ids:
        if phase_id not in query_phases or phase_id not in candidate_phases:
            continue
        scores.append(
            feature_similarity(
                query_phases[phase_id]["features"],
                candidate_phases[phase_id]["features"],
                categories=("gesture", "rhythm"),
            )
        )
    return float(np.mean(scores)) if scores else 0.0


def _subspace_similarity(path_a: str, path_b: str, *, config: dict[str, Any]) -> dict[str, Any]:
    comparison = compare_subspaces(
        np.load(path_a),
        np.load(path_b),
        similarity_alpha=float(get_nested(config, "subspace", "similarity_alpha", default=1.5)),
    )
    return {
        "similarity": comparison.similarity,
        "mean_angle_degrees": comparison.mean_angle_degrees,
        "projection_distance": comparison.projection_distance,
        "canonical_angles_degrees": comparison.canonical_angles_degrees.tolist(),
    }


def _public_query(query: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": query["id"],
        "metadata_path": query["metadata_path"],
        "sequence_path": query["sequence_path"],
        "ritual": query["ritual"],
    }


def _write_query_result(metadata_path: str, result: dict[str, Any]) -> None:
    path = Path(metadata_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["match_result"] = {
        "candidate_count": result["candidate_count"],
        "top_matches": result["top_matches"],
        "overall_match": result["overall_match"],
        "phase_matches": result["phase_matches"],
        "style_matches": result["style_matches"],
        "accepted": result["accepted"],
        "min_accept_similarity": result["min_accept_similarity"],
        "no_match_reason": result["no_match_reason"],
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _no_match_reason(
    query_confidence: float,
    has_candidates: bool,
    *,
    full_min_confidence: float,
    phase_gates_passed: bool,
) -> str:
    if not has_candidates:
        return "No registered Motion Ritual signatures were found."
    if query_confidence < full_min_confidence:
        return "Ritual quality was too low for a reliable lab twin match."
    if not phase_gates_passed:
        return "One or more ritual phases were too low quality for a reliable lab twin match."
    return "No strong Motion Ritual lab twin match was found."


def _build_query_id(timestamp: datetime) -> str:
    timestamp_part = timestamp.strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp_part}_query_motion_ritual"
