from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.config import get_nested, load_config
from shared.skeleton_normalizer import normalize_keypoint_sequence, normalize_to_motion_matrix
from shared.subspace import build_motion_subspace, compare_subspaces


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build normalized skeleton sequences and motion subspaces for registrations."
    )
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument(
        "--metadata",
        help="Path to a single registration metadata JSON file. Defaults to all database entries.",
    )
    return parser


def main() -> int:
    args = create_parser().parse_args()
    config = load_config(args.config)

    metadata_paths = _metadata_paths(config, args.metadata)
    if not metadata_paths:
        print("No registration metadata files found.")
        return 0

    for metadata_path in metadata_paths:
        metadata = _process_registration(config, metadata_path)
        print(
            f"Built subspace for {metadata['id']}: "
            f"{metadata['normalization']['output_frames']} frames, "
            f"{metadata['subspace']['basis_shape']} basis, "
            f"{metadata['subspace']['self_similarity']:.1f}% self-similarity"
        )

    _refresh_database(config, metadata_paths)
    return 0


def _metadata_paths(config: dict[str, Any], metadata_arg: str | None) -> list[Path]:
    if metadata_arg:
        return [Path(metadata_arg)]

    database_path = Path(get_nested(config, "data", "database_path", default="data/database.json"))
    if not database_path.exists():
        return []

    database = json.loads(database_path.read_text(encoding="utf-8"))
    registrations = database.get("registrations", []) if isinstance(database, dict) else database
    paths = []
    for entry in registrations:
        metadata_path = entry.get("metadata_path")
        if metadata_path:
            paths.append(Path(metadata_path))
    return paths


def _process_registration(config: dict[str, Any], metadata_path: Path) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sequence = np.load(metadata["sequence_path"])

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
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _refresh_database(config: dict[str, Any], metadata_paths: list[Path]) -> None:
    database_path = Path(get_nested(config, "data", "database_path", default="data/database.json"))
    if not database_path.exists():
        return

    database = json.loads(database_path.read_text(encoding="utf-8"))
    if not isinstance(database, dict):
        return

    metadata_by_id = {
        json.loads(path.read_text(encoding="utf-8"))["id"]: json.loads(path.read_text(encoding="utf-8"))
        for path in metadata_paths
        if path.exists()
    }
    for entry in database.get("registrations", []):
        metadata = metadata_by_id.get(entry.get("id"))
        if not metadata:
            continue
        for key in (
            "normalized_sequence_path",
            "motion_matrix_path",
            "subspace_basis_path",
            "subspace_mean_path",
            "normalization",
            "subspace",
        ):
            if key in metadata:
                entry[key] = metadata[key]

    database_path.write_text(json.dumps(database, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
