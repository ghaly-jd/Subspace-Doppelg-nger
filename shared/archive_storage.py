from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.config import get_nested


def load_archive_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    database_path = _database_path(config)
    if not database_path.exists():
        return []

    database = _read_database(database_path)
    registrations = database.get("registrations", [])
    if not isinstance(registrations, list):
        return []
    return registrations


def delete_registration(config: dict[str, Any], registration_id: str) -> dict[str, Any]:
    database_path = _database_path(config)
    if not database_path.exists():
        return {"deleted": False, "reason": "Database does not exist.", "files_deleted": []}

    database = _read_database(database_path)
    registrations = database.get("registrations", [])
    if not isinstance(registrations, list):
        return {"deleted": False, "reason": "Database has no registrations list.", "files_deleted": []}

    selected = None
    remaining = []
    for entry in registrations:
        if str(entry.get("id", "")) == registration_id:
            selected = entry
        else:
            remaining.append(entry)

    if selected is None:
        return {"deleted": False, "reason": f"Registration not found: {registration_id}", "files_deleted": []}

    paths = _referenced_paths(selected)
    metadata_path = selected.get("metadata_path")
    if metadata_path:
        metadata_file = _resolve_path(metadata_path)
        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                paths.update(_referenced_paths(metadata))
            except json.JSONDecodeError:
                paths.add(metadata_file)

    database["registrations"] = remaining
    database_path.write_text(json.dumps(database, indent=2), encoding="utf-8")

    deleted_files = []
    skipped_files = []
    allowed_roots = _allowed_roots(config, database_path)
    for path in sorted(paths):
        if path == database_path:
            continue
        if not _is_allowed_path(path, allowed_roots):
            skipped_files.append(str(path))
            continue
        if not path.exists() or not path.is_file():
            continue
        path.unlink()
        deleted_files.append(str(path))

    return {
        "deleted": True,
        "registration_id": registration_id,
        "files_deleted": deleted_files,
        "files_skipped": skipped_files,
    }


def _database_path(config: dict[str, Any]) -> Path:
    return Path(get_nested(config, "data", "database_path", default="data/database.json"))


def _read_database(database_path: Path) -> dict[str, Any]:
    data = json.loads(database_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"version": 1, "registrations": data}
    if isinstance(data, dict):
        data.setdefault("version", 1)
        data.setdefault("registrations", [])
        return data
    return {"version": 1, "registrations": []}


def _referenced_paths(value: Any) -> set[Path]:
    paths: set[Path] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and key.endswith("_path"):
                paths.add(_resolve_path(item))
            else:
                paths.update(_referenced_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.update(_referenced_paths(item))
    return paths


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _allowed_roots(config: dict[str, Any], database_path: Path) -> tuple[Path, ...]:
    roots = {Path.cwd().resolve(), database_path.resolve().parent}
    data_root = get_nested(config, "data", "root", default=None)
    if data_root:
        roots.add(_resolve_path(str(data_root)))
    registrations_dir = get_nested(config, "data", "registrations_dir", default=None)
    if registrations_dir:
        roots.add(_resolve_path(str(registrations_dir)))
    sessions_dir = get_nested(config, "data", "sessions_dir", default=None)
    if sessions_dir:
        roots.add(_resolve_path(str(sessions_dir)))
    return tuple(roots)


def _is_allowed_path(path: Path, allowed_roots: tuple[Path, ...]) -> bool:
    resolved = path.resolve()
    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False
