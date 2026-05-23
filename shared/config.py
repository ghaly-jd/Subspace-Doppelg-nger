from __future__ import annotations

from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when the app config cannot be loaded."""


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ConfigError(
            "Missing dependency: PyYAML. Activate the project venv and run "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ConfigError(f"Config root must be a mapping: {config_path}")

    return config


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current

