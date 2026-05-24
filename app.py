from __future__ import annotations

import argparse
from pathlib import Path

from shared.config import ConfigError, get_nested, load_config


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The Grassmann Mirror")
    parser.add_argument(
        "--config",
        default="configs/debug.yaml",
        help="Path to runtime config YAML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and print runtime summary without opening the UI.",
    )
    return parser


def print_runtime_summary(config_path: Path, config: dict) -> None:
    print("The Grassmann Mirror")
    print(f"Config: {config_path}")
    print(f"Pose engine: {get_nested(config, 'pose', 'engine', default='unknown')}")
    print(f"Runtime device: {get_nested(config, 'runtime', 'device', default='unknown')}")
    print(f"Fullscreen: {get_nested(config, 'runtime', 'fullscreen', default=False)}")
    print(f"Display mode: {get_nested(config, 'runtime', 'display_mode', default='legacy')}")
    print(f"Camera source: {get_nested(config, 'camera', 'source', default=0)}")


def run_ui(config_path: Path, config: dict) -> int:
    try:
        from PyQt6.QtWidgets import QApplication
        from ui.main_window import MainWindow
        from ui.terminal_theme import apply_theme
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: PyQt6. Activate the project venv and run "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    app = QApplication([])
    apply_theme(app)

    window = MainWindow(config, str(config_path))
    window.show_for_runtime()

    return app.exec()


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc

    if args.dry_run:
        print_runtime_summary(config_path, config)
        return 0

    return run_ui(config_path, config)


if __name__ == "__main__":
    raise SystemExit(main())
