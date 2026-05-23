from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: PyYAML. Activate the project venv and run "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> int:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: opencv-python. Activate the project venv and run "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    parser = argparse.ArgumentParser(description="Check webcam access.")
    parser.add_argument(
        "--config",
        default="configs/debug.yaml",
        help="Path to runtime config YAML file.",
    )
    parser.add_argument(
        "--source",
        type=int,
        default=None,
        help="Override camera source index from config.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    camera_config = config.get("camera", {})
    source = args.source if args.source is not None else camera_config.get("source", 0)
    backend_name = str(camera_config.get("backend", "auto")).lower()
    width = int(camera_config.get("width", 640))
    height = int(camera_config.get("height", 480))
    fps = int(camera_config.get("fps", 30))

    backend = None
    if backend_name == "avfoundation":
        backend = cv2.CAP_AVFOUNDATION
    elif backend_name == "dshow":
        backend = cv2.CAP_DSHOW
    elif backend_name == "msmf":
        backend = cv2.CAP_MSMF

    if backend is None:
        capture = cv2.VideoCapture(source)
    else:
        capture = cv2.VideoCapture(source, backend)

    if not capture.isOpened():
        print(f"Camera check failed: could not open source {source} with backend {backend_name}")
        return 1

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)

    ok, frame = capture.read()
    capture.release()

    if not ok or frame is None:
        print(f"Camera check failed: source {source} opened but no frame was read")
        return 1

    actual_height, actual_width = frame.shape[:2]
    print(f"Camera check passed: source {source} with backend {backend_name}")
    print(f"Requested: {width}x{height} at {fps} fps")
    print(f"Frame read: {actual_width}x{actual_height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
