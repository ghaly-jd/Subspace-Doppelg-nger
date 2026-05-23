from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


MODELS = {
    "pose-landmarker-lite": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "path": Path("models/mediapipe/pose_landmarker_lite.task"),
    },
    "yolo11n-pose": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt",
        "path": Path("models/yolo/yolo11n-pose.pt"),
    }
}


def download(url: str, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        print(f"Already exists: {destination}")
        print("Use --overwrite to download it again.")
        return

    print(f"Downloading: {url}")
    print(f"Destination: {destination}")
    urllib.request.urlretrieve(url, destination)
    print("Download complete.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download optional local model files.")
    parser.add_argument(
        "model",
        choices=sorted(MODELS),
        help="Model key to download.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing local model file.",
    )
    args = parser.parse_args()

    model = MODELS[args.model]
    download(model["url"], model["path"], args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
