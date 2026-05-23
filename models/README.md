# Models

This directory is the stable place for local pose model files.

## V1 Default: MediaPipe

V1 should use MediaPipe Pose through the `mediapipe` Python package.

If the implementation uses the classic `mp.solutions.pose` API, no manual model file is required.

If the implementation uses the newer MediaPipe Tasks Pose Landmarker API, place the downloaded `.task` model here:

```text
models/mediapipe/pose_landmarker_full.task
```

Official guide:

```text
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
```

## Later Option: YOLO11 Pose

If YOLO-pose is added later, put model weights here:

```text
models/yolo/yolo11n-pose.pt
```

Start with `yolo11n-pose.pt` because it is the smallest YOLO11 pose model and is appropriate for testing.

Official docs:

```text
https://docs.ultralytics.com/models/yolo11/
```

Official YOLO11n pose weight listed by Ultralytics:

```text
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt
```

Optional helper command from the repo root:

```bash
python tools/download_models.py yolo11n-pose
```

## Git Policy

Do not commit large downloaded model files by default.

Commit this README and the directory structure. Download model weights on each machine as needed, or add a controlled release asset later if the final demo needs offline setup.
