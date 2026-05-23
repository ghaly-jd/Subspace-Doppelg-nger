# Mac Mini to Lab PC Deployment Guide

This guide keeps development on the Mac mini and deployment on the lab PC connected to the TV simple and repeatable.

## Deployment Principle

The GitHub repo should contain:

- Source code.
- Documentation.
- Config templates.
- Dependency files.
- Model download instructions.
- Small intentional sample data only.

The GitHub repo should not contain:

- `.venv/`
- Raw videos.
- Generated local registrations.
- Large downloaded model weights, unless intentionally approved later.
- Machine-specific secrets or absolute local paths.

## Recommended Flow

```text
Mac mini
-> develop and test
-> commit to GitHub
-> lab PC pulls latest code
-> lab PC creates its own venv
-> lab PC installs the same requirements
-> lab PC runs with configs/lab_pc.yaml
```

## Python Version

Use Python 3.10 for both machines.

This avoids many MediaPipe/OpenCV/PyQt version mismatches and keeps the lab PC setup boring in the best possible way.

Check Python:

```bash
python --version
```

or:

```bash
python3 --version
```

Expected:

```text
Python 3.10.x
```

## Mac Mini Setup

From the repo root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Or use the setup script:

```bash
bash scripts/setup_mac.sh
```

Run in Mac development mode:

```bash
python app.py --config configs/mac_mini.yaml
```

When code changes are ready:

```bash
git add .
git commit -m "Update subspace terminal"
git push
```

## Lab PC Setup: Windows PowerShell

Clone the repo:

```powershell
git clone <your-repo-url>
cd Subspace-Doppelg-nger
```

Create and activate the venv:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Or use the setup script:

```powershell
.\scripts\setup_lab_pc_windows.ps1
```

Run in lab PC TV mode:

```powershell
python app.py --config configs/lab_pc.yaml
```

Pull updates later:

```powershell
git pull
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py --config configs/lab_pc.yaml
```

## Lab PC Setup: Linux

Clone the repo:

```bash
git clone <your-repo-url>
cd Subspace-Doppelg-nger
```

Create and activate the venv:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Or use the setup script:

```bash
bash scripts/setup_lab_pc_linux.sh
```

Run in lab PC TV mode:

```bash
python app.py --config configs/lab_pc.yaml
```

Pull updates later:

```bash
git pull
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py --config configs/lab_pc.yaml
```

## Dependency Reproducibility

For V1, `requirements.txt` uses constrained version ranges instead of floating packages. This gives the lab PC a stable install path while still allowing patch-level fixes.

After the app is tested on both Mac mini and lab PC, generate a lock file from the lab PC:

```bash
python -m pip freeze > requirements-lab-pc.lock.txt
```

Then commit it:

```bash
git add requirements-lab-pc.lock.txt
git commit -m "Add lab PC dependency lock"
git push
```

Use the lock file only after it exists and has been tested:

```bash
python -m pip install -r requirements-lab-pc.lock.txt
```

Keep `requirements.txt` as the human-maintained dependency list.

## Models Directory

Use this layout:

```text
models/
|-- README.md
|-- mediapipe/
|   `-- pose_landmarker_full.task
`-- yolo/
    `-- yolo11n-pose.pt
```

V1 default:

- Place YOLO11 pose weights under `models/yolo/`.
- Use `models/yolo/yolo11n-pose.pt` first because it is the smallest and best for testing.
- Download it with:

```bash
python tools/download_models.py yolo11n-pose
```

Optional MediaPipe path:

- MediaPipe Tasks uses `.task` files under `models/mediapipe/`.
- Download the lite task with:

```bash
python tools/download_models.py pose-landmarker-lite
```

The app should read model locations from config:

```yaml
pose:
  engine: yolo
  model_path: models/yolo/yolo11n-pose.pt
```

or later:

```yaml
pose:
  engine: mediapipe
  model_path: models/mediapipe/pose_landmarker_lite.task
```

## Model Sources

Use official model/documentation sources:

- MediaPipe Pose Landmarker Python guide: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
- Ultralytics YOLO11 model docs: https://docs.ultralytics.com/models/yolo11/
- YOLO11n pose weight listed by Ultralytics: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt

Optional helper command:

```bash
python tools/download_models.py yolo11n-pose
```

Do not commit large model files until the final deployment choice is clear.

## Config Separation

Use `configs/mac_mini.yaml` for development and `configs/lab_pc.yaml` for TV deployment.

Do not hardcode:

- Camera ID.
- Screen size.
- Fullscreen mode.
- Pose engine.
- Model path.
- Data path.

Those belong in config files.

## Lab PC First-Run Checklist

1. Install Git.
2. Install Python 3.10.
3. Clone the repo.
4. Create `.venv`.
5. Install `requirements.txt`.
6. Check camera source with `tools/check_camera.py` once it exists.
7. Confirm `configs/lab_pc.yaml` camera source.
8. Run app with `configs/lab_pc.yaml`.
9. Confirm fullscreen appears on the TV.
10. Register one test motion and verify saved data appears under `data/`.

## Troubleshooting

If Python is wrong:

- Install Python 3.10 and recreate `.venv`.

If pip install fails:

- Upgrade pip, setuptools, and wheel.
- Confirm the active Python version is 3.10.
- On Windows, install Microsoft C++ Build Tools if a package needs compilation.

If camera does not open:

- Change `camera.source` in `configs/lab_pc.yaml`.
- Try source `0`, `1`, and `2`.
- Check OS camera privacy settings.

On macOS, if OpenCV says camera access was denied:

1. Open `System Settings`.
2. Go to `Privacy & Security`.
3. Open `Camera`.
4. Enable camera access for the terminal app you are using.
5. Close and reopen the terminal, then activate `.venv` again.

If the terminal app does not appear in the Camera list yet, run:

```bash
source .venv/bin/activate
python tools/check_camera.py --config configs/debug.yaml
```

Then reopen the Camera privacy settings.

If fullscreen opens on the wrong display:

- Change OS display settings so the TV is primary.
- Keep `runtime.fullscreen: false` temporarily while debugging.

If MediaPipe is unstable:

- Confirm Python 3.10.
- Recreate `.venv`.
- Reinstall from `requirements.txt`.
