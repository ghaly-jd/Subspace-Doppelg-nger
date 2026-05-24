# Subspace Doppelganger Terminal

Interactive motion retrieval demo using skeleton keypoints, PCA/SVD motion subspaces, and canonical-angle similarity.

The system is designed as a privacy-safe lab terminal: lab members voluntarily register short motion signatures, visitors perform a short movement, and the app finds the registered skeleton-motion subspace that is most similar. It does not need face recognition or stored raw video.

## Current State

This repository currently contains the planning documents, PyQt app shell, YOLO live skeleton preview, registration capture/save, subspace processing, and the first matching flow.

The newest product direction is **The Grassmann Mirror: Motion Ritual Matcher**. Instead of selecting one short motion such as Wave, everyone performs the same guided 25-30 second ritual. The app compares the full ritual, each action phase, and style features such as velocity, energy, balance, and gesture.

Start here:

- `GRASSMANN_MIRROR_MOTION_RITUAL_PLAN.md` - updated pivot plan for the shared motion ritual matcher.
- `subspace_doppelganger_terminal_plan_updated.md` - full concept and technical specification.
- `PROCESS_PLAN.md` - phased development roadmap.
- `DEPLOYMENT.md` - Mac mini to lab PC setup, GitHub handoff, venv, and model placement guide.
- `STATUS.md` - living project status and decision log.

## Updated MVP Scope

The Grassmann Mirror MVP should support:

- Register nickname and avatar for one guided `Motion Ritual`.
- Capture one 25-30 second webcam skeleton ritual.
- Split the ritual into calibration, lateral step, arm raise, squat, rotation, balance hold, wave, clap, and point phases.
- Save full and per-phase normalized skeleton sequences and metadata.
- Wave and Clap use wrist trajectory/timing from body pose, so finger tracking is not required for the MVP.
- Build full-ritual and per-phase motion subspaces with PCA/SVD.
- Compare visitors and registered students with canonical angles plus velocity, rhythm, joint-angle, energy, and balance features.
- Return the closest overall lab motion twin, closest phase matches, and style matches.
- Replay visitor and matched skeletons side by side for the overall ritual and selected phases.
- Explain the match as skeleton-motion similarity, not identity recognition.

V1 should not include:

- Face recognition.
- Raw video storage by default.
- Unity or Unreal.
- Cloud backend.
- Accounts or authentication.
- Multi-camera setup.
- Docker-first deployment.

## Recommended Stack

- Python 3.10
- PyQt6 for desktop UI
- OpenCV for camera access
- YOLO11 pose for V1 pose tracking
- NumPy and SciPy for math
- scikit-learn if useful for PCA utilities
- JSON and NPY files for V1 storage

## Planned Repository Structure

```text
subspace_doppelganger_terminal/
|-- app.py
|-- README.md
|-- requirements.txt
|-- environment.yml
|-- configs/
|   |-- mac_mini.yaml
|   |-- lab_pc.yaml
|   `-- debug.yaml
|-- assets/
|   |-- ui/
|   |-- avatars/
|   `-- sounds/
|-- data/
|   |-- registrations/
|   |-- sessions/
|   `-- database.json
|-- shared/
|   |-- camera_manager.py
|   |-- pose_engine.py
|   |-- skeleton_normalizer.py
|   |-- skeleton_renderer.py
|   |-- subspace.py
|   |-- dtw_tools.py
|   |-- motion_features.py
|   `-- database.py
|-- ui/
|   |-- main_window.py
|   |-- terminal_theme.py
|   |-- home_screen.py
|   |-- register_screen.py
|   |-- match_screen.py
|   |-- result_screen.py
|   `-- explanation_screen.py
|-- modules/
|   `-- doppelganger/
|       |-- register_flow.py
|       |-- match_flow.py
|       |-- replay_view.py
|       `-- result_logic.py
`-- tools/
    `-- check_camera.py
```

## Motion Pipeline

```text
webcam frame
-> pose detection
-> keypoint extraction
-> keypoint normalization
-> motion sequence matrix
-> PCA/SVD subspace
-> canonical-angle comparison
-> similarity score
-> skeleton replay and explanation
```

Recommended V1 keypoints:

```text
left_shoulder, right_shoulder
left_elbow, right_elbow
left_wrist, right_wrist
left_hip, right_hip
left_knee, right_knee
left_ankle, right_ankle
```

Each frame has 12 keypoints with 2D coordinates, giving 24 features per frame. A captured sequence should be normalized and resampled to 120 frames.

## Setup

Use a local venv on each machine. Do not commit `.venv/` to GitHub.

Mac mini:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

or:

```bash
bash scripts/setup_mac.sh
```

Lab PC Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

or:

```powershell
.\scripts\setup_lab_pc_windows.ps1
```

Lab Mac mini M1 setup from GitHub:

```bash
git clone https://github.com/ghaly-jd/Subspace-Doppelg-nger Subspace-Doppelganger
cd Subspace-Doppelganger
```

Install Python 3.10 if it is not already installed:

```bash
brew install python@3.10
```

Create the local venv and install dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

or:

```bash
bash scripts/setup_mac.sh
```

Download the YOLO11 pose model on the lab Mac:

```bash
python tools/download_models.py yolo11n-pose
```

Run the app:

```bash
python app.py --config configs/mac_mini.yaml
```

For the TV demo mode, run:

```bash
python app.py --config configs/lab_pc.yaml
```

`configs/lab_pc.yaml` starts maximized by default so the app respects the active display's usable area and keeps the normal window controls available. Press `F11` to toggle true fullscreen after the window is on the right display, and press `Esc` to leave fullscreen.

On first launch, macOS may ask for camera permission. If the camera does not open, go to:

```text
System Settings -> Privacy & Security -> Camera
```

Then enable camera access for Terminal, iTerm, or whichever terminal app launched Python.

For detailed transfer steps, see `DEPLOYMENT.md`.

## Running

Mac mini development mode:

```bash
python app.py --config configs/mac_mini.yaml
```

From the home screen, choose:

```text
[5] LIVE SKELETON PREVIEW
```

Then press:

```text
[ START LIVE SKELETON PREVIEW ]
```

If skeleton detection fails, the screen should still show raw camera frames and report the issue in the status line.

To register a motion signature, choose:

```text
[1] REGISTER MOTION SIGNATURE
```

Enter a nickname, select an avatar and motion type, then start capture. The app saves skeleton-only keypoint sequences under `data/registrations/` and updates `data/database.json`; it does not store raw video.

Registration saves also build the derived V1 math artifacts:

- normalized 120-frame skeleton sequence
- `120 x 24` motion matrix
- `24 x 5` subspace basis
- subspace mean vector

To rebuild those artifacts for existing registrations:

```bash
python tools/build_subspaces.py --config configs/mac_mini.yaml
```

To find a match, choose:

```text
[2] FIND MY DOPPELGANGER
```

Select the same motion type as a saved registration, start the scan, and perform the motion. The query is saved under `data/sessions/`, compared against same-motion registrations, and the result screen shows similarity, mean canonical angle, projection distance, canonical angles, and the top matches.

Wave matching uses an upper-body profile so seated visitors or partially framed users can still be compared by shoulder, elbow, and wrist motion. Full-body motions still use the full skeleton profile.

Wave scans also use a quality gate. If the upper body is too small/unstable, too few arm keypoints are visible, or the closest score is still weak, the app reports the closest weak match instead of accepting it as a doppelganger.

To test pose detection outside the GUI:

```bash
python tools/check_pose.py --config configs/mac_mini.yaml
```

The tool prints MediaPipe landmark counts and saves a debug image to `data/sessions/pose_debug_frame.jpg` when a pose is detected.

The current V1 config uses YOLO11n-pose because MediaPipe triggered a Mac OpenGL service error on the Mac mini. The model path is `models/yolo/yolo11n-pose.pt`.

Run tests:

```bash
python -m unittest discover
```

Debug mode:

```bash
python app.py --config configs/debug.yaml
```

Lab PC TV mode:

```bash
python app.py --config configs/lab_pc.yaml
```

## Camera Test

Before running the full app on a new machine, test camera access:

```bash
python tools/check_camera.py --config configs/debug.yaml
```

Common things to check:

- Camera source index.
- Webcam permissions.
- Resolution support.
- PyQt display scaling.
- Fullscreen behavior on the lab TV.

On macOS, the first camera check may fail until the terminal app has Camera permission in `System Settings > Privacy & Security > Camera`.

## Data and Privacy

The app should store only skeleton-derived data by default:

```text
nickname
avatar/icon
motion type
normalized skeleton sequence
subspace basis
motion features
timestamp
```

Generated recordings and local registration data should not be committed unless they are intentional sample data.

## Development Workflow

Recommended order:

1. Build UI shell.
2. Add webcam feed.
3. Add pose overlay.
4. Save one motion recording.
5. Replay one skeleton.
6. Register multiple students.
7. Build subspace comparison.
8. Show best match.
9. Add side-by-side playback.
10. Add explanation.
11. Polish the terminal UI.

## Deployment Strategy

Develop on the Mac mini first, push to GitHub, then clone on the lab PC connected to the TV.

For V1:

- Use Conda or venv.
- Use MediaPipe on CPU.
- Keep hardware-specific settings in config files.
- Avoid Docker until webcam, GUI, fullscreen, and deployment behavior are stable.

Model files belong under `models/`, with paths controlled by config files. V1 MediaPipe can run without a manually downloaded model if using `mp.solutions.pose`; later YOLO weights should go under `models/yolo/`.

## Project Pitch

Subspace Doppelganger Terminal is a galactic-style interactive demo where a visitor's skeleton motion is transformed into a subspace and compared with voluntary lab-member motion signatures using canonical angles to find the closest motion doppelganger.
