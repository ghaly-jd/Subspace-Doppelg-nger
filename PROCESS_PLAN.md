# Subspace Doppelganger Terminal Process Plan

This document turns `subspace_doppelganger_terminal_plan_updated.md` into an implementation roadmap. The original file is the product and research specification; this file is the build process.

Update note: the newest product direction is now captured in `GRASSMANN_MIRROR_MOTION_RITUAL_PLAN.md`. That plan keeps the current PyQt, pose tracking, skeleton storage, normalization, and subspace foundation, but pivots the experience from selected short motions to one shared guided ritual with full-ritual, per-phase, and style matching.

## Project Goal

Build a privacy-safe, galactic terminal-style demo where lab members voluntarily register short skeleton-motion signatures and visitors find the registered motion signature whose PCA subspace is closest by canonical-angle similarity.

The V1 result should support:

- Registering a nickname, avatar, and motion type.
- Capturing 5-8 seconds of skeleton keypoints from a webcam.
- Normalizing and resampling skeleton sequences.
- Building a low-dimensional PCA/SVD motion subspace.
- Comparing visitor and registered subspaces with canonical angles.
- Returning the closest match with similarity score and explanation.
- Replaying both skeleton motions side by side.

## Guiding Constraints

- Do not use face recognition.
- Do not store raw video by default.
- Keep V1 in Python with PyQt6, OpenCV, MediaPipe Pose, NumPy, SciPy, scikit-learn, and JSON/NPY storage.
- Keep hardware-specific details in config files.
- Build the working prototype first, then polish the sci-fi terminal presentation.
- Use Conda or venv for V1. Add Docker only after the desktop app works.

## Phase 0: Repository Foundation

Goal: Make the repository ready for steady development.

Tasks:

- Create the planned folder structure.
- Add `requirements.txt` and optionally `environment.yml`.
- Add `configs/mac_mini.yaml`, `configs/lab_pc.yaml`, and `configs/debug.yaml`.
- Add empty data folders with `.gitkeep` files where needed.
- Add a small camera-check utility under `tools/check_camera.py`.
- Add ignore rules for generated data, caches, videos, and local environment files.

Deliverables:

- Project skeleton exists.
- Dependencies are documented.
- Config strategy is in place.
- A developer can run a simple camera check.

Exit criteria:

- `python tools/check_camera.py --config configs/debug.yaml` can test at least one camera source.
- The repo has clear setup and run instructions in `README.md`.

## Phase 1: Basic Terminal UI

Goal: Open the app and navigate between screens.

Tasks:

- Create `app.py` entry point with `--config`.
- Create a PyQt6 main window.
- Create terminal-style theme primitives.
- Add home screen navigation.
- Add placeholder screens for register, match, archive, replay, result, and explanation.
- Add fake loading text for the matching flow.

Deliverables:

- Running app shell.
- Home menu with all major modes.
- Screen navigation without camera or math yet.

Exit criteria:

- `python app.py --config configs/debug.yaml` opens the app.
- All placeholder screens can be reached and returned from.

## Phase 2: Camera and Pose Tracking

Goal: Show a webcam feed with live skeleton overlay.

Tasks:

- Implement camera capture with OpenCV. Done.
- Implement MediaPipe Pose wrapper. Done.
- Extract the 12 recommended body keypoints. Done.
- Draw skeleton overlay on the camera frame. Done.
- Display frames inside PyQt6 without freezing the UI. Done.
- Handle missing keypoints gracefully. Done.

Deliverables:

- Live camera preview.
- Live skeleton overlay.
- Pose engine module with a stable keypoint format.

Exit criteria:

- The app can show the user's skeleton in real time.
- Lost tracking does not crash the app.

## Phase 3: Registration Flow

Goal: Register one lab member motion signature and save it.

Tasks:

- Build nickname input, avatar selector, and motion type selector. Done.
- Add countdown and fixed-duration recording. Done.
- Collect keypoint frames during recording. Done.
- Save raw skeleton keypoint sequence as generated data, not raw video. Done.
- Save registration metadata as JSON. Done.
- Add user-visible success and retry states. Done.

Deliverables:

- Registration screen.
- Saved metadata and sequence file.
- Database index updated with the new registration.

Exit criteria:

- A lab member can register a motion signature from the app.
- The saved data can be loaded again by a script or app module.

## Phase 4: Normalization and Subspace Engine

Goal: Convert skeleton sequences into comparable subspaces.

Tasks:

- Center each frame on hip center. Done.
- Scale by torso length. Done.
- Smooth jitter with a simple moving average. Done.
- Resample sequences to 120 frames. Done.
- Flatten sequences to a `T x D` matrix. Done.
- Center the matrix and compute SVD. Done.
- Save the top `k=5` principal directions. Done.
- Add numerical tests for normalization and subspace comparison. Done.

Deliverables:

- `shared/skeleton_normalizer.py`
- `shared/subspace.py`
- Unit tests for key math paths.

Exit criteria:

- Given a saved registration sequence, the app can save a matching subspace basis.
- Canonical angles between identical or near-identical sequences are small.

## Phase 5: Matching Flow

Goal: Record a visitor query and return the closest registered match.

Tasks:

- Reuse capture flow for visitor motion. Done.
- Build query subspace. Done.
- Load registered entries with the same motion type. Done.
- Compute canonical angles and projection distance. Done.
- Convert distance to similarity score. Done.
- Rank candidates and return top 3. Done.
- Show best match on result screen. Done.

Deliverables:

- `modules/doppelganger/match_flow.py`
- `modules/doppelganger/result_logic.py`
- Result screen with match name, similarity, and mean canonical angle.

Exit criteria:

- With at least two registrations, the app can rank matches.
- Empty database and wrong motion-type cases show useful messages.

## Phase 6: Skeleton Replay

Goal: Compare visitor and match visually.

Tasks:

- Load visitor and registered normalized sequences.
- Resample both to the same length.
- Draw skeletons on blank canvases or replay panels.
- Animate side-by-side playback.
- Add labels, trails, and simple score overlays.

Deliverables:

- `modules/doppelganger/replay_view.py`
- Side-by-side replay screen.

Exit criteria:

- The result screen can open a synchronized replay.
- Replay works without the original camera feed or raw video.

## Phase 7: Explanation Layer

Goal: Explain why the match happened in simple research language.

Tasks:

- Compute arm swing amplitude.
- Compute torso stability.
- Compute hip drop or squat depth.
- Compute rhythm and motion energy features.
- Compare visitor and matched features.
- Generate three shared motion traits.
- Show canonical angles and a simple subspace explanation.

Deliverables:

- `shared/motion_features.py`
- Explanation screen.
- Human-readable trait generation.

Exit criteria:

- Each result can show canonical angles and at least three generated traits when data quality allows it.
- Explanation avoids claiming identity recognition.

## Phase 8: Terminal Polish

Goal: Make the demo feel like a sci-fi research terminal after the core system works.

Tasks:

- Add starfield or grid background.
- Add scanner lines and loading animations.
- Add avatars.
- Add subtle sound effects.
- Add fullscreen kiosk option.
- Add result-card export if time allows.
- Tune UI for Mac mini and lab PC TV resolution.

Deliverables:

- Polished theme.
- Fullscreen lab PC mode.
- Optional export feature.

Exit criteria:

- The demo is usable from several feet away on a TV.
- The app can run with `configs/lab_pc.yaml` in fullscreen.

## Phase 9: Deployment and Demo Hardening

Goal: Make the project reliable for open-campus or lab demo use.

Tasks:

- Test Mac mini development flow.
- Test lab PC clone and setup flow.
- Verify camera source, resolution, and fullscreen behavior.
- Add troubleshooting notes for webcam permissions and MediaPipe install issues.
- Add demo reset procedure.
- Add backup sample registrations.
- Decide whether Docker is worth adding after V1 is stable.

Deliverables:

- Lab PC setup checklist.
- Known issues and fixes in `STATUS.md`.
- Demo-ready release tag.

Exit criteria:

- A fresh lab PC setup can run the app from documented instructions.
- Demo operator can recover from common camera/config/database issues.

## Suggested Milestones

1. App shell and navigation.
2. Live skeleton preview.
3. First saved registration.
4. First computed subspace.
5. First correct match from stored registrations.
6. Side-by-side replay.
7. Explanation screen.
8. Fullscreen polished demo.
9. Lab PC deployment test.

## Risks and Mitigations

- Camera permissions differ between machines: keep a camera check script and config files.
- MediaPipe installation may vary by OS/Python version: pin dependencies once tested.
- Skeleton jitter can damage matching quality: smooth, resample, and handle low-confidence frames.
- Height and camera distance can dominate features: normalize by hip center and torso scale.
- Small registration database may produce weak matches: show top 3 and explain confidence carefully.
- PyQt UI can freeze during capture: use timers or worker threads for camera updates.
- Raw generated data may grow: ignore generated recordings and keep sample data small.

## Definition of Done for V1

V1 is done when a visitor can:

1. Open the app.
2. Register or use existing lab-member skeleton signatures.
3. Select a motion type.
4. Perform a short motion in front of the camera.
5. Receive a closest motion doppelganger result.
6. Watch side-by-side skeleton replay.
7. Read an explanation based on canonical angles and shared motion traits.
