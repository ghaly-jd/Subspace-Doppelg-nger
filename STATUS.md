# Project Status

Last updated: 2026-05-23

## Overall Status

Status: Repository foundation and basic UI are running on the Mac mini. Phase 2 live skeleton preview works with YOLO pose, Phase 3 registration capture/save is verified, Phase 4 normalization/subspace processing is implemented, and Phase 5 matching now rejects weak/unreliable Wave scans instead of always forcing a doppelganger.

Updated direction: the next product target is **The Grassmann Mirror: Motion Ritual Matcher**. The app should pivot from selected single-motion captures to one shared guided 25-30 second ritual, then compare the full ritual, each action phase, and style features such as velocity, rhythm, energy, balance, and gesture.

The repository currently contains:

- Updated ritual pivot plan: `GRASSMANN_MIRROR_MOTION_RITUAL_PLAN.md`
- Full project specification: `subspace_doppelganger_terminal_plan_updated.md`
- Build roadmap: `PROCESS_PLAN.md`
- Deployment guide: `DEPLOYMENT.md`
- Project README: `README.md`
- Status tracker: `STATUS.md`

## Current Phase

Current phase: Phase 5 - Matching Flow, with a planned pivot into the Grassmann Mirror ritual architecture.

Next target: run a real camera-based Motion Ritual registration and visitor scan from the user's terminal, then tune phase durations, quality thresholds, and scoring weights using actual student captures.

## Phase Checklist

| Phase | Name | Status |
| --- | --- | --- |
| 0 | Repository Foundation | Complete on Mac mini |
| 1 | Basic Terminal UI | Complete on Mac mini |
| 2 | Camera and Pose Tracking | Complete on Mac mini |
| 3 | Registration Flow | Complete on Mac mini |
| 4 | Normalization and Subspace Engine | Complete for V1 |
| 5 | Matching Flow | In progress |
| 6 | Skeleton Replay | Not started |
| 7 | Explanation Layer | Not started |
| 8 | Terminal Polish | Not started |
| 9 | Deployment and Demo Hardening | Not started |

## Immediate Next Tasks

- Create Python package/folder structure. Done.
- Add `requirements.txt`. Done.
- Add optional `environment.yml`. Done.
- Add `configs/mac_mini.yaml`, `configs/lab_pc.yaml`, and `configs/debug.yaml`. Done.
- Add `.gitignore`. Done.
- Add `DEPLOYMENT.md`. Done.
- Add `models/README.md`. Done.
- Add optional model download helper. Done.
- Add `tools/check_camera.py`. Done.
- Add minimal `app.py` with `--config` parsing. Done.
- Add Mac and lab PC setup scripts. Done.
- Add PyQt main window and placeholder screen navigation. Done.
- Add camera manager, MediaPipe pose engine, and skeleton renderer. Done.
- Add live skeleton preview screen. Done.
- Verify live skeleton overlay from interactive terminal. Done.
- Add registration countdown and fixed-duration capture. Done.
- Save raw skeleton keypoint sequence and metadata without raw video. Done.
- Update registration database index. Done.
- Verify one real registration capture from interactive terminal. Done.
- Add skeleton normalization and resampling to 120 frames. Done.
- Add motion matrix and SVD subspace builder. Done.
- Add canonical-angle comparison and similarity scoring. Done.
- Add build tool to process saved registrations into normalized/subspace artifacts. Done.
- Add numerical tests for normalization and subspace comparison. Done.
- Add visitor query capture flow. Done.
- Save query skeleton/session artifacts without raw video. Done.
- Load same-motion registered candidates and rank by canonical-angle similarity. Done.
- Show best match and top-3 ranking on the result screen. Done.
- Add upper-body Wave matching profile for seated/partial-body camera framing. Done.
- Verify one real match scan from interactive terminal. Pending.

## Decisions

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-05-20 | Use MediaPipe Pose for V1. | Easiest webcam-based pose engine for a laptop prototype. |
| 2026-05-20 | Use PyQt6 for V1 UI. | Suitable for fullscreen desktop kiosk-style app. |
| 2026-05-20 | Use JSON and NPY storage for V1. | Simple, inspectable, and enough for prototype data. |
| 2026-05-20 | Avoid Docker for V1. | Webcam, GUI, fullscreen, audio, and OS differences are easier outside Docker. |
| 2026-05-20 | Do not store raw video by default. | Keeps the demo privacy-safe and focused on skeleton motion. |
| 2026-05-20 | Use a separate `.venv` on each machine. | Avoids Mac/PC binary and path mismatch while keeping the GitHub repo clean. |
| 2026-05-20 | Store local model files under `models/`. | Keeps model paths stable across Mac mini and lab PC configs. |
| 2026-05-20 | Keep Python version at 3.10. | MediaPipe and desktop CV packages are more predictable on 3.10 than the system Python 3.13 currently present on the Mac mini. |
| 2026-05-20 | Pin V1 dependency versions after first successful Mac install. | Keeps Mac mini and lab PC installs closer instead of allowing silent dependency drift. |
| 2026-05-20 | Use YOLO11n-pose as the V1 pose backend. | MediaPipe hit a Mac OpenGL service error; YOLO CPU avoids that path and is easier to mirror on the lab PC. |

## Open Questions

- Which OS will the lab PC use: Windows, Linux, or macOS?
- Which camera will be used for the TV demo?
- Should public demo mode show real registered nicknames or anonymous labels?
- Should sample registration data be committed for testing?
- Is DTW needed in V1 or should it wait until the subspace score works well?
- What motion types should be enabled first: wave, squat, walk, or free motion?

## Known Risks

- Camera index may differ between Mac mini and lab PC.
- MediaPipe installation can be sensitive to Python and OS versions.
- PyQt fullscreen behavior may differ on the TV setup.
- Skeleton jitter may reduce matching quality until smoothing is tuned.
- Matching quality depends on having enough registered examples per motion type.

## Latest Work Log

### 2026-05-23

- Added event-aware Motion Ritual segmentation that refines timer phase boundaries using pose events such as squat hip drop, balance knee lift, wave wrist activity, clap wrist-distance peaks, arm-raise wrist height, and lateral body travel.
- Added per-phase segmentation metadata, including event center frame and segmentation method, to ritual artifacts.
- Added speed-curve DTW similarity to per-phase scoring so faster/slower versions of the same movement can still match.
- Added per-phase quality/activity gates for critical ritual phases, configurable under `ritual_quality`, so weak wave, clap, squat, or balance phases can reject an otherwise acceptable overall score.
- Added camera framing guidance during the pre-capture countdown for registration and matching, with visible-keypoint count and step closer/back/center messages.
- Added test coverage for event segmentation metadata, DTW tempo tolerance, and required phase-gate rejection.
- Motion Ritual registration and match completion now opens the skeleton replay automatically and starts playback.
- Replay now supports a 2-skeleton or 3-skeleton comparison mode, so visitors can compare themselves with the first and second ranked matches.
- Replay details now show score, coordinate similarity, velocity similarity, DTW/phase similarity, canonical angle summaries, and phase gate failures.
- Added CVLAB branding with the lab logo, glowing CVLAB badge, and a subtle floating-logo space background across app screens.
- Motion Archive now includes a scrollable gallery of animated skeleton preview cards for all registered signatures, with card/table selection kept in sync for deletion.

### 2026-05-21

- Added the first Grassmann Mirror implementation slice: ritual schema, timer-based phase segmentation, motion feature extraction, ritual artifact processing, ritual matching, and ritual result payloads.
- Registration now defaults to `Motion Ritual` and stores full-ritual plus per-phase coordinate/velocity subspaces, rhythm, joint-angle, energy, balance, gesture features, and phase quality metadata.
- Match flow now routes `Motion Ritual` scans through `shared/ritual_match_engine.py`, returning overall, phase, and style matches while preserving the legacy Wave matcher as a fallback path.
- Updated the PyQt home, register, match, result, and replay screens with Grassmann Mirror copy, guided phase prompts during ritual capture, and side-by-side skeleton playback for overall and phase matches.
- Improved ritual replay quality by separating math-normalized 120-frame artifacts from display-normalized playback artifacts, preserving sequence-level body travel, preferring raw skeleton paths for old captures, and adding replay speed/reset controls.
- Registration now immediately routes completed Motion Ritual captures to the comparison result screen, excluding the newly saved registration from candidate ranking so the user sees closest existing lab matches instead of a self-match.
- Expanded ritual result details with coordinate/velocity subspace similarities, mean canonical angles, projection distances, canonical angle lists, and per-phase coordinate/velocity/angle diagnostics.
- Replaced the experimental Peace Sign phase with Balance Hold so the ritual stays reliable with YOLO body pose. Wave and Clap remain wrist-based body-pose gestures; optional hand-landmark storage remains available internally but is not part of the MVP ritual.
- Implemented the Motion Archive browser with registration rows, refresh, and confirmed delete. Delete removes the database entry and referenced registration artifact files.
- Added large high-contrast on-screen ritual prompts above the camera preview in registration and match flows so phase instructions are readable while standing back from the webcam.
- Added staged processing prompts for registration and matching: skeleton sequence, motion matrix, PCA/SVD, Grassmann subspace search, canonical angles, and score fusion.
- Improved matching accuracy by quality-weighting phase scores, so low-confidence or poorly tracked phases contribute less to the overall ritual score and display their quality weight in result details.
- Added ritual config sections and scoring weights to Mac, debug, and lab PC configs.
- Added `tests/test_ritual_flow.py` covering registration, query matching, candidate loading, and phase segmentation.
- User confirmed a real registration saved with 128 captured frames.
- Added `shared/skeleton_normalizer.py` for hip-centered, torso-scaled, smoothed, resampled motion matrices.
- Added `shared/subspace.py` for SVD motion subspaces, canonical angles, projection distance, and similarity scores.
- Added `tools/build_subspaces.py` and processed the real registration into a 120-frame, 24-feature motion matrix and 24 x 5 subspace basis.
- Updated future registration saves to write normalized sequence, motion matrix, subspace basis, and subspace mean artifacts automatically.
- Added Phase 4 unittest coverage in `tests/test_phase4_math.py`.
- Added `shared/match_engine.py` for visitor query artifact generation, same-motion candidate loading, canonical-angle ranking, and top-k result payloads.
- Replaced the Match screen placeholder with a real countdown, fixed-duration query scan, skeleton preview, match computation, and retry handling.
- Updated the Result screen to show best match, similarity percentage, mean canonical angle, projection distance, canonical angles, and top matches.
- Added `tests/test_match_engine.py` and verified the synthetic end-to-end match path.
- Verified the saved Wave registration ranks as a 100% self-match in a non-camera smoke test.
- User reported same-wave seated scans scoring around 9% because lower-body/full-body features dominated when only arms were visible.
- Added motion-specific profiles: Wave now compares shoulder-centered, shoulder-scaled shoulders/elbows/wrists instead of the full body.
- Added per-motion config for Wave with `dimension_k=3` and `similarity_alpha=0.35` to make partial-body wave scores less harsh.
- Existing saved Wave query sessions now score around 61-66% against the available Wave registrations instead of single digits.
- Added test coverage to confirm Wave matching ignores missing lower-body keypoints when upper-body motion is present.
- User reported a non-wave scan still showed around 63% similarity.
- Added query quality gating for Wave based on upper-body visibility, shoulder scale reliability, and wrist activity.
- Added Wave acceptance threshold of 70%; lower scores are reported as weak closest matches instead of accepted doppelgangers.
- The Match screen now stops on weak/unreliable scans with a no-clear-match message and includes adjusted/raw similarity for debugging.

### 2026-05-20

- Read and understood the full project plan.
- Created `PROCESS_PLAN.md` with phases, tasks, deliverables, and exit criteria.
- Created `README.md` with project overview, setup plan, pipeline, and workflow.
- Created `STATUS.md` for current state, decisions, open questions, and next tasks.
- Added deployment handoff files for Mac mini to lab PC GitHub workflow.
- Added venv dependency files, config templates, model directory notes, and ignore rules.
- Added minimal config-loading `app.py` and `tools/check_camera.py` camera probe.
- Added `tools/download_models.py` for optional YOLO11n pose model download.
- Added `scripts/setup_mac.sh`, `scripts/setup_lab_pc_windows.ps1`, and `scripts/setup_lab_pc_linux.sh`.
- Confirmed current Mac has `python3` 3.13.5, but no `python3.10` yet.
- Added Phase 1 PyQt shell modules under `ui/` with home, register, match, archive, result, and explanation screens.
- Added shared config loader under `shared/config.py`.
- Verified Python syntax compilation with the available Python 3.13 interpreter.
- Installed Python 3.10.20 with Homebrew.
- Created project `.venv` and installed V1 dependencies successfully.
- Verified dry-run config loading with `.venv/bin/python app.py --config configs/mac_mini.yaml --dry-run`.
- Verified imports for PyQt6, OpenCV, MediaPipe, NumPy, SciPy, scikit-learn, and PyYAML.
- Camera probe reached OpenCV but macOS denied camera permission for the terminal app.
- User confirmed camera and GUI worked from their terminal.
- Added `shared/camera_manager.py`, `shared/pose_engine.py`, `shared/pose_schema.py`, and `shared/skeleton_renderer.py`.
- Added `ui/live_pose_screen.py` and wired it into Home, Register, and Match screens.
- Verified Phase 2 modules compile and import inside `.venv`.
- Tool-runner camera probe still lacks macOS Camera permission, so final live overlay verification should be done from the user's terminal.
- Hardened live preview so raw camera frames render even if MediaPipe pose initialization or detection fails.
- Added frame counters and in-GUI error messages for camera, pose, and render failures.
- Set Mac/debug camera backend to OpenCV AVFoundation.
- User confirmed raw camera preview works but skeleton overlay did not appear.
- Lowered MediaPipe detection/tracking confidence and skeleton visibility threshold.
- Added full MediaPipe landmark debug overlay and clearer no-pose diagnostics.
- Added high-contrast forced debug drawing for selected skeleton keypoints.
- Added `tools/check_pose.py` to test MediaPipe detection outside the GUI and save a debug overlay frame.
- Found MediaPipe classic and Tasks backends both fail on the Mac mini with `Could not create an NSOpenGLPixelFormat`.
- Installed `ultralytics==8.3.233`, downloaded `models/yolo/yolo11n-pose.pt`, and switched Mac/debug/lab configs to YOLO pose.
- Verified YOLO pose engine initializes successfully in `.venv`.
- User confirmed YOLO skeleton overlay works.
- Added configurable live-preview keypoint smoothing to reduce jitter.
- Added Phase 3 registration capture UI with nickname, avatar, motion type, countdown, fixed-duration recording, and retry handling.
- Added skeleton-only registration storage as `.npy` keypoint sequences plus JSON metadata and `data/database.json` indexing.
- Added capture timing controls to Mac, lab PC, and debug configs.
- Verified syntax compilation, dry-run config loading, and registration storage with a synthetic smoke test.
