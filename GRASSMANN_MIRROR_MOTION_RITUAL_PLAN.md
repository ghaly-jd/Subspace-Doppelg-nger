# The Grassmann Mirror: Motion Ritual Matcher

Last updated: 2026-05-21

## Purpose

This document is the updated build plan for pivoting the current Subspace Doppelganger Terminal into **The Grassmann Mirror**, a shared motion ritual matcher.

The existing repo already has the right foundation:

- PyQt6 terminal UI.
- Webcam capture through OpenCV.
- YOLO pose tracking.
- Skeleton-only registration storage.
- JSON/NPY local database.
- Skeleton normalization, resampling, SVD subspaces, and canonical-angle matching.
- A first Wave-only matching flow with quality gating.

The new direction keeps that foundation, but changes the experience from "choose one short motion type" to "everyone performs the same guided 25-30 second ritual." The result should find an overall lab motion twin and also reveal which student matches each action, rhythm, velocity, and gesture style.

## Product Goal

Build a privacy-safe lab demo where each student performs one guided skeleton-only motion ritual once. Visitors perform the same ritual, and the app compares their movement against the lab archive.

The system returns:

- Closest overall lab motion twin.
- Closest match for each ritual action.
- Closest match by rhythm, velocity, energy, balance, and gesture style.
- Side-by-side skeleton replay for the overall match and each phase match.

The system must not use face recognition, identity recognition, or raw video storage by default.

## Ritual Sequence

One capture, roughly 25-30 seconds.

| Phase | Action | Goal |
| --- | --- | --- |
| 1 | Calibration Pose | Stand still; estimate body scale, center, visibility, and baseline stability. |
| 2 | Lateral Step Scan | Step left, step right, return center; capture side-to-side balance and lower-body coordination. |
| 3 | Arm Raise Scan | Raise both arms, then lower them; capture shoulder range and upper-body symmetry. |
| 4 | Squat Scan | Squat down and stand up; capture vertical motion, knee/hip angle change, and control. |
| 5 | Rotation Scan | Turn shoulders/body left and right; capture torso rotation style. |
| 6 | Balance Hold | Lift one knee, hold, and return; capture balance, stability, and lower-body control. |
| 7 | Wave Hello | Wave naturally; capture personal gesture style. |
| 8 | Clap Twice | Clap twice naturally; capture timing, symmetry, and hand approach. |
| 9 | Point Forward | Point forward or give one simple gesture; capture final personal gesture signature. |

## What Changes From The Current Repo

The current system stores and compares a single selected motion type such as `Wave`. The updated system should store one `Motion Ritual` registration and derive multiple comparable artifacts from the same full sequence.

Core changes:

- Replace the motion-type selector with one guided ritual mode.
- Increase capture duration from 6 seconds to about 25-30 seconds.
- Add prompt/timer guidance for each ritual phase.
- Split each full recording into named action segments.
- Build a full-ritual subspace and separate per-phase subspaces.
- Add velocity, rhythm, joint-angle, energy, and balance feature extractors.
- Rank matches overall, per phase, and per style category.
- Expand the result screen from one best match to a structured ritual match report.
- Add skeleton replay for overall match and phase-specific matches.

## Data Model

For each student registration, store:

- `nickname`
- `avatar`
- `ritual_type`: `Motion Ritual`
- `full_sequence_path`
- `normalized_full_sequence_path`
- `phase_segments`
- `full_motion_matrix_path`
- `full_subspace_basis_path`
- `full_velocity_subspace_basis_path`
- `quality`
- `features`
- `raw_video_stored`: `false`

Each phase segment should include:

```json
{
  "phase_id": "wave_hello",
  "label": "Wave Hello",
  "start_frame": 540,
  "end_frame": 660,
  "sequence_path": "data/registrations/..._wave_hello_sequence.npy",
  "normalized_sequence_path": "data/registrations/..._wave_hello_normalized.npy",
  "motion_matrix_path": "data/registrations/..._wave_hello_motion_matrix.npy",
  "coordinate_subspace_basis_path": "data/registrations/..._wave_hello_coordinate_basis.npy",
  "velocity_subspace_basis_path": "data/registrations/..._wave_hello_velocity_basis.npy",
  "features": {
    "rhythm": {},
    "joint_angles": {},
    "energy": {},
    "balance": {}
  },
  "quality": {}
}
```

For V1, phase splitting can be timer-based because the ritual is guided. Later, phase boundaries can be refined with motion events such as clap peaks, wrist activity, hip height, or lateral center-of-mass shifts.

## Feature Artifacts

The current `shared/skeleton_normalizer.py` and `shared/subspace.py` stay central. Add a new feature layer on top.

Recommended new modules:

- `shared/ritual_schema.py`
  Defines phase IDs, labels, prompt text, target durations, keypoint profiles, and scoring weights.

- `shared/ritual_segmenter.py`
  Converts a full skeleton sequence into phase slices using configured time windows and frame rate.

- `shared/motion_features.py`
  Computes velocity, rhythm, joint-angle, energy, balance, symmetry, and gesture descriptors.

- `shared/ritual_processor.py`
  Builds all full-ritual and per-phase artifacts for registrations and queries.

- `shared/ritual_match_engine.py`
  Compares a visitor ritual against the archive and returns overall, phase, and style matches.

- `ui/ritual_capture_screen.py`
  Shared guided capture UI for registration and visitor scan.

- `ui/ritual_result_screen.py`
  Result report with overall match, phase matches, style matches, and replay entry points.

- `ui/ritual_replay_screen.py`
  Side-by-side skeleton playback for full ritual and selected phase.

## Processing Pipeline

```text
webcam
-> YOLO pose tracking
-> skeleton-only frame sequence
-> full-sequence normalization
-> timer-based ritual phase segmentation
-> per-phase normalization
-> coordinate motion matrices
-> velocity motion matrices
-> SVD subspace artifacts
-> rhythm / joint-angle / energy / balance features
-> archive comparison
-> overall, phase, and style match report
-> skeleton replay
```

## Scoring

All scores should be percentages from 0 to 100. Keep the current canonical-angle similarity, but combine it with additional feature similarities.

### Overall Score

```text
overall_score =
  0.35 * full_ritual_coordinate_subspace_similarity
+ 0.20 * average_phase_similarity
+ 0.15 * full_ritual_velocity_similarity
+ 0.10 * rhythm_similarity
+ 0.10 * joint_angle_similarity
+ 0.10 * energy_balance_similarity
```

The weights should live in config so they can be tuned after real captures.

### Phase Score

```text
phase_score =
  0.45 * coordinate_subspace_similarity
+ 0.25 * velocity_subspace_similarity
+ 0.15 * rhythm_similarity
+ 0.15 * joint_angle_similarity
```

### Style Matches

Style matches should be computed independently from the overall winner:

- `Velocity`: compare velocity subspaces and peak speed features.
- `Rhythm`: compare tempo, periodicity, clap interval, pulse interval, and speed-curve shape.
- `Energy`: compare total joint travel, acceleration, and motion amplitude.
- `Balance`: compare center drift, lateral sway, squat stability, and calibration stability.
- `Gesture`: compare Wave, Clap Twice, and Point Forward phase features.

## Result Screen

The result should feel like a scan report, not a single forced winner.

Example:

```text
OVERALL MOTION TWIN:
Student B - 87.4%

PHASE MATCHES:
Lateral Step  -> Student A - 91.2%
Arm Raise     -> Student B - 89.7%
Squat         -> Student C - 84.5%
Rotation      -> Student B - 86.9%
Wave Hello    -> Student D - 92.3%
Clap Twice    -> Student A - 90.2%
Point Gesture -> Student B - 85.8%

STYLE MATCHES:
Velocity -> Student B
Rhythm   -> Student B
Energy   -> Student A
Balance  -> Student C
Gesture  -> Student D
```

If quality is weak, show a careful result:

```text
RITUAL QUALITY LOW
Closest weak match: Student B - 62.1%
Reason: Not enough lower-body keypoints were visible during Lateral Step and Squat.
```

## Skeleton Playback

After the result, support:

- Overall replay: visitor vs overall closest match.
- Phase replay: visitor vs closest match for each action.
- Style replay: optional shortcut to gesture-heavy phases like Wave, Clap, and Point.

Examples:

```text
Overall:
You vs Student B

Wave Hello:
You vs Student D

Clap Twice:
You vs Student A
```

The replay can use normalized skeleton arrays and does not need raw video.

## UI Direction

Brand:

```text
THE GRASSMANN MIRROR
Motion Ritual: Find Your Lab Twin
```

Home options:

- `REGISTER LAB RITUAL`
- `FIND YOUR LAB TWIN`
- `ARCHIVE`
- `LIVE SKELETON PREVIEW`
- `EXPLAIN THE MIRROR`

Capture prompts should be concise and sci-fi flavored:

```text
CALIBRATION PHASE 01
Stand still.

LATERAL STEP SCAN 02
Step left. Step right. Return center.

PERSONAL GESTURE SCAN 07
Wave hello.

PERSONAL GESTURE SCAN 08
Clap twice.

PERSONAL GESTURE SCAN 09
Point forward.

Capturing individual motion signature...
```

Avoid implying biometric identity recognition. The language should say motion twin, ritual match, skeleton motion, and style similarity.

## Quality Gates

Keep the current idea of quality-adjusted similarity, but expand it per ritual phase.

Track:

- Overall tracked-frame ratio.
- Per-phase tracked-frame ratio.
- Keypoint visibility by phase.
- Body scale reliability.
- Camera framing stability.
- Minimum activity for active phases.
- Specific checks for Clap Twice, Wave Hello, Squat, and Lateral Step.

Examples:

- Wave Hello needs visible shoulders, elbows, wrists, and enough wrist travel.
- Clap Twice needs two hand-approach events or at least two clear wrist convergence peaks.
- Squat needs visible hips/knees/ankles and vertical hip movement.
- Lateral Step needs visible hips/ankles and lateral center movement.
- Calibration needs low movement and stable pose visibility.

## Config Changes

Replace short capture settings:

```yaml
capture:
  countdown_seconds: 3
  registration_seconds: 6
  query_seconds: 6
```

With ritual settings:

```yaml
ritual:
  name: Motion Ritual
  target_fps: 30
  countdown_seconds: 3
  total_seconds: 28
  min_tracked_frames: 300
  phases:
    - id: calibration_pose
      label: Calibration Pose
      duration_seconds: 3
    - id: lateral_step
      label: Lateral Step Scan
      duration_seconds: 4
    - id: arm_raise
      label: Arm Raise Scan
      duration_seconds: 3
    - id: squat
      label: Squat Scan
      duration_seconds: 4
    - id: rotation
      label: Rotation Scan
      duration_seconds: 4
    - id: balance_hold
      label: Balance Hold
      prompt: Lift one knee. Hold. Return.
      duration_seconds: 4
    - id: wave_hello
      label: Wave Hello
      duration_seconds: 3
    - id: clap_twice
      label: Clap Twice
      duration_seconds: 2
    - id: point_forward
      label: Point Forward
      duration_seconds: 2
```

Keep `subspace` config, but add ritual scoring weights:

```yaml
ritual_scoring:
  overall:
    full_coordinate_subspace: 0.35
    average_phase: 0.20
    full_velocity_subspace: 0.15
    rhythm: 0.10
    joint_angles: 0.10
    energy_balance: 0.10
  phase:
    coordinate_subspace: 0.45
    velocity_subspace: 0.25
    rhythm: 0.15
    joint_angles: 0.15
```

## Migration Plan

### Phase A: Rename The Experience

Goal: Make the UI and docs point toward The Grassmann Mirror without breaking current Wave testing.

Tasks:

- Update home, register, match, result, and explanation screen copy.
- Rename `Find My Doppelganger` to `Find Your Lab Twin`.
- Add a `Motion Ritual` mode while keeping the old motion selector temporarily for fallback tests.
- Update README and STATUS references.

Exit criteria:

- App still launches.
- Existing Wave registrations can still be tested while the ritual path is being built.

### Phase B: Guided Ritual Capture

Goal: Record one 25-30 second guided ritual for registration and query.

Tasks:

- Add `shared/ritual_schema.py`.
- Add phase prompts and phase countdown UI.
- Add one shared ritual capture component for student registration and visitor scan.
- Save full sequence with metadata and no raw video.
- Add config-driven phase durations.

Exit criteria:

- A student can complete the full ritual and save one full-sequence registration.
- A visitor can complete the same ritual and save a query session.

### Phase C: Phase Segmentation

Goal: Split full recordings into named action segments.

Tasks:

- Add timer-based `shared/ritual_segmenter.py`.
- Save phase start/end frames in metadata.
- Save per-phase normalized arrays and motion matrices.
- Build per-phase coordinate subspaces.

Exit criteria:

- Every ritual registration has full and per-phase artifacts.
- A script can inspect each phase and verify frame ranges.

### Phase D: Feature Extraction

Goal: Add style information beyond coordinate subspace similarity.

Tasks:

- Add velocity sequence generation.
- Add velocity subspaces for full ritual and phases.
- Add rhythm features for speed curves, pulses, and clap timing.
- Add joint-angle features for arms, squat, and torso rotation.
- Add energy and balance features.
- Add focused unit tests with synthetic motions.

Exit criteria:

- Registration and query metadata include full and per-phase feature dictionaries.
- Synthetic tests confirm obvious rhythm, energy, and angle differences are measurable.

### Phase E: Ritual Matching Engine

Goal: Return overall, phase, and style matches.

Tasks:

- Add `shared/ritual_match_engine.py`.
- Compare full ritual subspaces.
- Compare phase subspaces.
- Compare velocity, rhythm, joint-angle, energy, and balance features.
- Add weighted scoring from config.
- Return a structured result payload with top overall, per-phase, and per-style matches.
- Apply phase-level quality gates before accepting high-confidence results.

Exit criteria:

- A visitor query ranks all registered students overall.
- Each phase has an independent closest match.
- Weak captures show honest low-quality messaging instead of forced matches.

### Phase F: Result And Replay

Goal: Make the output understandable and visually satisfying.

Tasks:

- Replace the single-match result screen with an overall/phase/style report.
- Add replay selection for overall and each phase.
- Animate normalized skeletons side by side.
- Show labels and scores without clutter.
- Preserve top-3 overall ranking for debugging and demo confidence.

Exit criteria:

- User can see "You vs overall closest match."
- User can select Wave, Clap, Point, Squat, etc. and see the correct phase match replay.

### Phase G: Polish And Demo Hardening

Goal: Make the app reliable and coherent for a public lab demo.

Tasks:

- Tune phase durations with real users.
- Tune score weights with at least 5-10 student registrations.
- Add archive browser for ritual registrations.
- Add reset/demo database utility if needed.
- Test Mac mini and lab PC camera/framing.
- Update deployment docs and operator checklist.

Exit criteria:

- Demo operator can register students and run visitor scans without touching code.
- The result screen is readable on the target TV.
- The app gracefully handles missing camera, low tracking, and empty archive cases.

## MVP Build Order

1. Keep PyQt6 terminal UI and current capture/matching foundation.
2. Add Grassmann Mirror branding and ritual copy.
3. Add guided 25-30 second ritual capture.
4. Save full ritual student registrations.
5. Split recording into timer-based phases.
6. Build full and per-phase coordinate subspaces.
7. Add velocity, rhythm, joint-angle, energy, and balance features.
8. Compare visitor rituals against the lab archive.
9. Show overall, phase, and style matches.
10. Add side-by-side skeleton playback for overall and per-phase matches.
11. Tune quality gates and scoring with real student data.
12. Polish galactic UI for the final demo.

## Immediate Next Tasks

Recommended next code steps:

1. Add `shared/ritual_schema.py` with the nine phases, prompts, durations, and IDs.
2. Update config files with a `ritual` section and temporary `capture.total_seconds`.
3. Replace Register/Match motion selector labels with `Motion Ritual` while preserving legacy Wave test path if useful.
4. Build `shared/ritual_segmenter.py` and unit tests for frame slicing.
5. Refactor registration storage so a saved ritual can include `phase_segments`.
6. Add a first result payload shape for overall and phase matches, even before advanced style features are complete.

## Definition Of Done

The Grassmann Mirror MVP is done when:

1. A lab member can register one guided ritual.
2. A visitor can perform the same guided ritual.
3. The system stores skeleton-only full and phase data.
4. The system returns an overall motion twin.
5. The system returns closest matches for each major action.
6. The system returns style matches for velocity, rhythm, energy, balance, and gesture.
7. The user can watch overall and phase side-by-side skeleton replay.
8. Low-quality scans are rejected or clearly marked as weak.
9. The UI consistently presents the experience as The Grassmann Mirror.
