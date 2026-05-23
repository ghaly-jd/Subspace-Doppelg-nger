# Subspace Doppelgänger Terminal

## Interactive Motion Retrieval Using Canonical-Angle Similarity

**Project concept:**  
A galactic terminal-style computer vision demo where lab members voluntarily register short motion signatures using only skeleton keypoints. A visitor performs a short movement, and the system finds the registered lab member whose motion style is most similar.

The system does **not** need face recognition, identity recognition, or stored video. It compares motion styles using subspace representation and canonical angles.

---

## 1. Core Idea

A visitor performs a short movement in front of a camera.

The system:

1. Tracks the body skeleton.
2. Converts the skeleton sequence into a motion matrix.
3. Compresses the motion into a low-dimensional subspace.
4. Compares the visitor's subspace with registered lab-member subspaces.
5. Finds the closest match.
6. Shows side-by-side skeleton playback.
7. Explains the match using canonical angles and shared motion traits.

Example output:

```text
YOUR MOTION DOPPELGÄNGER IS:

Student B

Similarity: 88.4%
Mean Canonical Angle: 12.7°

Shared traits:
- similar arm swing rhythm
- similar torso stability
- similar squat depth
```

---

## 2. Why This Fits Fukui Lab

The project connects directly to subspace-based recognition ideas:

- Motion sequence representation
- PCA/subspace modeling
- Mutual subspace-style comparison
- Canonical angles
- Motion retrieval
- Human action analysis
- Explainable visual feedback

Instead of simply saying:

> "The AI detected your pose."

The system says:

> "Your movement was represented as a subspace and compared with other motion subspaces using canonical angles."

This makes the lab's core mathematical ideas interactive, visual, and understandable for visitors.

---

## 3. Privacy-Safe Design

This project should **not** use faces.

It should **not** store raw videos by default.

It only stores:

```text
nickname
avatar/icon
motion type
normalized skeleton sequence
subspace basis
motion features
timestamp
```

Recommended public/open-campus mode:

```text
Your motion doppelgänger is: Lab Ghost #07
Similarity: 84%
```

Recommended internal lab mode:

```text
Your motion doppelgänger is: Student B
Similarity: 88%
```

Only students who voluntarily register should appear in the database.

---

## 4. User Modes

### Mode A: Register Motion Signature

A lab member selects:

```text
[ Register Motion Signature ]
```

They type:

```text
Nickname: Student B
Avatar: Blue Star
Motion Type: Wave / Squat / Walk / Free Motion
```

Then the system records a short motion:

```text
Stand in frame.
Recording begins in 3...
2...
1...
```

The student performs a 5-8 second movement.

The app stores the skeleton sequence and subspace representation.

---

### Mode B: Find My Doppelgänger

A visitor selects:

```text
[ Find My Motion Doppelgänger ]
```

They choose:

```text
Wave
Squat
Walk
Free Motion
```

Then they perform the movement.

The system compares their motion subspace with stored registered subspaces and returns the closest match.

Example:

```text
MATCH FOUND

Your motion doppelgänger:
Student B

Subspace Similarity:
88.4%

Mean Canonical Angle:
12.7°

Shared traits:
- similar arm swing amplitude
- similar torso stability
- similar movement rhythm
```

---

### Mode C: Skeleton Replay

After matching, the system shows:

```text
YOU                          STUDENT B
skeleton replay              skeleton replay
```

Both skeletons replay side by side.

Optional overlays:

- wrist trails
- ankle trails
- hip trajectory
- center of mass
- motion speed curve
- key similarity moments

---

### Mode D: Explain the Match

The explanation screen shows:

```text
WHAT HAPPENED?

1. Your body movement was recorded as skeleton points over time.
2. The skeleton sequence became a motion matrix.
3. PCA compressed your motion into a subspace.
4. The system compared your subspace with stored lab-member subspaces.
5. Student B had the smallest canonical-angle distance.
```

It also shows:

```text
Canonical angles:
θ1 = 5.4°
θ2 = 9.8°
θ3 = 13.1°
θ4 = 18.2°
θ5 = 21.7°
```

Simple interpretation:

```text
Small canonical angles mean both motions live in similar subspaces.
That is why Student B is your motion doppelgänger.
```

---

## 5. Galactic Terminal UI Concept

The app should feel like a sci-fi research terminal.

### Visual Style

Use:

- dark background
- glowing grid
- star particles
- terminal text
- neon outlines
- scanner lines
- animated loading bars
- futuristic typography
- subtle sound effects

### Main Menu Example

```text
╔══════════════════════════════════════════════╗
║        FUKUI LAB SUBSPACE TERMINAL          ║
║        Motion Doppelgänger Finder           ║
╠══════════════════════════════════════════════╣
║  [1] Register Motion Signature              ║
║  [2] Find My Doppelgänger                    ║
║  [3] View Motion Archive                     ║
║  [4] Explain Subspace Matching               ║
╚══════════════════════════════════════════════╝
```

### Home Screen Tiles

```text
REGISTER
Capture a new lab motion signature.

MATCH
Find whose motion subspace is closest to yours.

ARCHIVE
Browse anonymous motion fossils.

EXPLAIN
Learn canonical angles visually.
```

### Matching Animation Text

```text
Scanning motion...
Building skeleton matrix...
Projecting into subspace...
Searching Grassmann archive...
Computing canonical angles...
Match found.
```

---

## 6. Technical Pipeline

Full pipeline:

```text
webcam frame
→ pose detection
→ keypoint extraction
→ keypoint normalization
→ motion sequence matrix
→ PCA/SVD subspace
→ canonical-angle comparison
→ similarity score
→ skeleton replay + explanation
```

Detailed structure:

```text
Camera
  ↓
Pose Engine
  ↓
Keypoint Sequence
  ↓
Normalization
  ↓
Feature Matrix X ∈ R^(T × D)
  ↓
Subspace U ∈ R^(D × k)
  ↓
Compare U_visitor with U_registered
  ↓
Canonical angles θ1, θ2, ..., θk
  ↓
Similarity score
  ↓
Doppelgänger result
```

---

## 7. Pose Data

Use **MediaPipe Pose** for V1 because it is easy to run on a laptop.

Later, YOLO-pose can be added.

### Recommended V1 Keypoints

Use 12 body keypoints:

```text
left_shoulder
right_shoulder
left_elbow
right_elbow
left_wrist
right_wrist
left_hip
right_hip
left_knee
right_knee
left_ankle
right_ankle
```

Each keypoint has 2D coordinates:

```text
12 keypoints × 2 coordinates = 24D per frame
```

A 5-second recording at 30 FPS gives:

```text
150 frames × 24 features
```

For easier comparison, resample every sequence to:

```text
T = 120 frames
```

Final motion matrix:

```text
X ∈ R^(120 × 24)
```

---

## 8. Skeleton Normalization

Normalization is critical so that height, camera distance, and body position do not dominate the matching.

For each frame:

### Step 1: Center

Use hip center:

```text
hip_center = (left_hip + right_hip) / 2
```

Subtract hip center from all keypoints.

### Step 2: Scale

Use torso length:

```text
shoulder_center = (left_shoulder + right_shoulder) / 2
scale = distance(shoulder_center, hip_center)
```

Divide all coordinates by this scale.

### Step 3: Smooth

Apply a simple moving average to reduce keypoint jitter.

### Step 4: Resample

Interpolate the sequence to a fixed length:

```text
T = 120 frames
```

---

## 9. Building the Motion Subspace

Given a normalized sequence:

```text
X = T × D motion matrix
```

Example:

```text
X = 120 × 24
```

Center the sequence:

```text
X_centered = X - mean(X)
```

Run SVD:

```text
X_centered = A S Vᵀ
```

The motion subspace is the first `k` principal directions:

```text
U = V_k
```

Recommended V1 setting:

```text
k = 5
```

So each motion becomes a 5-dimensional subspace.

---

## 10. Comparing Motion Subspaces

Given two subspaces:

```text
U = visitor subspace
V = registered student subspace
```

Compute:

```text
M = Uᵀ V
```

Take singular values:

```text
σ = svd(M)
```

Canonical angles:

```text
θᵢ = arccos(σᵢ)
```

Small angles mean the subspaces are similar.

### Distance Option

Projection distance:

```text
distance = sqrt(k - sum(σᵢ²))
```

### Similarity Score

Example:

```text
similarity = 100 × exp(-alpha × distance)
```

Display:

```text
Similarity: 88.4%
Mean canonical angle: 12.7°
```

---

## 11. Optional DTW Score

Subspace matching measures overall motion style.

DTW measures temporal alignment.

Use both:

```text
Subspace style similarity: 88%
DTW timing similarity: 74%
Final doppelgänger score: 84%
```

Recommended final score:

```text
final_score = 0.75 × subspace_score + 0.25 × dtw_score
```

For the Fukui Lab identity, the subspace score should remain the main score.

---

## 12. Shared Traits Explanation

The system should generate human-readable explanations using motion features.

### Useful Features

#### Arm Swing

```text
mean wrist displacement
wrist trajectory variance
left/right symmetry
```

#### Squat Depth / Body Drop

```text
hip vertical displacement
knee angle change
ankle stability
```

#### Timing / Rhythm

```text
velocity curve
peak timing
motion duration
```

#### Torso Stability

```text
shoulder center jitter
hip center jitter
torso angle variance
```

### Explanation Logic

Examples:

```text
If wrist amplitude difference is small:
"similar arm swing amplitude"

If hip drop difference is small:
"similar squat depth"

If velocity curve DTW distance is small:
"similar movement timing"

If torso variance is close:
"similar torso stability"
```

Example output:

```text
Shared traits:
- similar arm swing amplitude
- similar motion rhythm
- similar torso stability
```

---

## 13. Recommended Tech Stack

Keep V1 pure computer vision.

No Unity.  
No Unreal.  
No 3D animation engine.

Use:

```text
Python
PyQt6
OpenCV
MediaPipe Pose
NumPy
SciPy
scikit-learn
JSON or SQLite
```

### V1 Recommendation

```text
UI: PyQt6
Camera: OpenCV
Pose: MediaPipe
Math: NumPy/SciPy
Storage: JSON + NPY files
```

### Later Upgrades

```text
Pose: YOLO11-pose
Storage: SQLite
UI: Electron/React if needed
Deployment: lab PC fullscreen kiosk mode
```

---

## 14. Repository Structure

```text
subspace_doppelganger_terminal/
│
├── app.py
├── README.md
├── requirements.txt
├── environment.yml
│
├── configs/
│   ├── laptop.yaml
│   └── lab_pc.yaml
│
├── assets/
│   ├── ui/
│   ├── avatars/
│   └── sounds/
│
├── data/
│   ├── registrations/
│   ├── sessions/
│   └── database.json
│
├── shared/
│   ├── camera_manager.py
│   ├── pose_engine.py
│   ├── skeleton_normalizer.py
│   ├── skeleton_renderer.py
│   ├── subspace.py
│   ├── dtw_tools.py
│   ├── motion_features.py
│   └── database.py
│
├── ui/
│   ├── main_window.py
│   ├── terminal_theme.py
│   ├── home_screen.py
│   ├── register_screen.py
│   ├── match_screen.py
│   ├── result_screen.py
│   └── explanation_screen.py
│
└── modules/
    └── doppelganger/
        ├── register_flow.py
        ├── match_flow.py
        ├── replay_view.py
        └── result_logic.py
```

---

## 15. Data Format

Each registration should have a JSON metadata file and separate `.npy` arrays.

Example metadata:

```json
{
  "person_id": "student_b_20260520_001",
  "nickname": "Student B",
  "avatar": "blue_star",
  "motion_type": "wave",
  "created_at": "2026-05-20T15:30:00",
  "fps": 30,
  "num_frames": 120,
  "keypoints_used": [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
  ],
  "normalized_sequence_path": "data/registrations/student_b_wave_sequence.npy",
  "subspace_path": "data/registrations/student_b_wave_subspace.npy",
  "features": {
    "arm_swing": 0.82,
    "torso_stability": 0.74,
    "motion_energy": 0.91,
    "rhythm_regularity": 0.68
  }
}
```

---

## 16. Main Screens

### Home Screen

```text
FUKUI LAB SUBSPACE TERMINAL

[ Register Motion Signature ]
[ Find My Doppelgänger ]
[ Motion Archive ]
[ What is a Subspace? ]
```

### Registration Screen

```text
Enter nickname:
[______________]

Choose avatar:
[Star] [Ghost] [Crane] [Dragon]

Choose motion:
[Wave] [Squat] [Walk] [Free Motion]

[Start Capture]
```

### Capture Screen

```text
Initializing scanner...
Stand inside the frame.

Recording in:
3
2
1
```

Live skeleton overlay appears.

### Matching Screen

```text
Scanning motion...
Building subspace...
Searching Grassmann archive...
Computing canonical angles...
```

### Result Screen

```text
YOUR MOTION DOPPELGÄNGER

Student B
Similarity: 88.4%
Mean canonical angle: 12.7°

[Play Skeleton Comparison]
[Explain Match]
[Try Again]
```

### Replay Screen

```text
YOU                          STUDENT B
skeleton replay              skeleton replay

Subspace Style: 88%
Timing: 74%
Motion Energy Difference: small
```

### Explanation Screen

```text
Your skeleton sequence was converted into a motion subspace.
The closest registered subspace belonged to Student B.
The canonical angles between both subspaces were small.
```

---

## 17. MVP Development Plan

### Phase 1: Basic Terminal UI

Goal:

```text
Open app → show galactic dashboard → navigate screens
```

Tasks:

```text
Create PyQt6 fullscreen window
Create home screen
Create terminal-style theme
Create fake loading animation
Create screen navigation
```

---

### Phase 2: Camera + Skeleton Tracking

Goal:

```text
Show webcam feed with live skeleton overlay
```

Tasks:

```text
Open webcam with OpenCV
Run MediaPipe Pose
Extract selected keypoints
Draw skeleton on frame
Display inside PyQt6
```

---

### Phase 3: Registration

Goal:

```text
Student types nickname → records motion → app saves skeleton data
```

Tasks:

```text
Create nickname form
Create motion type selector
Record 5 seconds of keypoints
Normalize sequence
Save sequence as .npy
Save metadata as JSON
```

---

### Phase 4: Subspace Engine

Goal:

```text
Convert motion sequence into subspace
```

Tasks:

```text
Load sequence
Resample to fixed length
Center + scale skeletons
Flatten to T × D matrix
Run PCA/SVD
Save top-k subspace basis
```

---

### Phase 5: Matching

Goal:

```text
Visitor records motion → app finds closest registered student
```

Tasks:

```text
Record query sequence
Build query subspace
Load database entries with same motion type
Compute canonical-angle distance
Rank all candidates
Return top 3
```

---

### Phase 6: Skeleton Replay

Goal:

```text
Display visitor skeleton and matched skeleton side by side
```

Tasks:

```text
Load both normalized sequences
Resample both to same length
Draw skeletons on blank canvas
Animate frames
Add trails
Add labels
```

---

### Phase 7: Explanation Layer

Goal:

```text
Explain why the match happened
```

Tasks:

```text
Compute arm swing amplitude
Compute torso stability
Compute hip/knee depth
Compute rhythm similarity
Generate 3 shared traits
Show canonical angles
Show simple subspace diagram
```

---

### Phase 8: Polish

Goal:

```text
Make it feel like a sci-fi lab terminal
```

Tasks:

```text
Add sounds
Add loading scanline
Add starfield background
Add terminal text animation
Add avatars
Add result card export
Add fullscreen kiosk mode
```

---

## 18. Recommended Development Order

Build in this order:

```text
1. Build UI shell
2. Add webcam feed
3. Add pose overlay
4. Save one motion recording
5. Replay one skeleton
6. Register multiple students
7. Build subspace comparison
8. Show best match
9. Add side-by-side playback
10. Add explanation
11. Make it beautiful
```

Do not spend too long polishing the UI at the beginning.  
First build the ugly working prototype.  
Then turn it into the galactic terminal.

---

## 19. Configuration Strategy

Use config files so the project can move from laptop to lab PC easily.

Example:

```yaml
camera:
  source: 0
  width: 1280
  height: 720
  fps: 30

runtime:
  device: cpu
  fullscreen: false

pose:
  engine: mediapipe
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5

subspace:
  num_frames: 120
  dimension_k: 5
  similarity_alpha: 1.5
```

For lab PC:

```yaml
camera:
  source: 0
  width: 1920
  height: 1080
  fps: 30

runtime:
  device: cuda
  fullscreen: true
```

---

## 20. Pitch to Fukui-sensei

Suggested explanation:

> I want to build an interactive visualization system for subspace-based motion retrieval. Lab members voluntarily register short motion signatures using only skeleton keypoints. A visitor performs a motion, and the system converts it into a subspace, compares it with registered motion subspaces using canonical angles, and visualizes the closest match through side-by-side skeleton replay. The goal is to make mutual subspace-style recognition intuitive and engaging for lab visitors.

More casual version:

> I want to create a lab demo where visitors can experience subspace recognition with their own body movement. The system does not use face recognition. It only uses skeleton motion, converts it into a subspace, and compares it with registered lab motion signatures.

---

## 21. Possible Project Titles

Best title:

```text
Subspace Doppelgänger Terminal
```

Academic-style title:

```text
Subspace Doppelgänger: Interactive Motion Retrieval Using Canonical-Angle Similarity
```

Other options:

```text
Motion Doppelgänger: A Subspace Retrieval Terminal
Fukui Lab Subspace Arcade
Grassmann Motion Finder
Subspace Twin Finder
Canonical Doppelgänger
Motion Fossil Archive
The Grassmann Archive
```

---

## 22. V1 Scope

V1 should support:

```text
Register nickname
Choose avatar
Choose motion type
Record motion
Save skeleton sequence
Build subspace
Record query motion
Find closest match
Show similarity score
Replay skeletons side by side
Explain canonical angles simply
```

Do **not** add these in V1:

```text
face recognition
raw video storage
Unity
Unreal
3D avatars
cloud backend
complicated accounts
multi-camera setup
```

---

## 23. Future Upgrades

After V1 works:

### Upgrade 1: Motion Archive

Show all registered anonymous motion signatures as stars in a galaxy.

### Upgrade 2: Grassmann Galaxy Visualization

Each motion subspace becomes a point/star. Similar motions appear close together.

### Upgrade 3: Result Card Export

Generate shareable result cards:

```text
GHALI
Motion Doppelgänger: Student B
Subspace Similarity: 88%
Motion Type: Explosive Researcher
```

### Upgrade 4: Multi-Motion Signature

Register multiple motions per person:

```text
wave
walk
squat
free motion
```

Then compute an overall doppelgänger score.

### Upgrade 5: YOLO-Pose Engine

Switch from MediaPipe Pose to YOLO11-pose for stronger performance on the lab PC.

### Upgrade 6: Quantum Subspace Mode

Compare:

```text
Classical PCA subspace
Quantum/VQD PCA subspace
```

This connects to Ghali's own quantum-computer-vision research.

---

## 24. Why This Project Is Strong

This project has three layers:

### Fun Layer

```text
Who in the lab moves like you?
```

### Visual Layer

```text
Galactic terminal UI
Side-by-side skeleton playback
Motion archive
```

### Research Layer

```text
motion sequence → subspace → canonical angles → retrieval
```

That combination makes it more than a pose app.

It becomes a research visualization system for subspace-based motion recognition.

---

## 25. One-Sentence Summary

**Subspace Doppelgänger Terminal is a galactic-style interactive demo where a visitor's skeleton motion is transformed into a subspace and compared with voluntary lab-member motion signatures using canonical angles to find their closest motion doppelgänger.**

---

## 26. Development and Deployment Workflow: Mac mini → Lab PC TV Setup

The project can be developed on a **Mac mini** first, then pushed to GitHub and cloned onto the **lab PC connected to the TV**.

Recommended workflow:

```text
Mac mini = development machine
Lab PC + TV = demo/deployment machine
GitHub = transfer and version control
Conda/venv = environment management
Docker = optional later
```

For V1, do **not** start with Docker unless dependency issues become painful. A normal Python environment is easier for webcam access, PyQt windows, fullscreen display, and debugging.

---

### 26.1 Recommended Strategy

Use this setup:

```text
Mac mini development:
Conda or venv

Lab PC deployment:
Conda or venv first

Docker:
Only after the app works and needs reproducible deployment
```

This is better because the app needs:

```text
webcam access
fullscreen GUI
PyQt display
TV output
possibly sound effects
possibly GPU access later
```

These are usually easier outside Docker.

---

### 26.2 Why Not Docker First?

Docker is useful, but for this project it can become annoying because the system needs direct access to:

```text
camera device
display server / GUI
fullscreen window
audio
GPU, if YOLO-pose is used later
```

Also, if development happens on a Mac mini and deployment happens on a lab PC, the machines may have different architectures:

```text
Mac mini:
macOS, possibly Apple Silicon

Lab PC:
Windows or Linux, possibly NVIDIA GPU
```

Because of this, a Docker image built on Mac may not behave exactly the same on the lab PC.

For V1, use Conda/venv first.

---

### 26.3 Local Development on Mac mini

Create the environment:

```bash
conda create -n subspace-terminal python=3.10
conda activate subspace-terminal
pip install -r requirements.txt
```

Run the app with the Mac config:

```bash
python app.py --config configs/mac_mini.yaml
```

The Mac mini config should use CPU and non-fullscreen mode at first:

```yaml
camera:
  source: 0
  width: 1280
  height: 720
  fps: 30

runtime:
  device: cpu
  fullscreen: false

ui:
  screen_width: 1280
  screen_height: 720

pose:
  engine: mediapipe
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5

subspace:
  num_frames: 120
  dimension_k: 5
  similarity_alpha: 1.5
```

Development goal on Mac mini:

```text
Build and test:
- UI shell
- webcam feed
- MediaPipe skeleton overlay
- registration flow
- skeleton recording
- subspace matching
- side-by-side replay
```

---

### 26.4 Deployment on Lab PC Connected to TV

On the lab PC:

```bash
git clone <your-repo-url>
cd subspace_doppelganger_terminal
conda env create -f environment.yml
conda activate subspace-terminal
python app.py --config configs/lab_pc.yaml
```

The lab PC config should use fullscreen mode:

```yaml
camera:
  source: 0
  width: 1920
  height: 1080
  fps: 30

runtime:
  device: cpu
  fullscreen: true

ui:
  screen_width: 1920
  screen_height: 1080

pose:
  engine: mediapipe
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5

subspace:
  num_frames: 120
  dimension_k: 5
  similarity_alpha: 1.5
```

If the lab PC has an NVIDIA GPU and YOLO-pose is added later:

```yaml
runtime:
  device: cuda

pose:
  engine: yolo
  model_path: models/yolo11n-pose.pt
```

For V1, MediaPipe CPU mode is enough.

---

### 26.5 Keep Hardware-Specific Settings in Config Files

Do **not** hardcode:

```text
camera ID
screen resolution
fullscreen mode
model paths
device type
data paths
TV display settings
```

Keep them in config files:

```text
configs/
├── mac_mini.yaml
├── lab_pc.yaml
└── debug.yaml
```

Run with:

```bash
python app.py --config configs/mac_mini.yaml
```

or:

```bash
python app.py --config configs/lab_pc.yaml
```

This makes the same code work across machines.

---

### 26.6 GitHub Transfer Workflow

On Mac mini:

```bash
git init
git add .
git commit -m "Initial Subspace Doppelganger Terminal"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

On lab PC:

```bash
git clone <your-repo-url>
cd subspace_doppelganger_terminal
conda env create -f environment.yml
conda activate subspace-terminal
python app.py --config configs/lab_pc.yaml
```

Normal update workflow:

On Mac mini:

```bash
git add .
git commit -m "Update matching and replay system"
git push
```

On lab PC:

```bash
git pull
python app.py --config configs/lab_pc.yaml
```

---

### 26.7 Expected Portability Issues

The app should transfer cleanly, but these issues may appear:

```text
camera source index differs
screen resolution differs
fullscreen behavior differs
MediaPipe installation differs
PyQt scaling differs
file paths differ
webcam permissions differ
```

All of these should be handled by:

```text
config files
clear README setup steps
environment.yml
requirements.txt
camera test script
debug mode
```

Recommended helper script:

```text
tools/check_camera.py
```

It should test available camera IDs before running the full app.

---

### 26.8 When to Use Docker Later

Use Docker only after the app already works if:

```text
lab PC dependencies keep breaking
many students need to run the same system
you want a stable demo environment
you switch to YOLO-pose + CUDA
you deploy permanently on a Linux machine
```

Docker is most useful for a final deployment version, not the first prototype.

Recommended long-term setup:

```text
V1:
Conda/venv only

V2:
Conda/venv + clean environment.yml

V3:
Optional Docker for lab PC deployment

V4:
Optional startup script for kiosk mode
```

---

### 26.9 Final Recommendation

For this project:

```text
Start coding on the Mac mini.
Use Conda or venv.
Use MediaPipe first.
Push to GitHub.
Clone on the lab PC.
Use a separate lab_pc.yaml config.
Run fullscreen on the TV.
Avoid Docker until the app already works.
```

This keeps development fast and avoids unnecessary webcam/GUI/Docker problems.

