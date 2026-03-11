# Peg-in-Hole Insertion — MuJoCo Simulation

A physics-rich simulation for a peg-in-hole insertion task using a Robotiq 2F-85 gripper.

## Project Structure

```
peg_in_hole/
├── README.md
├── requirements.txt
├── simulation.py               # Entry point: --mode teleop | agent
├── scene/
│   ├── scene.xml               # MuJoCo scene definition
│   ├── assets/                 # All mesh files (gripper + peg + cuboid)
│   └── robotiq_2f85/
│       ├── 2f85.xml            # Original gripper model (untouched)
│       └── 2f85_free.xml       # Modified: added freejoint for mocap control
├── control/
│   ├── teleop.py               # Keyboard teleoperation (direct)
│   └── agent.py                # Agent logic used by simulation.py
└── test/
    └── test_viewer.py          # Debug: inspect scene and launch viewer
```

## Setup

```bash
pip install -r requirements.txt
```

### Gripper Model

Clone the Robotiq 2F-85 from mujoco_menagerie into `scene/`:

```bash
cd scene
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie && git sparse-checkout set robotiq_2f85 && cd ..
cp -r mujoco_menagerie/robotiq_2f85 .
rm -rf mujoco_menagerie
```

Copy gripper mesh files and your STLs into a shared `assets/` folder:

```bash
mkdir -p assets
cp robotiq_2f85/assets/* assets/
cp peg.stl assets/
cp cuboid.stl assets/
```

Create the modified gripper file for mocap control:

```bash
cp robotiq_2f85/2f85.xml robotiq_2f85/2f85_free.xml
```

Edit `robotiq_2f85/2f85_free.xml` — find:
```xml
<body name="base_mount" pos="0 0 0.007" childclass="2f85">
```
Replace with:
```xml
<body name="base_mount" pos="0 0 0.2" childclass="2f85">
  <freejoint name="gripper_free"/>
```

### Verify Setup

```bash
mjpython tests/test_viewer.py
```

## Usage

All commands run from project root.

```bash
# Teleoperation (standalone)
mjpython control/teleop.py

# Simulation wrapper with mode flag
mjpython simulation.py --mode teleop
mjpython simulation.py --mode agent
```

### Teleop Controls

| Key | Action |
|-----|--------|
| W/S | Move Y (forward/back) |
| A/D | Move X (left/right) |
| Q/E | Move Z (up/down) |
| I/K | Pitch |
| J/L | Yaw |
| U/O | Roll |
| Space | Toggle gripper open/close |

## Design Choices

- **Gripper model**: Robotiq 2F-85 from mujoco_menagerie (pre-tuned for MuJoCo with tendon actuation and collision pads).
- **Gripper control**: Mocap body + weld constraint. The mocap body sets the target 6D pose, the weld pulls the gripper there through physics. This gives realistic contact forces during insertion.
- **Jaw control**: Input is jaw separation in meters (0.014m–0.085m), converted to actuator command (0–255). If the peg blocks the fingers, MuJoCo's contact solver applies max force without penetration.
- **Peg**: Free body with gravity. Cuboid: fixed on ground.

## Build Progress

- [x] Scene with gripper, peg, cuboid
- [x] Mocap gripper control (6D pose + jaw separation)
- [x] Teleoperation
- [x] Logging
- [x] Simulated agent (via `simulation.py --mode agent`)