# Peg-in-Hole Insertion — MuJoCo Simulation

A physics-rich simulation for a peg-in-hole insertion task using a Robotiq 2F-85 gripper.

## Project Structure

```
peg_in_hole/
├── README.md
├── requirements.txt
├── simulation.py                  # Entry point: --mode teleop | agent
├── scene/
│   ├── scene.xml                  # MuJoCo scene definition
│   ├── peg.stl                    # Rectangular peg mesh (30x30x100 mm)
│   ├── cuboid.stl                 # Cuboid with 40x40 mm hole mesh
│   └── robotiq_2f85/
│       ├── 2f85.xml               # Original gripper model (untouched)
│       └── 2f85_free.xml          # Modified: freejoint for mocap control
├── control/
│   ├── agent.py                   # Shared utilities + make_agent() factory
│   ├── agent_simple.py            # SimpleAgent: minimal state machine
│   ├── agent_advanced.py          # AdvancedAgent: tip tracking, tilt correction, spiral search
│   ├── logger.py                  # SimLogger: structured logging per assignment spec
│   └── teleop.py                  # Standalone keyboard teleoperation
├── logs/
│   └── run.json                   # Output log from most recent run
└── test/
    └── test_viewer.py             # Debug: inspect scene and launch viewer
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

Copy gripper mesh files and your STLs into the scene folder:

```bash
cp robotiq_2f85/assets/* .
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
mjpython test/test_viewer.py
```

## Usage

All commands run from project root (`peg_in_hole/`).

```bash
# Agent mode (simple agent — default)
mjpython simulation.py --mode agent

# Agent mode (advanced agent)
mjpython simulation.py --mode agent --agent advanced

# Teleoperation
mjpython simulation.py --mode teleop

# Adjustable speed (2x faster)
mjpython simulation.py --mode agent --speed 2.0

# Downsample logging
mjpython simulation.py --mode agent --log-every 50
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `agent` | `agent` or `teleop` |
| `--agent` | `simple` | `simple` or `advanced` (agent mode only) |
| `--speed` | `1.0` | Time scale (>1 faster, <1 slower) |
| `--log` | `logs/run.json` | Output log file path |
| `--log-every` | `1` | Log every N-th physics step |
| `--steps` | `30000` | Max simulation steps (agent mode only) |

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

## Agent Variants

### SimpleAgent (`--agent simple`)

Minimal 10-state machine: approach, grip, lift, swing to vertical, regrip, fly above hole, descend straight down, release. Uses peg COM for alignment. No tilt correction, no spiral search, no contact-aware descent. Relies on the 5 mm hole clearance being sufficient for a straight insertion.

### AdvancedAgent (`--agent advanced`)

11-state machine with the same pickup sequence plus a sophisticated two-phase insertion:
- **Phase A (hover)**: Aligns peg bottom *tip* (not COM) with hole centre using P-gain control; corrects peg tilt via quaternion adjustment on the mocap body.
- **Phase B (descent)**: Descends while continuously tracking tip XY, applying spiral search to find the hole opening, and pausing Z-descent when peg-cuboid contact forces exceed a threshold.

## Design Choices

- **Physics engine**: MuJoCo — fast, stable contact solver with native support for tendons, welds, and mocap bodies.
- **Gripper model**: Robotiq 2F-85 from mujoco_menagerie, pre-tuned for MuJoCo with tendon actuation and silicone pad collision geometry.
- **Gripper control**: A mocap body sets the target 6D pose; a weld equality constraint pulls the physical gripper toward it through the physics solver.
- **Jaw control**: Input is jaw separation in metres (0.014 m – 0.085 m), converted to actuator command (0–255). If the peg blocks the fingers, MuJoCo's contact solver applies maximum force without penetration.
- **Cuboid collision geometry**: The cuboid STL mesh is used for rendering only (`contype="0" conaffinity="0"`). Five box primitives define the actual collision shape of the hole (4 walls + 1 floor), because MuJoCo computes the convex hull of mesh geoms which would fill in the hole.
- **Peg**: Free body with gravity; starts flat on the table.
- **Logging**: `SimLogger` records at each physics step: commanded gripper pose & jaw, actual gripper pose & jaw (measured from pad body positions), peg pose, and contact forces categorised into gripper-peg, peg-table, peg-cuboid, and cuboid-table pairs. Output is compact JSON for manageable file sizes.

## Log Format

Each entry in `logs/run.json`:

```json
{
  "step": 0,
  "time": 0.002,
  "commanded": {"gripper_pos": [...], "gripper_quat": [...], "jaw_separation": 0.085},
  "actual":    {"gripper_pos": [...], "gripper_quat": [...], "jaw_separation": 0.085},
  "peg":       {"pos": [...], "quat": [...]},
  "contacts":  {"gripper_peg": [...], "peg_table": [...], "peg_cuboid": [...], "cuboid_table": [...]}
}
```
