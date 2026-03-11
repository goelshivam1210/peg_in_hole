"""
agent.py — Simple pick-and-place agent for peg insertion scene.

Used by `simulation.py` when running:

    mjpython simulation.py --mode agent

State machine:
    1. move_above_peg   — move to safe height above peg
    2. lower_to_peg     — descend to grip height
    3. grip             — close fingers until force threshold or timeout
    4. lift             — raise to safe travel height
    5. move_above_hole  — translate over cuboid hole (XY only)
    6. done             — hold position

Actuator note:
    ctrl=0   → fully open  (~85 mm jaw separation)
    ctrl=255 → fully closed (0 mm jaw separation)
    Conversion: ctrl = 255 * (1 - jaw_sep / 0.085)

Coordinate frame: Z-up.
    Peg centre:    (0.15, 0,    0.015)  lying flat, long axis along X
    Cuboid centre: (0.35, 0,    0.05)
    base_mount:    (0,    0,    0.2)    initial gripper position

Gripper geometry (Robotiq 2F-85):
    Finger pads sit ~0.130 m below base_mount when fingers point downward.
    Pad centre target z ≈ peg centre z (0.015) → base_mount z ≈ 0.145
    We use Z_GRIP = 0.155 to leave a small clearance margin.

Gripper orientation:
    Default pose (no rotation): fingers open along Y, base pointing up (+Z).
    For picking a peg lying along X we want fingers to open along Y — so the
    DEFAULT quaternion [1, 0, 0, 0] is already correct.
    If your gripper loads pointing up, rotate 180° around X to point fingers down:
        Q_DOWN = [0, 1, 0, 0]
"""

import numpy as np
import mujoco


# ── Actuator 

CTRL_MIN = 0.0      # fully open
CTRL_MAX = 255.0    # fully closed
JAW_MAX  = 0.085    # metres, fully open

def jaw_to_ctrl(jaw_sep: float) -> float:
    """Jaw separation (m) → actuator ctrl value [0, 255]."""
    jaw_sep = float(np.clip(jaw_sep, 0.0, JAW_MAX))
    return CTRL_MAX * (1.0 - jaw_sep / JAW_MAX)

def ctrl_to_jaw(ctrl: float) -> float:
    """Actuator ctrl [0, 255] → jaw separation (m)."""
    ctrl = float(np.clip(ctrl, CTRL_MIN, CTRL_MAX))
    return JAW_MAX * (1.0 - ctrl / CTRL_MAX)


# ── Quaternion helpers ─────────────────────────────────────────────────────────

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    h = angle / 2.0
    return np.array([np.cos(h), *(axis * np.sin(h))])

def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3D vector v by quaternion q (w,x,y,z)."""
    vq = np.array([0.0, v[0], v[1], v[2]])
    return quat_mul(quat_mul(q, vq), quat_conj(q))[1:]

def wrap_pi(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ── Motion helper ──────────────────────────────────────────────────────────────

def move_toward(current: np.ndarray, target: np.ndarray,
                speed: float, dt: float) -> np.ndarray:
    """Step current toward target at given speed (m/s)."""
    delta = target - current
    dist  = np.linalg.norm(delta)
    if dist < 1e-9:
        return target.copy()
    step = min(speed * dt, dist)
    return current + (delta / dist) * step


# ── Contact force helper ───────────────────────────────────────────────────────

def pad_peg_force(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Return total normal-force magnitude between finger pads and peg."""
    pad_bodies = {"left_inner_finger_pad", "right_inner_finger_pad"}
    total = 0.0
    for i in range(data.ncon):
        c  = data.contact[i]
        b1 = model.body(model.geom_bodyid[c.geom1]).name
        b2 = model.body(model.geom_bodyid[c.geom2]).name
        if (b1 in pad_bodies and b2 == "peg") or \
           (b2 in pad_bodies and b1 == "peg"):
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f)
            total += float(np.linalg.norm(f[:3]))
    return total


def peg_cuboid_force(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Net contact force vector between peg and cuboid (world frame)."""
    total = np.zeros(3)
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.body(model.geom_bodyid[c.geom1]).name
        b2 = model.body(model.geom_bodyid[c.geom2]).name
        if (b1 == "peg" and b2 == "cuboid") or (b2 == "peg" and b1 == "cuboid"):
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f)
            total += f[:3]
    return total


# ── Agent ──────────────────────────────────────────────────────────────────────

class PickPlaceAgent:
    """
    Simple 6-state pick-and-place agent.

    Typical usage (handled for you by `simulation.py`):
        agent = PickPlaceAgent(model, data)
        cmd_pos, cmd_quat, cmd_jaw = agent.step(data, dt)
    """

    STATES = [
        "move_above_peg",      # safe-height waypoint above peg
        "lower_to_peg",        # descend to grip height
        "grip",                # light grip near peg end
        "lift",                # initial lift with light grip
        "swing_to_vertical",   # let peg swing until vertical
        "regrip",              # tighten grip once peg is vertical
        "approach_hole",       # intermediate waypoint between peg and cuboid
        "move_above_hole",     # move over cuboid hole
        "fine_align",          # small pose alignment above hole
        "lower_into_hole",     # insert peg along hole axis
        "release",             # open gripper and let go
        "done",
    ]

    # ── Gripper orientation 
    # 180° around X so fingers point downward (-Z).
    Q_DOWN = np.array([0.0, 1.0, 0.0, 0.0])
    # Additional yaw so fingers approach peg from the side (about world Z).
    _Q_YAW = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)
    Q_GRASP = quat_mul(_Q_YAW, Q_DOWN)
    # Alias name used in state machine snippet.
    Q_GRIP = Q_GRASP

    # ── Heights (m) — all relative to base_mount
    Z_ABOVE  = 0.40    # safe travel height (clears cuboid top ~0.10 m)
    Z_GRIP   = 0.155   # grip height: pads at ~z=0.025, just above peg centre
    Z_LIFT   = 0.45    # ceiling for swing lift
    # Insertion depth: tuned so peg enters hole while gripper body stays clear.
    Z_INSERT = 0.04
    RELEASE_LIFT = 0.03  # extra height to lift gripper during release

    # ── Jaw widths
    JAW_OPEN   = 0.085   # fully open
    JAW_CLOSED = 0.030   # nominal closed width on peg
    #   The force controller will stop closing when it hits the peg even if the
    #   command is narrower — so set this a bit tighter than peg width.

    # ── Force control 
    GRIP_FORCE_TARGET       = 1.0   # N  — very light grip to let peg swing
    GRIP_FORCE_TARGET_HIGH  = 8.0   # N  — tighter grip once peg is vertical
    GRIP_CLOSE_RATE         = 0.03  # m/s — jaw closing speed
    RELEASE_OPEN_RATE       = 0.04  # m/s — how fast we open during release
    RELEASE_Z_RATE          = 0.01  # m/s — how fast we lower during release
    ALIGN_YAW_RATE          = 0.8   # rad/s — max yaw correction speed in fine_align
    ALIGN_YAW_TOL           = 0.10  # rad   — consider yaw aligned
    ALIGN_POS_TOL           = 0.003 # m     — tighter XY tolerance above hole
    ALIGN_TIMEOUT           = 1.5   # s
    INSERT_Z_RATE           = 0.02  # m/s — faster insertion descent
    INSERT_XY_GAIN          = 0.20  # unitless — stronger XY correction gain in insert state
    INSERT_XY_MAX_STEP      = 0.003 # m     — allow slightly larger XY correction per step

    # ── Swing / geometry parameters
    GRIP_ALONG_OFFSET = 0.03   # m — offset along peg long axis X (closer to peg end)
    SWING_Z_STEP      = 0.002  # m per step to raise swing ceiling
    VERTICAL_THRESHOLD = 0.99  # peg long axis nearly vertical

    # ── Motion speed 
    SPEED = 0.03   # m/s — slower to reduce overshoot when moving to the hole

    # ── Position tolerance 
    POS_TOL      = 0.008   # m — "close enough" for most transitions
    POS_TOL_FINE = 0.004   # m — tighter tolerance for lowering onto peg

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self._model   = model
        self._peg_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        self._cuboid_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cuboid")

        self.state    = self.STATES[0]
        self.state_t  = 0.0          # time spent in current state
        self._jaw_cmd = self.JAW_OPEN  # tracked jaw command for force control

        # Freeze peg XY once we start gripping so a bouncing peg doesn't confuse us
        self._locked_peg_pos = None

        # Swing ceiling z during swing_to_vertical
        self._swing_z = None
        self._aligned_xy = None

        print(f"[Agent] starting → {self.state}")

    # Public 

    @property
    def is_done(self) -> bool:
        return self.state == "done"

    def step(self, data: mujoco.MjData, dt: float):
        """
        Advance state machine one timestep.

        Returns
        -------
        cmd_pos  : np.ndarray shape (3,)   — desired mocap position
        cmd_quat : np.ndarray shape (4,)   — desired mocap quaternion [w,x,y,z]
        cmd_jaw  : float                   — desired jaw separation (m)
        """
        self.state_t += dt

        # Read world state 
        peg_pos    = data.xpos[self._peg_id].copy()
        cuboid_pos = data.xpos[self._cuboid_id].copy()
        cur_pos    = data.mocap_pos[0].copy()

        # Lock peg reference once grasping starts
        if self.state in (
            "grip",
            "lift",
            "approach_hole",
            "move_above_hole",
            "fine_align",
            "lower_into_hole",
            "release",
            "done",
        ):
            if self._locked_peg_pos is None:
                self._locked_peg_pos = peg_pos.copy()
        else:
            self._locked_peg_pos = None

        ref_peg = self._locked_peg_pos if self._locked_peg_pos is not None else peg_pos

        # Waypoints 
        # Grip near one end of peg long axis (X) so it can swing vertical on lift.
        grip_x     = ref_peg[0] + self.GRIP_ALONG_OFFSET
        above_peg  = np.array([grip_x, ref_peg[1],    self.Z_ABOVE])
        grip_peg   = np.array([grip_x, ref_peg[1],    self.Z_GRIP])
        lift_peg   = np.array([grip_x, ref_peg[1],    self.Z_LIFT])
        above_hole = np.array([cuboid_pos[0], cuboid_pos[1], self.Z_ABOVE])
        insert     = np.array([cuboid_pos[0], cuboid_pos[1], self.Z_INSERT])
        release_pos = np.array([cuboid_pos[0], cuboid_pos[1],
                                self.Z_INSERT + self.RELEASE_LIFT])
        # Intermediate waypoint between peg region and cuboid to avoid a large
        # lateral jump that looks like "flying off then coming back".
        mid_xy = 0.5 * (np.array([grip_x, ref_peg[1]]) +
                        np.array([cuboid_pos[0], cuboid_pos[1]]))
        approach_hole = np.array([mid_xy[0], mid_xy[1], self.Z_ABOVE])

        # State machine 
        s = self.state

        if s == "move_above_peg":
            tgt_pos  = above_peg
            tgt_quat = self.Q_GRIP
            jaw      = self.JAW_OPEN
            if self._close_enough(cur_pos, tgt_pos, 0.01):
                self._advance()

        elif s == "lower_to_peg":
            tgt_pos  = grip_peg
            tgt_quat = self.Q_GRIP
            jaw      = self.JAW_OPEN
            if self._close_enough(cur_pos, tgt_pos, 0.005):
                self._advance()

        elif s == "grip":
            tgt_pos  = grip_peg
            tgt_quat = self.Q_GRIP
            # Force-controlled grip: close slowly until pad-peg contact force reaches target
            pad_force = self._pad_peg_force(data)
            if pad_force < self.GRIP_FORCE_TARGET:
                self._jaw_cmd = max(0.0, self._jaw_cmd - self.GRIP_CLOSE_RATE * dt)
            jaw = self._jaw_cmd
            # Once target force is hit, close 5 mm more to create a preload, then lift
            if pad_force >= self.GRIP_FORCE_TARGET or self.state_t > 4.0:
                self._jaw_cmd = max(0.0, self._jaw_cmd - 0.015)
                self._advance()

        elif s == "lift":
            tgt_pos  = lift_peg
            tgt_quat = self.Q_GRIP
            # Hold preloaded position — steady friction, no impulse
            jaw      = self._jaw_cmd
            if self._close_enough(cur_pos, tgt_pos, 0.01):
                self._advance()

        elif s == "swing_to_vertical":
            # Initialize _swing_z on first entry
            if self._swing_z is None:
                self._swing_z = cur_pos[2]

            # Increment ceiling each timestep, capped at Z_LIFT
            self._swing_z = min(self._swing_z + self.SWING_Z_STEP, self.Z_LIFT)

            tgt_pos  = np.array([grip_x, ref_peg[1], self._swing_z])
            tgt_quat = self.Q_GRIP
            jaw      = self._jaw_cmd   # <-- Added default assignment here

            # Check vertical: world-Z component of peg's long axis
            pq = data.xquat[self._peg_id]
            long_axis_z = abs(1.0 - 2.0 * (pq[1]**2 + pq[2]**2))
            
            # Did it hit our threshold? 
            if long_axis_z > self.VERTICAL_THRESHOLD:
                # It aligned! Advance to regrip to snap the jaws shut.
                self._advance()
            else:
                #  Not vertical yet? Loosen the grip to let it pivot!
                self._jaw_cmd = min(self.JAW_OPEN, self._jaw_cmd + 0.01 * dt)
                jaw = self._jaw_cmd    # <-- Updates if we loosen

                if self._swing_z >= self.Z_LIFT or self.state_t > 8.0:
                    self._advance()

        elif s == "regrip":
            # Hold position wherever swing_to_vertical stopped
            if self._swing_z is None:
                self._swing_z = cur_pos[2]
            tgt_pos  = np.array([grip_x, ref_peg[1], self._swing_z])
            tgt_quat = self.Q_GRIP

            # Force-controlled close to higher target
            pad_force = self._pad_peg_force(data)
            if pad_force < self.GRIP_FORCE_TARGET_HIGH:
                self._jaw_cmd = max(0.0, self._jaw_cmd - self.GRIP_CLOSE_RATE * dt)
            jaw = self._jaw_cmd

            if pad_force >= self.GRIP_FORCE_TARGET_HIGH or self.state_t > 4.0:
                # Apply 15 mm preload then advance
                self._jaw_cmd = max(0.0, self._jaw_cmd - 0.015)
                self._advance()

        elif s == "approach_hole":
            tgt_pos  = approach_hole
            tgt_quat = self.Q_GRIP
            jaw      = self._jaw_cmd
            if self._close_enough(cur_pos, tgt_pos, 0.01):
                self._advance()

        elif s == "move_above_hole":
            tgt_pos  = above_hole
            tgt_quat = self.Q_GRIP
            jaw      = self._jaw_cmd
            if self._close_enough(cur_pos, tgt_pos, 0.01):
                self._advance()

        elif s == "fine_align":
            # 1. Calculate the error between the PEG and the HOLE
            err_xy = cuboid_pos[:2] - peg_pos[:2]
            
            # 2. Adjust the gripper's target to close this gap
            tgt_pos = np.array([
                cur_pos[0] + err_xy[0],
                cur_pos[1] + err_xy[1],
                self.Z_ABOVE
            ])
            tgt_quat = self.Q_GRIP
            jaw      = self._jaw_cmd

            # 3. Wait until the PEG is perfectly centered, then lock the X/Y coordinates
            if np.linalg.norm(err_xy) < self.ALIGN_POS_TOL and self.state_t > 0.5:
                self._aligned_xy = tgt_pos[:2].copy()
                self._advance()
            elif self.state_t > self.ALIGN_TIMEOUT:
                self._aligned_xy = tgt_pos[:2].copy() # Lock and proceed anyway
                self._advance()

        elif s == "lower_into_hole":
            # Dedicated insertion: keep XY near the cuboid COM using the locked
            # alignment, and descend along Z toward Z_INSERT.
            tgt_quat = self.Q_GRIP
            jaw      = self._jaw_cmd

            # Base XY at the locked aligned position from fine_align.
            xy = self._aligned_xy.copy()

            # Optional small correction toward the true cuboid COM to reduce any
            # residual XY error accumulated during motion.
            err_xy = cuboid_pos[:2] - xy
            dxy    = self.INSERT_XY_GAIN * err_xy
            dxy    = np.clip(dxy, -self.INSERT_XY_MAX_STEP, self.INSERT_XY_MAX_STEP)
            xy    += dxy

            # Z descent
            z = float(cur_pos[2])
            z = max(float(self.Z_INSERT), z - self.INSERT_Z_RATE * dt)

            tgt_pos = np.array([xy[0], xy[1], z])

            if abs(z - float(self.Z_INSERT)) < 0.003 or self.state_t > 3.0:
                self._advance()

        elif s == "release":
            # Slow, controlled release: gradually open and slowly lower.
            # Start from current z, clamp between Z_INSERT and release_pos.z.
            cur_z = cur_pos[2]
            target_z = release_pos[2] - self.RELEASE_Z_RATE * self.state_t
            target_z = np.clip(target_z, self.Z_INSERT, release_pos[2])
            tgt_pos  = np.array([release_pos[0], release_pos[1], target_z])
            tgt_quat = self.Q_GRIP

            # Gradually open from current jaw_cmd toward JAW_OPEN
            self._jaw_cmd = min(self.JAW_OPEN,
                                self._jaw_cmd + self.RELEASE_OPEN_RATE * dt)
            jaw = self._jaw_cmd

            # After a short duration, consider release complete
            if self.state_t > 1.5:
                self._advance()

        else:  # done
            # After release, move up to a safe height over the hole.
            tgt_pos  = above_hole
            tgt_quat = self.Q_GRIP
            jaw      = self.JAW_OPEN

        return tgt_pos, tgt_quat, jaw

    # ── Private ────────────────────────────────────────────────────────────────

    def _close_enough(self, current: np.ndarray,
                      target: np.ndarray, tol: float) -> bool:
        return float(np.linalg.norm(current - target)) < tol

    def _pad_peg_force(self, data: mujoco.MjData) -> float:
        """Wrapper around pad_peg_force using this agent's model."""
        return pad_peg_force(self._model, data)

    def _advance(self):
        idx = self.STATES.index(self.state)
        if idx + 1 < len(self.STATES):
            self.state   = self.STATES[idx + 1]
            self.state_t = 0.0
            print(f"[Agent] → {self.state}")