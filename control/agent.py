"""
agent.py — Pick-and-place agent for peg insertion scene.

Used by `simulation.py` when running:

    mjpython simulation.py --mode agent

State machine:
    1. move_above_peg    — safe height above peg
    2. lower_to_peg      — descend to grip height
    3. grip              — force-controlled grasp near peg end
    4. lift              — raise peg; gravity swings it toward vertical
    5. swing_to_vertical — slow upward drift lets peg pendulum into vertical
    6. regrip            — tighten grip once peg is vertical
    7. move_above_hole   — fly directly above hole at safe height
    8. align_and_insert  — hover to settle XY, then descend while correcting
                           XY continuously all the way to insertion depth
    9. insert_press      — hold position at depth so peg seats fully
   10. release           — open gripper and lift clear
   11. done

Coordinate frame: Z-up.
    Peg centre:    (0.15, 0,   0.015)  lying flat, long axis along X
    Cuboid centre: (0.35, 0,   0.05)
    base_mount:    (0,    0,   0.2)    initial gripper position

Actuator:
    ctrl=0   -> fully open  (JAW_MAX = 85 mm)
    ctrl=255 -> fully closed
"""

import numpy as np
import mujoco


# Actuator helpers

CTRL_MIN = 0.0
CTRL_MAX = 255.0
JAW_MAX  = 0.085   # m, fully open


def jaw_to_ctrl(jaw_sep: float) -> float:
    """Jaw separation (m) -> actuator ctrl [0, 255]."""
    return CTRL_MAX * (1.0 - float(np.clip(jaw_sep, 0.0, JAW_MAX)) / JAW_MAX)


def ctrl_to_jaw(ctrl: float) -> float:
    """Actuator ctrl [0, 255] -> jaw separation (m)."""
    return JAW_MAX * (1.0 - float(np.clip(ctrl, CTRL_MIN, CTRL_MAX)) / CTRL_MAX)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1;  w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    h = angle / 2.0
    return np.array([np.cos(h), *(axis * np.sin(h))])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3D vector v by quaternion q [w,x,y,z]."""
    vq = np.array([0.0, v[0], v[1], v[2]])
    return quat_mul(quat_mul(q, vq), quat_conj(q))[1:]


# Motion helper (exported -- used by simulation.py for mocap tracking)

def move_toward(current: np.ndarray, target: np.ndarray,
                speed: float, dt: float) -> np.ndarray:
    """Step current toward target at speed (m/s)."""
    delta = target - current
    dist  = np.linalg.norm(delta)
    if dist < 1e-9:
        return target.copy()
    return current + (delta / dist) * min(speed * dt, dist)


# Contact-force helper

def pad_peg_force(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Total normal force (N) between finger pads and peg.

    Body names from 2f85_free.xml: 'left_pad' / 'right_pad'.
    """
    pad_bodies = {"left_pad", "right_pad"}
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


# Agent

class PickPlaceAgent:
    """
    State-machine agent for peg-in-hole insertion.

    Usage:
        agent = PickPlaceAgent(model, data)
        cmd_pos, cmd_quat, cmd_jaw = agent.step(data, dt)
    """

    STATES = [
        "move_above_peg",
        "lower_to_peg",
        "grip",
        "lift",
        "swing_to_vertical",
        "regrip",
        "move_above_hole",
        "align_and_insert",
        "insert_press",
        "release",
        "done",
    ]

    # Gripper orientation: 180 deg around X (fingers -Z), then 90 deg yaw
    Q_DOWN = np.array([0.0, 1.0, 0.0, 0.0])
    _Q_YAW = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)
    Q_GRIP = quat_mul(_Q_YAW, Q_DOWN)

    # Heights (m)
    Z_ABOVE   = 0.40    # safe travel height
    Z_GRIP    = 0.155   # grip height (pads just above peg centre z=0.015)
    Z_LIFT    = 0.45    # ceiling for lift / swing
    Z_INSERT  = 0.27    # gripper Z when peg is fully seated at hole floor
    Z_RELEASE = 0.45    # retreat height after release (above cuboid)

    # Jaw widths (m)
    JAW_OPEN = 0.085

    # Grip force control
    GRIP_FORCE_LIGHT = 1.0    # N -- light grip for swing phase
    GRIP_FORCE_TIGHT = 8.0    # N -- firm grip for insertion
    GRIP_CLOSE_RATE  = 0.03   # m/s -- jaw closing speed

    # Swing
    GRIP_ALONG_OFFSET  = 0.03   # m -- grip offset from peg centre along X
    SWING_Z_STEP       = 0.001  # m per timestep -- slow rise lets peg pendulum
    VERTICAL_THRESHOLD = 0.97   # |cos| ~14 deg from vertical

    # align_and_insert: Phase A (XY hover)
    ALIGN_HOVER_TIME = 1.0     # s -- minimum hover before descending
    ALIGN_MAX_TIME   = 4.0     # s -- hard timeout: proceed to descent even if not perfect
    ALIGN_XY_TOL     = 0.004   # m -- XY error to consider aligned
    ALIGN_XY_GAIN    = 2.0     # P-gain during hover
    ALIGN_XY_MAX     = 0.005   # m -- max XY correction per step during hover

    # align_and_insert: Phase B (descend while tracking)
    INSERT_Z_RATE  = 0.025   # m/s -- faster descent gives momentum to enter hole
    INSERT_XY_GAIN = 2.0     # P-gain during descent (aggressive tip tracking)
    INSERT_XY_MAX  = 0.005   # m -- max XY correction per step during descent

    # align_and_insert: tilt correction (both phases)
    TILT_GAIN     = 0.5    # P-gain on peg tilt angles (rad/rad/s)
    TILT_MAX_STEP = 0.02   # rad -- max rotational correction per step
    TILT_TOL      = 0.07   # rad (~4 deg) -- tilt must be below this to start descent

    # Peg geometry (from STL: 100mm long × 0.001 scale)
    PEG_HALF_LEN = 0.05   # m -- half-length of peg

    # Phase B: contact-aware descent + spiral search
    CONTACT_FORCE_PAUSE = 25.0  # N -- push through rim-sitting forces (~17-21N)
    SEARCH_FREQ         = 2.0   # Hz -- spiral rotation speed
    SEARCH_R_GROW       = 0.002 # m/s -- spiral radius growth rate
    SEARCH_R_MAX        = 0.004 # m -- max spiral search radius
    DESCENT_MAX_TIME    = 15.0  # s -- Phase B hard timeout

    # Offset from cuboid body origin to hole centre (tune if STL origin != hole)
    HOLE_XY_OFFSET = np.array([0.0, 0.0])

    # insert_press
    INSERT_PRESS_TIME  = 1.0   # s -- hold at depth to let peg settle

    # release
    RELEASE_OPEN_RATE = 0.06   # m/s -- jaw opening speed
    RELEASE_Z_RATE    = 0.05   # m/s -- gripper lift speed

    # General
    SPEED   = 0.05    # m/s -- used by simulation.py move_toward
    POS_TOL = 0.008   # m

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self._model     = model
        self._peg_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        self._cuboid_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cuboid")

        self.state    = self.STATES[0]
        self.state_t  = 0.0
        self._jaw_cmd = self.JAW_OPEN

        self._locked_peg_pos = None   # stable peg XY once grasped
        self._swing_z        = None   # swing ceiling raised gradually
        self._descent_t      = 0.0   # time spent in Phase B (for spiral)

        print(f"[Agent] starting -> {self.state}")

    @property
    def is_done(self) -> bool:
        return self.state == "done"

    def step(self, data: mujoco.MjData, dt: float):
        """
        Returns
        -------
        cmd_pos  : np.ndarray (3,)  desired mocap position
        cmd_quat : np.ndarray (4,)  desired mocap quaternion [w,x,y,z]
        cmd_jaw  : float            desired jaw separation (m)
        """
        self.state_t += dt

        peg_pos    = data.xpos[self._peg_id].copy()
        cuboid_pos = data.xpos[self._cuboid_id].copy()
        cur_pos    = data.mocap_pos[0].copy()

        # Lock peg reference once grasped
        if self.state in ("grip", "lift", "swing_to_vertical", "regrip",
                          "move_above_hole", "align_and_insert",
                          "insert_press", "release", "done"):
            if self._locked_peg_pos is None:
                self._locked_peg_pos = peg_pos.copy()
        else:
            self._locked_peg_pos = None

        ref_peg = self._locked_peg_pos if self._locked_peg_pos is not None else peg_pos
        hole_xy = cuboid_pos[:2] + self.HOLE_XY_OFFSET
        grip_x  = ref_peg[0] + self.GRIP_ALONG_OFFSET

        # Waypoints
        above_peg  = np.array([grip_x,     ref_peg[1],  self.Z_ABOVE])
        grip_peg   = np.array([grip_x,     ref_peg[1],  self.Z_GRIP])
        lift_peg   = np.array([grip_x,     ref_peg[1],  self.Z_LIFT])
        above_hole = np.array([hole_xy[0], hole_xy[1],  self.Z_ABOVE])
        insert_tgt = np.array([hole_xy[0], hole_xy[1],  self.Z_INSERT])
        retreat    = np.array([hole_xy[0], hole_xy[1],  self.Z_RELEASE])

        s = self.state

        # 1. move_above_peg
        if s == "move_above_peg":
            tgt_pos, tgt_quat, jaw = above_peg, self.Q_GRIP, self.JAW_OPEN
            if self._close_enough(cur_pos, tgt_pos, 0.01):
                self._advance()

        # 2. lower_to_peg
        elif s == "lower_to_peg":
            tgt_pos, tgt_quat, jaw = grip_peg, self.Q_GRIP, self.JAW_OPEN
            if self._close_enough(cur_pos, tgt_pos, 0.005):
                self._advance()

        # 3. grip
        elif s == "grip":
            tgt_pos, tgt_quat = grip_peg, self.Q_GRIP
            pad_force = self._pad_peg_force(data)
            if pad_force < self.GRIP_FORCE_LIGHT:
                self._jaw_cmd = max(0.0, self._jaw_cmd - self.GRIP_CLOSE_RATE * dt)
            jaw = self._jaw_cmd
            if pad_force >= self.GRIP_FORCE_LIGHT or self.state_t > 4.0:
                self._jaw_cmd = max(0.0, self._jaw_cmd - 0.015)
                self._advance()

        # 4. lift
        elif s == "lift":
            tgt_pos, tgt_quat, jaw = lift_peg, self.Q_GRIP, self._jaw_cmd
            if self._close_enough(cur_pos, tgt_pos, 0.01):
                self._swing_z = cur_pos[2]
                self._advance()

        # 5. swing_to_vertical
        # Rise slowly in a straight line. The peg hangs from one end with a
        # loose grip -- gravity naturally swings it vertical as the gripper
        # rises. No lateral nudge (that caused the peg to be thrown).
        elif s == "swing_to_vertical":
            self._swing_z = min(self._swing_z + self.SWING_Z_STEP, self.Z_LIFT)
            tgt_pos  = np.array([grip_x, ref_peg[1], self._swing_z])
            tgt_quat = self.Q_GRIP
            # Loosen grip just enough to let the peg pivot freely
            self._jaw_cmd = min(self.JAW_OPEN, self._jaw_cmd + 0.003 * dt)
            jaw = self._jaw_cmd

            if self._peg_is_vertical(data):
                print(f"[Agent]   peg vertical at t={self.state_t:.2f}s")
                self._advance()
            elif self._swing_z >= self.Z_LIFT or self.state_t > 10.0:
                print(f"[Agent]   swing timeout -- proceeding")
                self._advance()

        # 6. regrip
        elif s == "regrip":
            tgt_pos  = np.array([grip_x, ref_peg[1], self._swing_z])
            tgt_quat = self.Q_GRIP
            pad_force = self._pad_peg_force(data)
            if pad_force < self.GRIP_FORCE_TIGHT:
                self._jaw_cmd = max(0.0, self._jaw_cmd - self.GRIP_CLOSE_RATE * dt)
            jaw = self._jaw_cmd
            if pad_force >= self.GRIP_FORCE_TIGHT or self.state_t > 4.0:
                self._jaw_cmd = max(0.0, self._jaw_cmd - 0.015)
                self._advance()

        # 7. move_above_hole
        elif s == "move_above_hole":
            tgt_pos, tgt_quat, jaw = above_hole, self.Q_GRIP, self._jaw_cmd
            if self._close_enough(cur_pos, tgt_pos, 0.01):
                self._advance()

        # 8. align_and_insert
        #
        # Key insight: align the peg BOTTOM TIP (not COM) with the hole.
        # A small tilt causes the tip to be several mm off from the COM XY.
        #
        # Phase A -- hover at Z_ABOVE, correct tip XY + tilt.
        # Phase B -- descend while tracking tip XY + tilt + spiral search.
        elif s == "align_and_insert":
            jaw = self._jaw_cmd

            # --- Peg tip computation ---
            peg_q      = data.xquat[self._peg_id].copy()
            peg_z_axis = quat_rotate(peg_q, np.array([0.0, 0.0, 1.0]))
            sign       = 1.0 if peg_z_axis[2] > 0 else -1.0
            tip_pos    = peg_pos - sign * peg_z_axis * self.PEG_HALF_LEN
            tip_xy     = tip_pos[:2]

            # --- XY error (tip to hole) ---
            xy_err  = hole_xy - tip_xy
            xy_dist = np.linalg.norm(xy_err)

            # --- Peg tilt ---
            tilt_x   = float(np.arctan2(peg_z_axis[1], peg_z_axis[2]))
            tilt_y   = float(np.arctan2(-peg_z_axis[0], peg_z_axis[2]))
            tilt_mag = float(np.sqrt(tilt_x**2 + tilt_y**2))

            # Rotational correction: tilt mocap opposite to measured peg tilt
            corr_x = np.clip(-self.TILT_GAIN * tilt_x * dt,
                             -self.TILT_MAX_STEP, self.TILT_MAX_STEP)
            corr_y = np.clip(-self.TILT_GAIN * tilt_y * dt,
                             -self.TILT_MAX_STEP, self.TILT_MAX_STEP)
            dq_x = axis_angle_to_quat(np.array([1.0, 0.0, 0.0]), corr_x)
            dq_y = axis_angle_to_quat(np.array([0.0, 1.0, 0.0]), corr_y)
            tgt_quat = quat_mul(dq_y, quat_mul(dq_x, self.Q_GRIP))

            # Phase gate
            timed_out = self.state_t > self.ALIGN_MAX_TIME
            still_hovering = (not timed_out
                              and (self.state_t < self.ALIGN_HOVER_TIME
                                   or xy_dist > self.ALIGN_XY_TOL
                                   or tilt_mag > self.TILT_TOL))

            if still_hovering:
                # Phase A: hold Z, nudge XY toward tip alignment
                dxy    = np.clip(self.ALIGN_XY_GAIN * xy_err * dt,
                                 -self.ALIGN_XY_MAX, self.ALIGN_XY_MAX)
                tgt_xy = cur_pos[:2] + dxy
                tgt_z  = self.Z_ABOVE
                self._descent_t = 0.0
            else:
                # Phase B: contact-aware helical descent
                self._descent_t += dt

                # XY: P-gain tracking of tip to hole center
                dxy = np.clip(self.INSERT_XY_GAIN * xy_err * dt,
                              -self.INSERT_XY_MAX, self.INSERT_XY_MAX)

                # Spiral search: velocity-based circular motion overlaid
                r     = min(self.SEARCH_R_GROW * self._descent_t,
                            self.SEARCH_R_MAX)
                omega = 2.0 * np.pi * self.SEARCH_FREQ
                angle = omega * self._descent_t
                spiral_v = r * omega * np.array([-np.sin(angle),
                                                  np.cos(angle)])
                tgt_xy = cur_pos[:2] + dxy + spiral_v * dt

                # Z: only descend when peg-cuboid force is low (not caught)
                peg_cub_f = self._peg_cuboid_force(data)
                if peg_cub_f < self.CONTACT_FORCE_PAUSE:
                    tgt_z = max(float(self.Z_INSERT),
                                cur_pos[2] - self.INSERT_Z_RATE * dt)
                else:
                    tgt_z = cur_pos[2]

            tgt_pos = np.array([tgt_xy[0], tgt_xy[1], tgt_z])

            # Advance when at depth or Phase B times out
            if not still_hovering and (tgt_z <= self.Z_INSERT + 0.003
                                       or self._descent_t > self.DESCENT_MAX_TIME):
                self._advance()

        # 9. insert_press
        # Hold at insertion depth to let peg settle in the hole.
        elif s == "insert_press":
            tgt_pos  = insert_tgt
            tgt_quat = self.Q_GRIP
            jaw      = self._jaw_cmd
            if self.state_t > self.INSERT_PRESS_TIME:
                self._advance()

        # 10. release
        elif s == "release":
            # Open jaw and lift straight up -- never moves down
            self._jaw_cmd = min(self.JAW_OPEN,
                                self._jaw_cmd + self.RELEASE_OPEN_RATE * dt)
            lift_z  = min(self.Z_RELEASE,
                          cur_pos[2] + self.RELEASE_Z_RATE * dt)
            tgt_pos  = np.array([hole_xy[0], hole_xy[1], lift_z])
            tgt_quat = self.Q_GRIP
            jaw      = self._jaw_cmd
            if self.state_t > 2.0:
                self._advance()

        # 11. done
        else:
            tgt_pos, tgt_quat, jaw = retreat, self.Q_GRIP, self.JAW_OPEN

        return tgt_pos, tgt_quat, jaw

    # Private helpers

    def _peg_is_vertical(self, data: mujoco.MjData) -> bool:
        """True when peg long axis is within ~14 deg of world Z.

        R[2][2] = 1 - 2*(qx^2 + qy^2) equals 1 when local-Z aligns with world-Z.
        """
        pq = data.xquat[self._peg_id]
        return abs(1.0 - 2.0 * (pq[1]**2 + pq[2]**2)) > self.VERTICAL_THRESHOLD

    def _close_enough(self, current: np.ndarray,
                      target: np.ndarray, tol: float) -> bool:
        return float(np.linalg.norm(current - target)) < tol

    def _pad_peg_force(self, data: mujoco.MjData) -> float:
        return pad_peg_force(self._model, data)

    def _peg_cuboid_force(self, data: mujoco.MjData) -> float:
        """Total contact force between peg and cuboid."""
        total = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = self._model.body(self._model.geom_bodyid[c.geom1]).name
            b2 = self._model.body(self._model.geom_bodyid[c.geom2]).name
            if (b1 == "peg" and b2 == "cuboid") or \
               (b2 == "peg" and b1 == "cuboid"):
                f = np.zeros(6)
                mujoco.mj_contactForce(self._model, data, i, f)
                total += float(np.linalg.norm(f[:3]))
        return total

    def _advance(self):
        idx = self.STATES.index(self.state)
        if idx + 1 < len(self.STATES):
            elapsed      = self.state_t
            self.state   = self.STATES[idx + 1]
            self.state_t = 0.0
            print(f"[Agent] -> {self.state}  (prev took {elapsed:.2f}s)")