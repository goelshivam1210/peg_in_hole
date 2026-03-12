"""
agent_simple.py — Simple pick-and-place agent for peg insertion.

Minimal state machine: grip, lift, swing vertical, fly above hole, descend
straight down, release. No tip tracking, no tilt correction, no spiral search,
no contact-aware descent — relies on the 5 mm clearance of the real hole.

Usage:
    mjpython simulation.py --mode agent --agent simple
"""

import numpy as np
import mujoco

from control.agent import pad_peg_force, quat_mul, axis_angle_to_quat


class SimpleAgent:

    STATES = [
        "move_above_peg",
        "lower_to_peg",
        "grip",
        "lift",
        "swing_to_vertical",
        "regrip",
        "move_above_hole",
        "lower_into_hole",
        "release",
        "done",
    ]

    # Gripper orientation: 180 deg around X then 90 deg yaw
    Q_DOWN = np.array([0.0, 1.0, 0.0, 0.0])
    _Q_YAW = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)
    Q_GRIP = quat_mul(_Q_YAW, Q_DOWN)

    # Heights
    Z_ABOVE   = 0.40
    Z_GRIP    = 0.155
    Z_LIFT    = 0.45
    Z_INSERT  = 0.27
    Z_RELEASE = 0.45

    JAW_OPEN = 0.085

    # Grip force
    GRIP_FORCE_LIGHT = 1.0
    GRIP_FORCE_TIGHT = 8.0
    GRIP_CLOSE_RATE  = 0.03

    # Swing
    GRIP_ALONG_OFFSET  = 0.03
    SWING_Z_STEP       = 0.001
    VERTICAL_THRESHOLD = 0.97

    # Release
    RELEASE_OPEN_RATE = 0.06
    RELEASE_Z_RATE    = 0.05

    SPEED   = 0.05
    POS_TOL = 0.008

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self._model     = model
        self._peg_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        self._cuboid_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cuboid")

        self.state    = self.STATES[0]
        self.state_t  = 0.0
        self._jaw_cmd = self.JAW_OPEN

        self._locked_peg_pos = None
        self._swing_z        = None

        print(f"[SimpleAgent] starting -> {self.state}")

    @property
    def is_done(self) -> bool:
        return self.state == "done"

    def step(self, data: mujoco.MjData, dt: float):
        self.state_t += dt

        peg_pos    = data.xpos[self._peg_id].copy()
        cuboid_pos = data.xpos[self._cuboid_id].copy()
        cur_pos    = data.mocap_pos[0].copy()

        if self.state in ("grip", "lift", "swing_to_vertical", "regrip",
                          "move_above_hole", "lower_into_hole",
                          "release", "done"):
            if self._locked_peg_pos is None:
                self._locked_peg_pos = peg_pos.copy()
        else:
            self._locked_peg_pos = None

        ref_peg = self._locked_peg_pos if self._locked_peg_pos is not None else peg_pos
        hole_xy = cuboid_pos[:2]
        grip_x  = ref_peg[0] + self.GRIP_ALONG_OFFSET

        above_peg  = np.array([grip_x,     ref_peg[1], self.Z_ABOVE])
        grip_peg   = np.array([grip_x,     ref_peg[1], self.Z_GRIP])
        lift_peg   = np.array([grip_x,     ref_peg[1], self.Z_LIFT])
        above_hole = np.array([hole_xy[0], hole_xy[1], self.Z_ABOVE])
        insert_tgt = np.array([hole_xy[0], hole_xy[1], self.Z_INSERT])
        retreat    = np.array([hole_xy[0], hole_xy[1], self.Z_RELEASE])

        s = self.state

        if s == "move_above_peg":
            tgt_pos, tgt_quat, jaw = above_peg, self.Q_GRIP, self.JAW_OPEN
            if self._close_enough(cur_pos, tgt_pos, 0.01):
                self._advance()

        elif s == "lower_to_peg":
            tgt_pos, tgt_quat, jaw = grip_peg, self.Q_GRIP, self.JAW_OPEN
            if self._close_enough(cur_pos, tgt_pos, 0.005):
                self._advance()

        elif s == "grip":
            tgt_pos, tgt_quat = grip_peg, self.Q_GRIP
            pf = self._pad_peg_force(data)
            if pf < self.GRIP_FORCE_LIGHT:
                self._jaw_cmd = max(0.0, self._jaw_cmd - self.GRIP_CLOSE_RATE * dt)
            jaw = self._jaw_cmd
            if pf >= self.GRIP_FORCE_LIGHT or self.state_t > 4.0:
                self._jaw_cmd = max(0.0, self._jaw_cmd - 0.015)
                self._advance()

        elif s == "lift":
            tgt_pos, tgt_quat, jaw = lift_peg, self.Q_GRIP, self._jaw_cmd
            if self._close_enough(cur_pos, tgt_pos, 0.01):
                self._swing_z = cur_pos[2]
                self._advance()

        elif s == "swing_to_vertical":
            self._swing_z = min(self._swing_z + self.SWING_Z_STEP, self.Z_LIFT)
            tgt_pos  = np.array([grip_x, ref_peg[1], self._swing_z])
            tgt_quat = self.Q_GRIP
            self._jaw_cmd = min(self.JAW_OPEN, self._jaw_cmd + 0.003 * dt)
            jaw = self._jaw_cmd
            if self._peg_is_vertical(data):
                self._advance()
            elif self._swing_z >= self.Z_LIFT or self.state_t > 10.0:
                self._advance()

        elif s == "regrip":
            tgt_pos  = np.array([grip_x, ref_peg[1], self._swing_z])
            tgt_quat = self.Q_GRIP
            pf = self._pad_peg_force(data)
            if pf < self.GRIP_FORCE_TIGHT:
                self._jaw_cmd = max(0.0, self._jaw_cmd - self.GRIP_CLOSE_RATE * dt)
            jaw = self._jaw_cmd
            if pf >= self.GRIP_FORCE_TIGHT or self.state_t > 4.0:
                self._jaw_cmd = max(0.0, self._jaw_cmd - 0.015)
                self._advance()

        elif s == "move_above_hole":
            tgt_pos, tgt_quat, jaw = above_hole, self.Q_GRIP, self._jaw_cmd
            if self._close_enough(cur_pos, tgt_pos, 0.01):
                self._advance()

        elif s == "lower_into_hole":
            tgt_pos, tgt_quat, jaw = insert_tgt, self.Q_GRIP, self._jaw_cmd
            if self._close_enough(cur_pos, tgt_pos, 0.005) and self.state_t > 0.5:
                self._advance()

        elif s == "release":
            # First, open the gripper while holding the peg at insertion depth
            # so the peg is left in the hole. Then lift the empty gripper.
            self._jaw_cmd = min(self.JAW_OPEN,
                                self._jaw_cmd + self.RELEASE_OPEN_RATE * dt)
            if self.state_t < 0.5:
                tgt_pos = insert_tgt
            else:
                lift_z  = min(self.Z_RELEASE,
                              cur_pos[2] + self.RELEASE_Z_RATE * dt)
                tgt_pos = np.array([hole_xy[0], hole_xy[1], lift_z])
            tgt_quat = self.Q_GRIP
            jaw      = self._jaw_cmd
            if self.state_t > 2.0:
                self._advance()

        else:
            tgt_pos, tgt_quat, jaw = retreat, self.Q_GRIP, self.JAW_OPEN

        return tgt_pos, tgt_quat, jaw

    # helpers

    def _peg_is_vertical(self, data: mujoco.MjData) -> bool:
        pq = data.xquat[self._peg_id]
        return abs(1.0 - 2.0 * (pq[1]**2 + pq[2]**2)) > self.VERTICAL_THRESHOLD

    def _close_enough(self, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
        return float(np.linalg.norm(a - b)) < tol

    def _pad_peg_force(self, data: mujoco.MjData) -> float:
        return pad_peg_force(self._model, data)

    def _advance(self):
        idx = self.STATES.index(self.state)
        if idx + 1 < len(self.STATES):
            elapsed      = self.state_t
            self.state   = self.STATES[idx + 1]
            self.state_t = 0.0
            print(f"[SimpleAgent] -> {self.state}  (prev took {elapsed:.2f}s)")
