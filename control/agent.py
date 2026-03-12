"""
agent.py — Shared utilities for peg-insertion agents.

Contains:
  - Actuator helpers: jaw_to_ctrl, ctrl_to_jaw
  - Quaternion math:  quat_mul, axis_angle_to_quat, quat_conj, quat_rotate
  - Motion helper:    move_toward
  - Contact helper:   pad_peg_force
  - Factory:          make_agent(kind, model, data)

Agent implementations are in agent_simple.py and agent_advanced.py.
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

# Quaternion math

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

# Motion helper (used by simulation.py for mocap tracking)

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
    """Total normal force (N) between finger pads and peg."""
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

# Different agents 
def make_agent(kind: str, model: mujoco.MjModel, data: mujoco.MjData):
    """Instantiate a SimpleAgent or AdvancedAgent.

    Parameters
    ----------
    kind : str
        ``"simple"`` or ``"advanced"``.
    """
    if kind == "advanced":
        from control.agent_advanced import AdvancedAgent
        return AdvancedAgent(model, data)
    from control.agent_simple import SimpleAgent
    return SimpleAgent(model, data)
