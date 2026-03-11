"""
teleop.py — Keyboard teleoperation of the gripper with logging.

Translation:  W/S (Y), A/D (X), Q/E (Z)
Rotation:     I/K (pitch), J/L (yaw), U/O (roll)
Gripper:      Space (toggle open/close)

Run from project root: mjpython control/teleop.py
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys

sys.path.insert(0, os.getcwd())
from control.logger import SimLogger

SCENE_PATH = "scene/scene.xml"

# Load scene
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)

# Logger
os.makedirs("logs", exist_ok=True)
logger = SimLogger(model, data)

# Settings
MOVE_STEP = 0.005
ROT_STEP = 0.05
JAW_OPEN = 0.085
JAW_CLOSED = 0.014

# State
gripper_open = True


def jaw_sep_to_ctrl(jaw_sep):
    """Convert jaw separation (meters) to actuator command (0-255)."""
    t = (jaw_sep - JAW_CLOSED) / (JAW_OPEN - JAW_CLOSED)
    t = max(0.0, min(1.0, t))
    return 255.0 * (1.0 - t)


def quat_multiply(q1, q2):
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def axis_angle_to_quat(axis, angle):
    """Convert axis-angle to quaternion (w, x, y, z)."""
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))])


def rotate_mocap(axis, angle):
    """Apply a small rotation to the mocap body."""
    current = data.mocap_quat[0].copy()
    delta = axis_angle_to_quat(axis, angle)
    data.mocap_quat[0] = quat_multiply(current, delta)


def on_key(keycode):
    global gripper_open
    pos = data.mocap_pos[0]

    # Translation
    if keycode == ord('W'):   pos[1] += MOVE_STEP
    elif keycode == ord('S'): pos[1] -= MOVE_STEP
    elif keycode == ord('A'): pos[0] -= MOVE_STEP
    elif keycode == ord('D'): pos[0] += MOVE_STEP
    elif keycode == ord('Q'): pos[2] += MOVE_STEP
    elif keycode == ord('E'): pos[2] -= MOVE_STEP

    # Rotation
    elif keycode == ord('I'): rotate_mocap([1, 0, 0], ROT_STEP)
    elif keycode == ord('K'): rotate_mocap([1, 0, 0], -ROT_STEP)
    elif keycode == ord('J'): rotate_mocap([0, 0, 1], ROT_STEP)
    elif keycode == ord('L'): rotate_mocap([0, 0, 1], -ROT_STEP)
    elif keycode == ord('U'): rotate_mocap([0, 1, 0], ROT_STEP)
    elif keycode == ord('O'): rotate_mocap([0, 1, 0], -ROT_STEP)

    # Gripper
    elif keycode == ord(' '): gripper_open = not gripper_open


# Main loop
with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    while viewer.is_running():
        jaw_sep = JAW_OPEN if gripper_open else JAW_CLOSED
        data.ctrl[0] = jaw_sep_to_ctrl(jaw_sep)

        mujoco.mj_step(model, data)

        # Log
        logger.log(
            data.mocap_pos[0].copy(),
            data.mocap_quat[0].copy(),
            jaw_sep,
        )

        viewer.sync()
        time.sleep(model.opt.timestep)

# Save log when viewer closes
logger.save("logs/teleop_run.json")