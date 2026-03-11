"""
simulation.py — Run peg-insertion simulation in agent or teleop mode.

Usage (macOS requires mjpython):
    mjpython simulation.py --mode agent
    mjpython simulation.py --mode teleop
    mjpython simulation.py --mode agent --speed 2.0   # 2× faster
"""

import argparse
import json
import time
import os

import mujoco
import mujoco.viewer
import numpy as np

# Import agent 
from control.agent import PickPlaceAgent, jaw_to_ctrl, ctrl_to_jaw, move_toward

SCENE_PATH = "scene/scene.xml" 


# Contact force helper

def get_contact_forces(model: mujoco.MjModel, data: mujoco.MjData) -> list:
    """Return a list of contact dicts for the current timestep."""
    contacts = []
    for i in range(data.ncon):
        c  = data.contact[i]
        b1 = model.body(model.geom_bodyid[c.geom1]).name
        b2 = model.body(model.geom_bodyid[c.geom2]).name
        f  = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, f)
        contacts.append({
            "body1":     b1,
            "body2":     b2,
            "force":     f[:3].tolist(),
            "torque":    f[3:].tolist(),
            "magnitude": float(np.linalg.norm(f[:3])),
        })
    return contacts


# Logger

class Logger:
    def __init__(self, path: str):
        self.path    = path
        self.records = []

    def log(self, *, sim_time, cmd_pos, cmd_quat, cmd_jaw,
            actual_pos, actual_quat, actual_jaw,
            peg_pos, peg_quat, contacts):
        self.records.append({
            "time": round(sim_time, 6),
            "commanded": {
                "gripper_pos":  cmd_pos,
                "gripper_quat": cmd_quat,
                "jaw_sep":      round(cmd_jaw, 6),
            },
            "actual": {
                "gripper_pos":  actual_pos,
                "gripper_quat": actual_quat,
                "jaw_sep":      round(actual_jaw, 6),
            },
            "peg": {
                "pos":  peg_pos,
                "quat": peg_quat,
            },
            "contacts": contacts,
        })

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.records, f, indent=2)
        print(f"[Logger] saved {len(self.records)} records → {self.path}")


# Teleop key handler

def make_key_handler(data):
    """Return a key_callback for teleop mode."""
    MOVE_STEP = 0.005
    ROT_STEP  = 0.05
    state     = {"open": True}

    def axis_angle_to_quat(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        h = angle / 2.0
        return np.array([np.cos(h), *(axis * np.sin(h))])

    def quat_mul(q1, q2):
        w1,x1,y1,z1 = q1;  w2,x2,y2,z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def rotate(axis, angle):
        q = quat_mul(data.mocap_quat[0].copy(),
                     axis_angle_to_quat(axis, angle))
        data.mocap_quat[0] = q

    def on_key(keycode):
        pos = data.mocap_pos[0]
        if   keycode == ord('W'): pos[1] += MOVE_STEP
        elif keycode == ord('S'): pos[1] -= MOVE_STEP
        elif keycode == ord('A'): pos[0] -= MOVE_STEP
        elif keycode == ord('D'): pos[0] += MOVE_STEP
        elif keycode == ord('Q'): pos[2] += MOVE_STEP
        elif keycode == ord('E'): pos[2] -= MOVE_STEP
        elif keycode == ord('I'): rotate([1,0,0],  ROT_STEP)
        elif keycode == ord('K'): rotate([1,0,0], -ROT_STEP)
        elif keycode == ord('J'): rotate([0,0,1],  ROT_STEP)
        elif keycode == ord('L'): rotate([0,0,1], -ROT_STEP)
        elif keycode == ord('U'): rotate([0,1,0],  ROT_STEP)
        elif keycode == ord('O'): rotate([0,1,0], -ROT_STEP)
        elif keycode == ord(' '):
            state["open"] = not state["open"]
            jaw = 0.085 if state["open"] else 0.014
            data.ctrl[0] = jaw_to_ctrl(jaw)

    return on_key


#  Main loop

def run(mode: str = "agent",
        log_path: str = "logs/run.json",
        max_steps: int = 30000,
        time_scale: float = 1.0):

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # IDs
    peg_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    mocap_idx = 0   # first (and only) mocap body

    # Initial mocap pose — fingers pointing down
    INIT_POS  = np.array([0.0, 0.0, 0.20])
    INIT_QUAT = np.array([0.0, 1.0, 0.0, 0.0])   # 180° around X → fingers -Z
    data.mocap_pos[mocap_idx][:]  = INIT_POS
    data.mocap_quat[mocap_idx][:] = INIT_QUAT
    data.ctrl[0] = 0.0   # fully open

    mujoco.mj_forward(model, data)

    logger = Logger(log_path)
    agent  = PickPlaceAgent(model, data) if mode == "agent" else None

    # Commanded state (updated each step)
    cmd_pos  = INIT_POS.copy()
    cmd_quat = INIT_QUAT.copy()
    cmd_jaw  = 0.085

    dt = float(model.opt.timestep)

    key_cb = make_key_handler(data) if mode == "teleop" else None

    with mujoco.viewer.launch_passive(
            model, data,
            key_callback=key_cb) as viewer:

        viewer.cam.distance  = 1.2
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 135

        print(f"\n[Sim] mode={mode}  dt={dt}  max_steps={max_steps}  speed={time_scale}×")
        if mode == "teleop":
            print("  W/S=Y  A/D=X  Q/E=Z  I/K=pitch  J/L=yaw  U/O=roll  Space=grip")

        for step in range(max_steps):
            t_wall = time.perf_counter()

            # Commands
            if mode == "agent" and agent is not None:
                cmd_pos, cmd_quat, cmd_jaw = agent.step(data, dt)

                # Smooth position tracking
                data.mocap_pos[mocap_idx][:] = move_toward(
                    data.mocap_pos[mocap_idx].copy(), cmd_pos,
                    PickPlaceAgent.SPEED, dt
                )
                data.mocap_quat[mocap_idx][:] = cmd_quat
                data.ctrl[0] = jaw_to_ctrl(cmd_jaw)

                if agent.is_done:
                    print("[Agent] task complete — close viewer to exit.")
                    while viewer.is_running():
                        mujoco.mj_step(model, data)
                        viewer.sync()
                        time.sleep(dt)
                    break

            else:
                # Teleop: read back what the keyboard set
                cmd_pos  = data.mocap_pos[mocap_idx].copy()
                cmd_quat = data.mocap_quat[mocap_idx].copy()
                cmd_jaw  = ctrl_to_jaw(float(data.ctrl[0]))

            # Physics 
            mujoco.mj_step(model, data)

            # Logging (every 50 steps) 
            if step % 50 == 0:
                logger.log(
                    sim_time    = float(data.time),
                    cmd_pos     = cmd_pos.tolist(),
                    cmd_quat    = cmd_quat.tolist(),
                    cmd_jaw     = float(cmd_jaw),
                    actual_pos  = data.mocap_pos[mocap_idx].tolist(),
                    actual_quat = data.mocap_quat[mocap_idx].tolist(),
                    actual_jaw  = ctrl_to_jaw(float(data.ctrl[0])),
                    peg_pos     = data.xpos[peg_id].tolist(),
                    peg_quat    = data.xquat[peg_id].tolist(),
                    contacts    = get_contact_forces(model, data),
                )

            viewer.sync()

            # Real-time pacing 
            elapsed = time.perf_counter() - t_wall
            budget  = dt / time_scale
            if elapsed < budget:
                time.sleep(budget - elapsed)

    logger.save()


# Entry point

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Peg insertion simulation")
    p.add_argument("--mode",  choices=["agent", "teleop"], default="agent")
    p.add_argument("--log",   default="logs/run.json")
    p.add_argument("--steps", type=int,   default=30000)
    p.add_argument("--speed", type=float, default=1.0,
                   help=">1 faster, <1 slower")
    args = p.parse_args()

    run(mode=args.mode, log_path=args.log,
        max_steps=args.steps, time_scale=args.speed)