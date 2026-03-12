"""
simulation.py — Run peg-insertion simulation in agent or teleop mode.

Usage (macOS requires mjpython):
    mjpython simulation.py --mode agent                       # simple agent (default)
    mjpython simulation.py --mode agent --agent advanced      # advanced agent
    mjpython simulation.py --mode teleop
    mjpython simulation.py --mode agent --speed 2.0           # 2x faster
    mjpython simulation.py --mode agent --log-every 50        # log every 50th step
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from control.agent import make_agent, jaw_to_ctrl, ctrl_to_jaw, move_toward
from control.logger import SimLogger

SCENE_PATH = "scene/scene.xml"


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
        w1, x1, y1, z1 = q1;  w2, x2, y2, z2 = q2
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
        elif keycode == ord('I'): rotate([1, 0, 0],  ROT_STEP)
        elif keycode == ord('K'): rotate([1, 0, 0], -ROT_STEP)
        elif keycode == ord('J'): rotate([0, 0, 1],  ROT_STEP)
        elif keycode == ord('L'): rotate([0, 0, 1], -ROT_STEP)
        elif keycode == ord('U'): rotate([0, 1, 0],  ROT_STEP)
        elif keycode == ord('O'): rotate([0, 1, 0], -ROT_STEP)
        elif keycode == ord(' '):
            state["open"] = not state["open"]
            jaw = 0.085 if state["open"] else 0.014
            data.ctrl[0] = jaw_to_ctrl(jaw)

    return on_key


def run(mode: str = "agent",
        agent_kind: str = "simple",
        log_path: str = "logs/run.json",
        log_every: int = 1,
        max_steps: int = 300000,
        time_scale: float = 1.0):

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    mocap_idx = 0

    INIT_POS  = np.array([0.0, 0.0, 0.20])
    INIT_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
    data.mocap_pos[mocap_idx][:]  = INIT_POS
    data.mocap_quat[mocap_idx][:] = INIT_QUAT
    data.ctrl[0] = 0.0

    mujoco.mj_forward(model, data)

    logger = SimLogger(model, data)
    agent  = make_agent(agent_kind, model, data) if mode == "agent" else None

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

        print(f"\n[Sim] mode={mode}  dt={dt}  max_steps={max_steps}"
              f"  speed={time_scale}x  log_every={log_every}")
        if mode == "teleop":
            print("  W/S=Y  A/D=X  Q/E=Z  I/K=pitch  J/L=yaw  U/O=roll  Space=grip")

        # print("Starting in 5 seconds (start recording now)...") # add a timer to start recording after 5 seconds
        # time.sleep(5)

        step = 0
        while viewer.is_running() and (mode == "teleop" or step < max_steps):
            t_wall = time.perf_counter()

            if mode == "agent" and agent is not None:
                cmd_pos, cmd_quat, cmd_jaw = agent.step(data, dt)

                data.mocap_pos[mocap_idx][:] = move_toward(
                    data.mocap_pos[mocap_idx].copy(), cmd_pos,
                    agent.SPEED, dt
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
                cmd_pos  = data.mocap_pos[mocap_idx].copy()
                cmd_quat = data.mocap_quat[mocap_idx].copy()
                cmd_jaw  = ctrl_to_jaw(float(data.ctrl[0]))

            mujoco.mj_step(model, data)

            if step % log_every == 0:
                logger.log(cmd_pos, cmd_quat, cmd_jaw)

            viewer.sync()

            elapsed = time.perf_counter() - t_wall
            budget  = dt / time_scale
            if elapsed < budget:
                time.sleep(budget - elapsed)

            step += 1

    logger.save(log_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Peg insertion simulation")
    p.add_argument("--mode",  choices=["agent", "teleop"], default="agent")
    p.add_argument("--agent", choices=["simple", "advanced"], default="simple",
                   help="Agent variant (only used when --mode agent)")
    p.add_argument("--log",       default="logs/run.json")
    p.add_argument("--log-every", type=int, default=1,
                   help="Log every N-th physics step (default: 1 = every step)")
    p.add_argument("--steps",     type=int, default=30000)
    p.add_argument("--speed",     type=float, default=1.0,
                   help=">1 faster, <1 slower")
    args = p.parse_args()

    run(mode=args.mode, agent_kind=args.agent, log_path=args.log,
        log_every=args.log_every, max_steps=args.steps, time_scale=args.speed)

