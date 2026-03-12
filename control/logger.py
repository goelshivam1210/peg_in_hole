"""
logger.py — Log simulation state at each timestep.

Records commanded and actual gripper pose, jaw separation,
peg state, and contact forces between gripper/peg/cuboid/table.

Usage:
    logger = SimLogger(model, data)
    # in sim loop:
    logger.log(commanded_pos, commanded_quat, commanded_jaw)
    # at end:
    logger.save("logs/run.json")
"""
import json
import os
import numpy as np
import mujoco


_GRIPPER_BODIES = {
    "base_mount", "base",
    "right_driver", "right_coupler", "right_spring_link",
    "right_follower", "right_pad", "right_silicone_pad",
    "left_driver", "left_coupler", "left_spring_link",
    "left_follower", "left_pad", "left_silicone_pad",
}

class SimLogger:
    """Structured logger for the peg-insertion simulation.

    Reads actual gripper/peg state directly from MjData so the caller
    only needs to supply the *commanded* values.  Contact forces are
    summed by category pair (gripper-peg, peg-table, peg-cuboid,
    cuboid-table) as required by the assignment specification.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data  = data
        self.entries: list[dict] = []
        self.step  = 0

        self._peg_body  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        self._base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_mount")

        self._geom_category: dict[int, str] = {}
        for i in range(model.ngeom):
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[i]
            )
            if body_name in _GRIPPER_BODIES:
                self._geom_category[i] = "gripper"
            elif body_name == "peg":
                self._geom_category[i] = "peg"
            elif body_name == "cuboid":
                self._geom_category[i] = "cuboid"
            elif body_name == "world":
                self._geom_category[i] = "table"

        self._right_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_pad")
        self._left_pad  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_pad")

    def _jaw_separation(self) -> float:
        r = self.data.xpos[self._right_pad]
        l = self.data.xpos[self._left_pad]
        return float(np.linalg.norm(r - l))

    def _contact_forces(self) -> dict[str, list[float]]:
        forces = {
            "gripper_peg":   np.zeros(3),
            "peg_table":     np.zeros(3),
            "peg_cuboid":    np.zeros(3),
            "cuboid_table":  np.zeros(3),
        }
        for i in range(self.data.ncon):
            c    = self.data.contact[i]
            cat1 = self._geom_category.get(c.geom1)
            cat2 = self._geom_category.get(c.geom2)
            if cat1 is None or cat2 is None:
                continue
            key = f"{min(cat1,cat2)}_{max(cat1,cat2)}"
            if key in forces:
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                forces[key] += f[:3]
        return {k: v.tolist() for k, v in forces.items()}

    def log(self, commanded_pos: np.ndarray,
            commanded_quat: np.ndarray,
            commanded_jaw: float) -> None:
        self.entries.append({
            "step": self.step,
            "time": round(self.data.time, 6),
            "commanded": {
                "gripper_pos":    commanded_pos.tolist(),
                "gripper_quat":   commanded_quat.tolist(),
                "jaw_separation": round(float(commanded_jaw), 6),
            },
            "actual": {
                "gripper_pos":    self.data.xpos[self._base_body].tolist(),
                "gripper_quat":   self.data.xquat[self._base_body].tolist(),
                "jaw_separation": round(self._jaw_separation(), 6),
            },
            "peg": {
                "pos":  self.data.xpos[self._peg_body].tolist(),
                "quat": self.data.xquat[self._peg_body].tolist(),
            },
            "contacts": self._contact_forces(),
        })
        self.step += 1

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.entries, f, separators=(",", ":"))
        print(f"[Logger] saved {len(self.entries)} timesteps -> {filepath}")