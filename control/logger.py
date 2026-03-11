"""
logger.py — Log simulation state at each timestep.

Records commanded and actual gripper pose, jaw separation,
peg state, and contact forces to a JSON file.

Usage:
    logger = SimLogger(model, data)
    # in sim loop:
    logger.log(commanded_pos, commanded_quat, commanded_jaw)
    # at end:
    logger.save("logs/run.json")
"""
import json
import numpy as np
import mujoco


# Gripper body names — used to classify contacts
GRIPPER_BODIES = {
    "base_mount", "base",
    "right_driver", "right_coupler", "right_spring_link",
    "right_follower", "right_pad", "right_silicone_pad",
    "left_driver", "left_coupler", "left_spring_link",
    "left_follower", "left_pad", "left_silicone_pad",
}


class SimLogger:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.entries = []
        self.step = 0

        # Cache body and geom IDs
        self.peg_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        self.base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_mount")
        self.cuboid_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cuboid")

        # Map each geom to a category: "gripper", "peg", "cuboid", "table"
        self.geom_category = {}
        for i in range(model.ngeom):
            body_id = model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name in GRIPPER_BODIES:
                self.geom_category[i] = "gripper"
            elif body_name == "peg":
                self.geom_category[i] = "peg"
            elif body_name == "cuboid":
                self.geom_category[i] = "cuboid"
            elif body_name == "world":
                self.geom_category[i] = "table"

        # Jaw measurement: pad body IDs
        self.right_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_pad")
        self.left_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_pad")

    def _get_jaw_separation(self):
        """Measure actual jaw separation from pad body positions."""
        r = self.data.xpos[self.right_pad]
        l = self.data.xpos[self.left_pad]
        return float(np.linalg.norm(r - l))

    def _get_contact_forces(self):
        """Sum contact forces by category pair."""
        forces = {
            "gripper_peg": np.zeros(3),
            "peg_table": np.zeros(3),
            "peg_cuboid": np.zeros(3),
            "cuboid_table": np.zeros(3),
        }

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            cat1 = self.geom_category.get(contact.geom1)
            cat2 = self.geom_category.get(contact.geom2)

            if cat1 is None or cat2 is None:
                continue

            # Sort to get consistent key
            pair = tuple(sorted([cat1, cat2]))
            key = f"{pair[0]}_{pair[1]}"

            if key in forces:
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                forces[key] += f[:3]

        return {k: v.tolist() for k, v in forces.items()}

    def log(self, commanded_pos, commanded_quat, commanded_jaw):
        """Record one timestep."""
        entry = {
            "step": self.step,
            "time": round(self.data.time, 6),
            "commanded": {
                "gripper_pos": commanded_pos.tolist(),
                "gripper_quat": commanded_quat.tolist(),
                "jaw_separation": round(commanded_jaw, 6),
            },
            "actual": {
                "gripper_pos": self.data.xpos[self.base_body].tolist(),
                "gripper_quat": self.data.xquat[self.base_body].tolist(),
                "jaw_separation": round(self._get_jaw_separation(), 6),
            },
            "peg": {
                "pos": self.data.xpos[self.peg_body].tolist(),
                "quat": self.data.xquat[self.peg_body].tolist(),
            },
            "contacts": self._get_contact_forces(),
        }

        self.entries.append(entry)
        self.step += 1

    def save(self, filepath):
        """Write all logged entries to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.entries, f, indent=2)
        print(f"Saved {len(self.entries)} timesteps to {filepath}")