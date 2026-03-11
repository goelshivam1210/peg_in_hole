"""
test_viewer.py — Load scene, inspect meshes via MuJoCo, launch viewer.

1. Load the XML
2. Query MuJoCo for mesh/geom info
3. Open the viewer to visually confirm
"""
import mujoco
import mujoco.viewer
import time


# Load
model = mujoco.MjModel.from_xml_path("scene/scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# ---- Scene overview ----
print("=== Scene Overview ===")
print(f"Bodies:   {model.nbody}")
print(f"Geoms:    {model.ngeom}")
print(f"Meshes:   {model.nmesh}")
print(f"Gravity:  {model.opt.gravity}")
print(f"Timestep: {model.opt.timestep}")
print()

# ---- Inspect each mesh ----
print("=== Mesh Dimensions (meters, after scaling) ===")
for i in range(model.nmesh):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)
    vert_start = model.mesh_vertadr[i]
    vert_count = model.mesh_vertnum[i]
    vertices = model.mesh_vert[vert_start:vert_start + vert_count]

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    dims = maxs - mins

    print(f"  {name}:")
    print(f"    min:  ({mins[0]:.4f}, {mins[1]:.4f}, {mins[2]:.4f})")
    print(f"    max:  ({maxs[0]:.4f}, {maxs[1]:.4f}, {maxs[2]:.4f})")
    print(f"    size: {dims[0]:.4f} x {dims[1]:.4f} x {dims[2]:.4f} m")
    print()

# ---- Inspect each body position ----
print("=== Body Positions ===")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"  {name}: {data.xpos[i]}")
print()

# ---- Launch viewer ----
print("Launching viewer... (close window to exit)")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
