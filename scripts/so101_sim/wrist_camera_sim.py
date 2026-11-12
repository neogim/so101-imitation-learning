"""Optional SO-101 wrist camera rig for the existing MuJoCo MjSpec.

Geometry and optical pose are *provisional*. Dimensions (metres) approximate a
32 mm PCB, a projecting lens, and a bent 3D-printed mount. The real attachment
photo was reviewed but no calibrated CAD-to-link transform has been measured.
The camera and collision proxy follow gripper_link (J5); not the world frame.
No source URDF/STL files are modified.
"""
from __future__ import annotations

import math

RIG_BODY = "r2s_wrist_camera_rig"
CAMERA = "r2s_wrist_rgb"
COLLISION_NAMES = (
    "r2s_wrist_bracket_vertical_collision",
    "r2s_wrist_bracket_top_collision",
    "r2s_wrist_pcb_collision",
    "r2s_wrist_lens_collision",
)


def validate_mount(position, rpy_deg):
    """Validate a local attachment transform without importing MuJoCo."""
    if len(position) != 3 or len(rpy_deg) != 3:
        raise ValueError("Expected 3D attachment position and RPY")
    nums = [float(x) for x in (*position, *rpy_deg)]
    if not all(math.isfinite(x) for x in nums):
        raise ValueError("Attachment values must be finite")
    if any(abs(float(x)) > .5 for x in position):
        raise ValueError("Attachment position exceeds 0.5 m; check units")
    return tuple(float(x) for x in position), tuple(float(x) for x in rpy_deg)


def install(spec, *, pos=(0.0, -0.014, 0.037),
            rpy_deg=(0.0, 0.0, 0.0), fovy=75.0):
    """Add camera + visually distinct collidable body, before spec.compile().

    Assumptions: camera attached to gripper_link (moves with wrist_roll);
    provisional local +Z is 'up' toward the camera. MuJoCo cameras view -Z.
    PCB and lens shapes are intentionally simple so they don't suffer the
    large single-convex-hull artifact found in the source robot meshes.
    """
    import mujoco

    pos, rpy_deg = validate_mount(pos, rpy_deg)
    if not 15 <= fovy <= 150 or not math.isfinite(fovy):
        raise ValueError("Invalid vertical FOV")
    parent = spec.body("gripper_link")
    if parent is None:
        raise ValueError("SO-101 gripper_link was not found in URDF")
    if spec.body(RIG_BODY) is not None:
        raise ValueError("Wrist rig already installed")
    rad = [math.radians(x) for x in rpy_deg]
    # Fixed child body: no joint; the camera follows the parent, including J5.
    rig = parent.add_body(name=RIG_BODY, pos=pos, euler=rad,
                          mass=0.025, ipos=[0.0, 0.0, 0.042],
                          inertia=[2.0e-5, 2.0e-5, 5.0e-6])
    common = dict(contype=1, conaffinity=1, friction=[0.8, 0.01, 0.001])
    rig.add_geom(name=COLLISION_NAMES[0], type=mujoco.mjtGeom.mjGEOM_BOX,
                 pos=[0, 0, 0.023], size=[0.007, 0.009, 0.027],
                 rgba=[0.86, 0.84, 0.77, 1], **common)
    rig.add_geom(name=COLLISION_NAMES[1], type=mujoco.mjtGeom.mjGEOM_BOX,
                 pos=[0, 0, 0.052], size=[0.019, 0.019, 0.004],
                 rgba=[0.9, 0.88, 0.82, 1], **common)
    rig.add_geom(name=COLLISION_NAMES[2], type=mujoco.mjtGeom.mjGEOM_BOX,
                 pos=[0, 0, 0.060], size=[0.016, 0.016, 0.003],
                 rgba=[0.12, 0.23, 0.28, 1], **common)
    # Lens points towards local -Y, approximately matching the user photo.
    rig.add_geom(name=COLLISION_NAMES[3], type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 pos=[0, -0.013, 0.060], euler=[math.pi / 2, 0, 0],
                 size=[0.008, 0.013, 0], rgba=[0.08, 0.09, 0.10, 1],
                 **common)
    # Align optical axis with the approximate lens outward direction (-Y).
    rig.add_camera(name=CAMERA, pos=[0, -0.028, 0.060],
                   xyaxes=[1, 0, 0, 0, 0, 1], fovy=fovy)
    return rig


def enable_environment_contacts(model, *, include_objects=True):
    """Turn on contacts for saved static Real-to-Sim geom previews.

    Collidable environment geoms remain fixed. This enables contact
    *detection/response* against the robot during mj_step, but does NOT turn
    the Muhi bottle into a movable object, or establish realistic contact
    parameters. No global disablement of robot self-collision occurs.
    """
    import mujoco
    names = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if (name == "r2s_preview_table" or
                name.startswith("r2s_table_front_") or
                (include_objects and (
                    name.startswith("r2s_muhi_") or
                    name.startswith("r2s_cup_")))):
            model.geom_contype[i] = 1
            model.geom_conaffinity[i] = 1
            names.append(name)
    if "r2s_preview_table" not in names:
        raise ValueError("Expected saved Real-to-Sim tabletop is missing")
    return names


def contact_summary(model, data):
    """All wrist-rig contacts: negative dist means geometric penetration."""
    import mujoco
    wrist_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in COLLISION_NAMES
    }
    if -1 in wrist_ids:
        raise ValueError("Wrist camera collision geoms are not in the model")
    result = []
    for c in data.contact[:data.ncon]:
        if int(c.geom1) not in wrist_ids and int(c.geom2) not in wrist_ids:
            continue
        result.append({
            "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)),
            "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)),
            "distance_mm": float(c.dist) * 1000,
            "constraint_active": int(c.efc_address) >= 0,
        })
    return result
