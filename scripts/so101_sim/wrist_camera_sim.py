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
# In the URDF, gripper_link is AFTER wrist_roll (J5), BEFORE gripper (J6).
# The fingers extend along approximately gripper_link -Z; the photographed
# mount starts on their common root, not up at wrist_link / J4.
DEFAULT_MOUNT_POS = (0.0, 0.0, -0.048)  # m, provisional fit to photos
DEFAULT_MOUNT_RPY_DEG = (0.0, 0.0, 0.0)
# Bracket foot attaches to the fingertip-root block (gripper_link-local).
# The mount is one contiguous L-like approximation, not floating blocks.
BRACKET_BASE_Z = 0.0
BRACKET_STEM_CENTER_Z = 0.025
BRACKET_TOP_Z = 0.055
PCB_Z = 0.0615  # overlaps bracket top so the PCB does not float
LENS_Z = 0.048
CAMERA_Z = 0.032
COLLISION_NAMES = (
    "r2s_wrist_bracket_foot_collision",
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


def install(spec, *, pos=DEFAULT_MOUNT_POS,
            rpy_deg=DEFAULT_MOUNT_RPY_DEG, fovy=75.0):
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
    # The fixed child body is anchored at the J5/J6 gripper BASE, not the
    # upstream J4 wrist. A foot box overlaps the solid base by design; its
    # contiguous upright connects the PCB to the base in every frame.
    rig = parent.add_body(name=RIG_BODY, pos=pos, euler=rad,
                          mass=0.025, ipos=[0.0, 0.0, 0.032],
                          inertia=[2.0e-5, 2.0e-5, 5.0e-6])
    common = dict(contype=1, conaffinity=1, friction=[0.8, 0.01, 0.001])
    rig.add_geom(name=COLLISION_NAMES[1], type=mujoco.mjtGeom.mjGEOM_BOX,
                 pos=[0, 0, BRACKET_STEM_CENTER_Z],
                 size=[0.006, 0.009, 0.029],
                 rgba=[0.88, 0.87, 0.83, 1], **common)
    # A broad bracket foot actually touches the parent gripper's root; the
    # stem spans Z=-0.004..+0.054 and the top starts at +0.052.
    rig.add_geom(name=COLLISION_NAMES[0],
                 type=mujoco.mjtGeom.mjGEOM_BOX,
                 pos=[0, 0, BRACKET_BASE_Z],
                 size=[0.014, 0.013, 0.006],
                 rgba=[0.88, 0.87, 0.83, 1], **common)
    rig.add_geom(name=COLLISION_NAMES[2], type=mujoco.mjtGeom.mjGEOM_BOX,
                 pos=[0, 0, BRACKET_TOP_Z],
                 size=[0.019, 0.019, 0.004],
                 rgba=[0.92, 0.91, 0.87, 1], **common)
    rig.add_geom(name=COLLISION_NAMES[3], type=mujoco.mjtGeom.mjGEOM_BOX,
                 pos=[0, 0, PCB_Z], size=[0.016, 0.016, 0.003],
                 rgba=[0.12, 0.23, 0.28, 1], **common)
    # Lens protrudes toward the gripper fingers (-Z), not sideways -Y.
    rig.add_geom(name=COLLISION_NAMES[4],
                 type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 pos=[0, 0, LENS_Z], size=[0.008, 0.012, 0],
                 rgba=[0.08, 0.09, 0.10, 1], **common)
    # Sensor looks almost along -Z toward the finger tips. Slight tilt in
    # local Y keeps a little of the end effector in view. Exact lens extrinsics
    # and FOV require comparison with a real Wrist frame.
    rig.add_camera(name=CAMERA, pos=[0, 0, CAMERA_Z],
                   xyaxes=[1, 0, 0, 0, 0.966, 0.259], fovy=fovy)
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
        id1, id2 = int(c.geom1), int(c.geom2)
        body1, body2 = int(model.geom_bodyid[id1]), int(model.geom_bodyid[id2])
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, id1)
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, id2)
        other_id = id2 if id1 in wrist_ids else id1
        other_body = int(model.geom_bodyid[other_id])
        other_name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_id)
                      or f"geom{other_id}")
        if other_name.startswith(("r2s_preview_table", "r2s_table_front_")):
            category = "table"
        elif other_name.startswith(("r2s_muhi_", "r2s_cup_")):
            category = "object"
        elif other_body == 0:
            category = "world"
        else:
            category = "robot_self"
        result.append({
            "geom1": name1 or f"geom{id1}", "geom2": name2 or f"geom{id2}",
            "geom1_id": id1, "geom2_id": id2,
            "body1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or f"body{body1}",
            "body2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or f"body{body2}",
            "category": category,
            "distance_mm": float(c.dist) * 1000,
            "constraint_active": int(c.efc_address) >= 0,
        })
    return result
