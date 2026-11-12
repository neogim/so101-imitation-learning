"""Optional J5-to-J6 SO-101 wrist camera and photographed bent mounting plate.

Visual geoms and collision proxies are different sets. The source robot URDF,
STL assets, SAC policies and recordings are untouched. Dimensions and frame
pose below are photo-based *estimates*, not an optical/mechanical calibration.
"""

from __future__ import annotations

import math

RIG_BODY = "r2s_wrist_camera_rig"
CAMERA = "r2s_wrist_rgb"
MOUNT_INSPECTION_CAMERA = "r2s_wrist_mount_inspection"

# The J5 wrist_roll child and J6 finger parent is gripper_link. The common
# jaw tips are in approximately the negative Z direction in this frame.
DEFAULT_MOUNT_POS = (0.0, 0.024, -0.045)
DEFAULT_MOUNT_RPY_DEG = (0.0, 0.0, 0.0)
DEFAULT_LOOK_AT = (-0.008, -0.002, -0.165)  # rig-local, beyond jaw tips
DEFAULT_FOVY_DEG = 70.0

# The side-view silhouette is a bent L: a foot screwed to the jaw root,
# a vertical riser and a horizontal upper shelf. Sizes are in metres.
FOOT_POS = (0.0, 0.0, 0.0)
RISER_POS = (0.0, 0.0, 0.018)
TOP_POS = (0.0, 0.022, 0.035)
BOARD_POS = (0.0, 0.043, 0.035)
PCB_HALF = (0.016, 0.016, 0.002)
BARREL_RADIUS_M = 0.008
BARREL_HALF_LENGTH_M = 0.013

COLLISION_NAMES = (
    "r2s_wrist_bracket_foot_collision",
    "r2s_wrist_bracket_vertical_collision",
    "r2s_wrist_bracket_top_collision",
    "r2s_wrist_pcb_collision",
    "r2s_wrist_lens_collision",
)


def validate_mount(position, rpy_deg):
    if len(position) != 3 or len(rpy_deg) != 3:
        raise ValueError("Expected 3D attachment position and RPY")
    vals = [float(v) for v in (*position, *rpy_deg)]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("Non-finite mount coordinate")
    if any(abs(float(v)) > 0.5 for v in position):
        raise ValueError("Mount position exceeds 0.5m; check units")
    return tuple(float(v) for v in position), tuple(float(v) for v in rpy_deg)


def normalize(vec):
    length = math.sqrt(sum(float(x) ** 2 for x in vec))
    if length < 1e-9 or not math.isfinite(length):
        raise ValueError("Zero-length or invalid vector")
    return tuple(float(v) / length for v in vec)


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def camera_axes(origin, target):
    """Return orthonormal MuJoCo xyaxes; the optical axis is local -Z."""
    if len(target) != 3 or any(not math.isfinite(float(v)) for v in target):
        raise ValueError("Bad target")
    # +Z is backwards from the target; camera looks along -Z.
    back = normalize(tuple(float(a)-float(b) for a,b in zip(origin,target)))
    right = normalize(cross((0, 1, 0), back))
    up = normalize(cross(back, right))
    return right, up, back


def quat_from_basis(right, up, back):
    """Rotation quaternion wxyz for local axes (right, up, back)."""
    m00,m01,m02 = right[0],up[0],back[0]
    m10,m11,m12 = right[1],up[1],back[1]
    m20,m21,m22 = right[2],up[2],back[2]
    trace=m00+m11+m22
    if trace > 0:
        s=math.sqrt(trace+1.0)*2
        return (0.25*s,(m21-m12)/s,(m02-m20)/s,(m10-m01)/s)
    if m00 > m11 and m00 > m22:
        s=math.sqrt(1+m00-m11-m22)*2
        return ((m21-m12)/s,0.25*s,(m01+m10)/s,(m02+m20)/s)
    if m11 > m22:
        s=math.sqrt(1+m11-m00-m22)*2
        return ((m02-m20)/s,(m01+m10)/s,0.25*s,(m12+m21)/s)
    s=math.sqrt(1+m22-m00-m11)*2
    return ((m10-m01)/s,(m02+m20)/s,(m12+m21)/s,0.25*s)


def install(spec, *, pos=DEFAULT_MOUNT_POS, rpy_deg=DEFAULT_MOUNT_RPY_DEG,
            fovy=DEFAULT_FOVY_DEG, look_at=DEFAULT_LOOK_AT):
    """Install photo-based L-shaped rig with separate hidden collision proxies."""
    import mujoco
    pos, rpy_deg = validate_mount(pos, rpy_deg)
    if not math.isfinite(fovy) or not 20 <= fovy <= 130:
        raise ValueError("FOV must be 20..130 degrees")
    parent = spec.body("gripper_link")
    if parent is None:
        raise ValueError("Expected J5/J6 gripper_link in source URDF")
    if spec.body(RIG_BODY) is not None:
        raise ValueError("Wrist rig is already installed")

    # A fixed child body follows J5, not the J6 moving jaw.
    rig = parent.add_body(name=RIG_BODY, pos=pos,
                          euler=[math.radians(v) for v in rpy_deg],
                          mass=0.025, ipos=[0, 0.023, 0.024],
                          inertia=[2e-5,2e-5,5e-6])

    # Crucial: collision proxies are group 3 (hidden in ordinary rendering);
    # visual geoms are group 0, no contacts. Both sets share the same poses.
    # Original LeRobot meshes are still rendered normally.
    collision = dict(contype=1, conaffinity=1, group=3,
                     rgba=[0.8,0.3,0.3,0.1], friction=[0.8,0.01,0.001])
    visual = dict(contype=0, conaffinity=0, group=0)

    # Curved real fixture approximated with a connected 90-degree bent L.
    parts = (
        ("foot", FOOT_POS, (0.015, 0.012, 0.004)),
        ("vertical", RISER_POS, (0.007,0.007,0.020)),
        ("top", TOP_POS, (0.010,0.023,0.004)),
    )
    for i,(name,center,halfsize) in enumerate(parts):
        rig.add_geom(name=COLLISION_NAMES[i],
                     type=mujoco.mjtGeom.mjGEOM_BOX, pos=center,
                     size=halfsize, **collision)
        rig.add_geom(name="r2s_wrist_visual_"+name,
                     type=mujoco.mjtGeom.mjGEOM_BOX, pos=center,
                     size=halfsize, rgba=[0.92,0.91,0.87,1], **visual)

    # The board is oriented perpendicular to the *actual optical direction*,
    # not always parallel to the wrist X-Y plane. The lens and optical point
    # are constructed from this same basis, so they cannot disagree.
    right,up,back = camera_axes(BOARD_POS, look_at)
    rotation = quat_from_basis(right,up,back)
    board = tuple(float(v) for v in BOARD_POS)
    lens_center = tuple(board[i]-back[i]*0.014 for i in range(3))
    optical_center = tuple(board[i]-back[i]*0.029 for i in range(3))

    rig.add_geom(name=COLLISION_NAMES[3],
                 type=mujoco.mjtGeom.mjGEOM_BOX, pos=board,
                 quat=rotation, size=PCB_HALF, **collision)
    rig.add_geom(name="r2s_wrist_visual_pcb",
                 type=mujoco.mjtGeom.mjGEOM_BOX, pos=board,
                 quat=rotation, size=PCB_HALF,
                 rgba=[0.12,0.23,0.25,1], **visual)
    rig.add_geom(name=COLLISION_NAMES[4],
                 type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 pos=lens_center, quat=rotation,
                 size=[BARREL_RADIUS_M,BARREL_HALF_LENGTH_M,0], **collision)
    rig.add_geom(name="r2s_wrist_visual_lens",
                 type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 pos=lens_center, quat=rotation,
                 size=[BARREL_RADIUS_M,BARREL_HALF_LENGTH_M,0],
                 rgba=[0.08,0.08,0.09,1], **visual)
    rig.add_camera(name=CAMERA, pos=optical_center,
                   xyaxes=list(right+up), fovy=fovy)

    # The inspection camera is not an ACT input. It stays on the rig so it
    # follows J5 and shows the L from the side for mechanical verification.
    inspection = (0.17,0.14,0.10)
    x,y,_=camera_axes(inspection,(0.0,0.025,0.010))
    rig.add_camera(name=MOUNT_INSPECTION_CAMERA,
                   pos=inspection, xyaxes=list(x+y), fovy=45.)
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
