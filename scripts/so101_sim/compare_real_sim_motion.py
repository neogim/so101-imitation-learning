#!/usr/bin/env python3
"""Offline recorded-side-video and MuJoCo FK side-by-side comparison.

There are two deliberately separate inputs:
- poses: 5 static photos from the ArUco manifest (no implied temporal order)
- motion: recorded LeRobot episode Side MP4 and logged joint states
The motion mode never executes a policy or drives hardware.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import compare_id1_eye_to_hand as comparison
import real_to_sim as r2s

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INTRINSICS = ROOT / "assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json"
DEFAULT_URDF = ROOT / "data/urdf/so101/so101_new_calib.urdf"


def sample_evenly(samples, count):
    if count < 1:
        raise ValueError("count must be positive")
    if not samples:
        raise ValueError("No valid static poses")
    n = min(count, len(samples))
    return [samples[i] for i in sorted(set(round(float(x)) for x in np.linspace(0, len(samples)-1, n)))]


def pixel_to_base_plane(T_base_cv, intrinsics, u, v, z_m):
    """Intersect an OpenCV camera ray with horizontal plane z=z_m in base frame."""
    T = np.asarray(T_base_cv, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("Invalid camera pose")
    if not all(np.isfinite([u, v, z_m])):
        raise ValueError("Target pixels and height must be finite")
    ray = np.array([(u-float(intrinsics["cx"]))/float(intrinsics["fx"]),
                    (v-float(intrinsics["cy"]))/float(intrinsics["fy"]), 1.])
    ray_base = T[:3, :3] @ ray
    if abs(ray_base[2]) < 1e-6:
        raise ValueError("Camera ray is parallel to tabletop plane")
    distance = (z_m - T[2, 3]) / ray_base[2]
    if distance <= 0:
        raise ValueError("Target pixel ray does not intersect the plane in front of camera")
    return T[:3, 3] + distance * ray_base


def video_object_layout(T_base_cv, intrinsics, muhi_pixel=(255., 360.),
                        cup_pixel=(375., 310.), muhi_yaw_deg=45.,
                        cup_click_height_m=.055):
    """Approximate positions read visually from the *recorded video* first frame.

    These are display-only initial estimates, not measured 3D object poses.
    """
    shift = comparison.workcell_shift_left_front(T_base_cv, .08, .10)
    muhi = pixel_to_base_plane(T_base_cv, intrinsics, *muhi_pixel, .019) - shift
    cup = pixel_to_base_plane(T_base_cv, intrinsics, *cup_pixel, cup_click_height_m) - shift
    return dict(muhi_xy=muhi[:2].tolist(), cup_xy=cup[:2].tolist(),
                muhi_yaw_deg=muhi_yaw_deg)


def make_calibrated_model(extrinsics, intrinsics, urdf, object_layout=None,
                          table_extra_left_m=.04, table_extra_front_m=.05):
    import mujoco

    position, quat = comparison.mujoco_camera_pose(extrinsics["T_base_camera"])
    fovy = comparison.fovy_from_intrinsics(float(intrinsics["fy"]), int(intrinsics["height"]))
    spec = mujoco.MjSpec.from_file(str(urdf.resolve()))
    spec.compiler.discardvisual = False
    shift = comparison.workcell_shift_left_front(extrinsics["T_base_camera"], .08, .10)
    extra = comparison.workcell_shift_left_front(extrinsics["T_base_camera"],
                                                 table_extra_left_m, table_extra_front_m)
    comparison.add_approximate_workcell(spec, shift, extra, 15., **(object_layout or {}))
    comparison.add_backdrop(spec, extrinsics["T_base_camera"])
    comparison.add_simple_studio_light(spec)
    camera = "r2s_side_calibrated_candidate"
    spec.worldbody.add_camera(name=camera, pos=position, quat=quat, fovy=fovy)
    model = spec.compile()
    comparison.apply_white_follower_preview(model)
    return model, camera


def apply_table_base_offset(model, base_xyz_cm=(0., 0., 0.)):
    """Translate ONLY tabletop and four legs in SO-101 base XYZ, never rotate.

    The camera, robot, Muhi and cup stay unchanged. Centimeters are converted
    to meters. This is additive to the older camera-relative table offsets.
    """
    import mujoco

    delta = np.asarray(base_xyz_cm, dtype=float)
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise ValueError("Expected three finite table base offsets in centimeters")
    model.geom_pos[table_geom_ids(model)] += delta / 100.


def table_geom_ids(model):
    """Principal tabletop, four legs and all curved-front segments."""
    import mujoco
    names = ("r2s_preview_table",) + tuple(
        f"r2s_table_leg_{x}{y}" for x in ("m", "p") for y in ("m", "p"))
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
           for name in names]
    if any(i < 0 for i in ids):
        raise ValueError("Preview tabletop or legs missing")
    extras = sorted(
        (name, i) for i in range(model.ngeom)
        if (name := (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
            ).startswith("r2s_table_front_"))
    return ids + [i for _, i in extras]



def set_table_pose_delta(model, reference_positions, reference_quats,
                         base_xyz_m=(0., 0., 0.), yaw_delta_deg=0.):
    """Rigid tabletop+legs transform about tabletop center in SO-101 base frame.

    The provided baseline positions/quats are immutable. Every trial uses
    that baseline, avoiding cumulative drift. Camera, robot and props are fixed.
    """
    from scipy.spatial.transform import Rotation

    ids = table_geom_ids(model)
    original = np.asarray(reference_positions, dtype=float)
    quats = np.asarray(reference_quats, dtype=float)
    offset = np.asarray(base_xyz_m, dtype=float)
    if (original.shape != (len(ids), 3) or quats.shape != (len(ids), 4)
            or offset.shape != (3,) or not np.isfinite(original).all()
            or not np.isfinite(quats).all() or not np.isfinite(offset).all()
            or not np.isfinite(yaw_delta_deg)):
        raise ValueError("Invalid tabletop reference or pose adjustment")
    yaw = Rotation.from_euler("z", float(yaw_delta_deg), degrees=True)
    center = original[0].copy()
    model.geom_pos[ids] = center + yaw.apply(original - center) + offset
    base_rotation = Rotation.from_quat(quats[:, [1, 2, 3, 0]])
    rotated = (yaw * base_rotation).as_quat()
    model.geom_quat[ids] = rotated[:, [3, 0, 1, 2]]


def nearest_table_edge(model, anchor_xy_m=(0., 0.)):
    """Find nearest tabletop edge to robot anchor, using its current world pose.

    Returns a local edge identifier, not a calibrated attachment measurement.
    """
    from scipy.spatial.transform import Rotation

    ids = table_geom_ids(model)
    center = model.geom_pos[ids[0], :2]
    q = model.geom_quat[ids[0]]
    rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    anchor = np.asarray(anchor_xy_m, dtype=float)
    if anchor.shape != (2,) or not np.isfinite(anchor).all():
        raise ValueError("Invalid robot anchor xy")
    local = rotation[:2, :2].T @ (anchor - center)
    if len(ids) > 5:
        half_x, rear_y = .400, -.505 / 2
        front_y = float(comparison.tabletop_front_y_m(
            float(np.clip(local[0], -half_x, half_x))))
        options = {
            "x+": abs(local[0] - half_x), "x-": abs(local[0] + half_x),
            "y+": abs(local[1] - front_y), "y-": abs(local[1] - rear_y),
        }
    else:
        half = model.geom_size[ids[0], :2]
        options = {
            "x+": abs(local[0] - half[0]), "x-": abs(local[0] + half[0]),
            "y+": abs(local[1] - half[1]), "y-": abs(local[1] + half[1]),
        }
    return min(options, key=options.get)


def anchor_table_edge_to_robot(model, edge="auto", anchor_xy_m=(0., 0.),
                               inset_m=0.):
    """Move table+legs ONLY so robot's specified anchor is on a desk edge.

    The geometric constraint uses measured 800x510mm footprint. It does NOT
    move or rotate the robot or camera. The robot origin may not coincide
    with its real clamp point, so an explicit anchor/inset can be supplied.
    """
    from scipy.spatial.transform import Rotation

    if not np.isfinite(inset_m) or inset_m < 0:
        raise ValueError("Invalid robot edge inset")
    selected = nearest_table_edge(model, anchor_xy_m) if edge == "auto" else edge
    if selected not in ("x+", "x-", "y+", "y-"):
        raise ValueError("edge must be auto, x+, x-, y+ or y-")
    ids = table_geom_ids(model)
    table = ids[0]
    center = model.geom_pos[table, :2].copy()
    q = model.geom_quat[table]
    rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()[:2, :2]
    anchor = np.asarray(anchor_xy_m, dtype=float)
    if anchor.shape != (2,) or not np.isfinite(anchor).all():
        raise ValueError("Invalid robot anchor xy")
    axis = 0 if selected[0] == "x" else 1
    sign = 1. if selected[1] == "+" else -1.
    local = rotation.T @ (anchor - center)
    if len(ids) > 5:
        # For the measured curved desk, the front (+Y) depends on X.
        if axis == 0:
            boundary = sign * .400
            extent = .400
        elif sign > 0:
            boundary = float(comparison.tabletop_front_y_m(
                float(np.clip(local[0], -.400, .400))))
            extent = .505
        else:
            boundary = -.505 / 2
            extent = .505
        if inset_m >= extent:
            raise ValueError("Inset must be smaller than table extent")
        target_local = boundary - sign * inset_m
    else:
        half = float(model.geom_size[table, axis])
        if inset_m >= half:
            raise ValueError("Inset must be smaller than table half extent")
        target_local = sign * (half - inset_m)
    displacement = (local[axis] - target_local) * rotation[:, axis]
    model.geom_pos[ids, :2] += displacement
    # Recompute residual and include selection, since this is a constraint,
    # not a guarantee that the specified robot point is the real clamp.
    updated_center = model.geom_pos[table, :2]
    after_local = rotation.T @ (anchor - updated_center)
    return {
        "edge": selected,
        "anchor_xy_m": anchor.tolist(),
        "inset_m": float(inset_m),
        "correction_xy_m": displacement.tolist(),
        "edge_constraint_residual_m": float(after_local[axis] - target_local),
        "table_center_xy_m": updated_center.tolist(),
    }


def project_base_xyz_to_side(T_base_cv, intrinsics, xyz):
    """Project a base-frame 3D point through the same candidate Side camera."""
    T = np.asarray(T_base_cv, dtype=float)
    point = np.asarray(xyz, dtype=float)
    if T.shape != (4, 4) or point.shape != (3,):
        raise ValueError("Invalid base camera transform or point")
    camera_point = T[:3, :3].T @ (point - T[:3, 3])
    if camera_point[2] <= 0:
        raise ValueError("Object point behind candidate Side camera")
    return [
        float(intrinsics["fx"] * camera_point[0] / camera_point[2] + intrinsics["cx"]),
        float(intrinsics["fy"] * camera_point[1] / camera_point[2] + intrinsics["cy"]),
    ]


def muhi_axis_reprojection(model, T_base_cv, intrinsics, annotation):
    """Compare three-click physical endpoints with simulated cap/body ends.

    This diagnostic does not alter geometry; it detects projection / convention
    mismatch instead of silently applying an incorrect yaw correction.
    """
    import mujoco
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r2s_muhi_body")
    cap = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r2s_muhi_cap")
    if min(body, cap) < 0:
        raise ValueError("Missing two-piece Muhi preview geometry")
    b, t = model.geom_pos[body].copy(), model.geom_pos[cap].copy()
    axis = t - b
    axis /= np.linalg.norm(axis)
    body_end = b - axis * float(model.geom_size[body, 1])
    cap_end = t + axis * float(model.geom_size[cap, 1])
    simulated = {
        "body_end_px": project_base_xyz_to_side(T_base_cv, intrinsics, body_end),
        "cap_end_px": project_base_xyz_to_side(T_base_cv, intrinsics, cap_end),
    }
    result = {"sim": simulated, "status": "projection_candidate_not_verified"}
    if annotation is not None:
        orig_body = annotation.get("muhi_body_end_px")
        orig_cap = annotation.get("muhi_cap_end_px")
        if orig_body is not None and orig_cap is not None:
            expected = np.asarray([orig_body, orig_cap], dtype=float)
            projected = np.asarray([simulated["body_end_px"], simulated["cap_end_px"]])
            error = np.linalg.norm(expected - projected, axis=1)
            result["three_click"] = {"body_end_px": orig_body, "cap_end_px": orig_cap}
            result["reprojection_error_px"] = [float(v) for v in error]
            result["max_reprojection_error_px"] = float(error.max())
    return result


def draw_muhi_axis_preview(real_image, diagnostics):
    """Debug on the true real Side frame: green clicks, red simulated axis."""
    from PIL import ImageDraw

    canvas = real_image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    def axis(points, color, label):
        a, b = [tuple(map(float, p)) for p in points]
        draw.line([a, b], fill=color, width=3)
        for point in (a, b):
            x, y = point
            draw.ellipse((x-4, y-4, x+4, y+4), outline=color, width=2)
        draw.text((a[0]+5, a[1]-14), label, fill=color)
    sim = diagnostics["sim"]
    axis([sim["body_end_px"], sim["cap_end_px"]], "red", "SIM body -> cap")
    if "three_click" in diagnostics:
        ref = diagnostics["three_click"]
        axis([ref["body_end_px"], ref["cap_end_px"]], "lime", "CLICK body -> cap")
    return canvas


def comparison_panel(real, sim, subtitle=""):
    width, height = 640, 480
    out = Image.new("RGB", (1280, 508), (25, 25, 25))
    out.paste(real.convert("RGB").resize((width, height)), (0, 28))
    out.paste(sim.convert("RGB").resize((width, height)), (width, 28))
    label = ImageDraw.Draw(out)
    label.text((8, 7), f"REAL: Side {subtitle}", fill="white")
    label.text((width+8, 7), "SIM: preliminary calibrated Side / FK", fill="white")
    return out


def run_poses(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    valid = [s for s in manifest["samples"] if any(int(m["id"]) == 1 for m in s["markers"])]
    poses = sample_evenly(valid, args.count)
    if args.output.exists():
        raise ValueError("Output exists; use a new directory")
    args.output.mkdir(parents=True, exist_ok=False)
    panels, results = [], []
    for i, selected in enumerate(poses):
        frame = int(selected["frame_index"])
        folder = args.output / f"pose_{i+1:02d}_frame_{frame:06d}"
        comparison.render_one(args.manifest, args.extrinsics, args.intrinsics,
                              args.urdf, folder, sample_number=frame)
        path = folder / "real_vs_mujoco.png"
        with Image.open(path) as picture:
            panels.append(picture.convert("RGB"))
        results.append({"frame_index": frame, "image": str(path)})
    combined = Image.new("RGB", (panels[0].width, len(panels)*panels[0].height))
    for i, picture in enumerate(panels):
        combined.paste(picture, (0, i*picture.height))
    output = args.output / "five_pose_comparison.png"
    combined.save(output)
    (args.output / "report.json").write_text(json.dumps({
        "mode": "STATIC_POSE_COMPARISON_NOT_CONTINUOUS_MOTION",
        "calibration": "provisional", "frames": results,
    }, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return output


def side_video_source(root, episode):
    info, metadata = episode["info"], episode["metadata"]
    key = "observation.images.side"
    if key not in info.get("features", {}):
        raise ValueError("Dataset has no observation.images.side")
    if info.get("codebase_version", "").startswith("v3"):
        if metadata is None:
            raise ValueError("v3 episode metadata missing")
        prefix = f"videos/{key}/"
        keys = [prefix+s for s in ("chunk_index","file_index","from_timestamp","to_timestamp")]
        if not all(k in metadata for k in keys):
            raise ValueError("Missing v3 Side video offsets")
        path = root / info["video_path"].format(
            video_key=key, chunk_index=int(metadata[keys[0]]),
            file_index=int(metadata[keys[1]]))
        start_index = round(float(metadata[keys[2]])*float(info["fps"]))
    else:
        found = list(root.glob(f"videos/{key}/chunk-*/episode_{episode['episode']:06d}.mp4"))
        if len(found) != 1:
            raise ValueError(f"Expected 1 Side video for episode; got {len(found)}")
        path, start_index = found[0], 0
    if not path.is_file():
        raise FileNotFoundError(f"Missing Side video: {path}")
    return path, start_index


def read_frames_once(path, indices):
    import av
    needed, selected = set(indices), {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in needed:
                selected[i] = frame.to_image()
                if len(selected) == len(needed):
                    break
    if set(selected) != needed:
        raise ValueError(f"Missing Side frames: {sorted(needed - set(selected))[:5]}")
    return selected


class MuhiGripperAttachment:
    """Kinematic visualization: fixed relative Muhi pose while grasp is active.

    No collision forces, no suction or friction dynamics. This updates only
    the ephemeral simulation model's world geom transforms.
    """
    def __init__(self, model, data):
        import mujoco
        self.model, self.data = model, data
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_link")
        if self.body < 0:
            raise ValueError("MuJoCo gripper_link body missing")
        self.ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("r2s_muhi_body", "r2s_muhi_cap")
        ]
        if any(i < 0 for i in self.ids):
            raise ValueError("Expected Muhi geoms are missing")
        self.rel_pos = None
        self.rel_rot = None
        self.initial_distance_m = None

    def attach(self):
        from scipy.spatial.transform import Rotation
        grip_pos = self.data.xpos[self.body].copy()
        grip_R = self.data.xmat[self.body].reshape(3, 3).copy()
        self.rel_pos = []
        self.rel_rot = []
        for gid in self.ids:
            p = self.data.geom_xpos[gid].copy()
            R = self.data.geom_xmat[gid].reshape(3, 3).copy()
            self.rel_pos.append(grip_R.T @ (p - grip_pos))
            self.rel_rot.append(grip_R.T @ R)
        self.initial_distance_m = float(np.linalg.norm(
            self.data.geom_xpos[self.ids[0]] - grip_pos))

    def follow(self):
        import mujoco
        from scipy.spatial.transform import Rotation
        if self.rel_pos is None:
            raise ValueError("Muhi not attached")
        grip_pos = self.data.xpos[self.body]
        grip_R = self.data.xmat[self.body].reshape(3, 3)
        for gid, p_rel, R_rel in zip(self.ids, self.rel_pos, self.rel_rot):
            self.model.geom_pos[gid] = grip_pos + grip_R @ p_rel
            quat = Rotation.from_matrix(grip_R @ R_rel).as_quat()
            self.model.geom_quat[gid] = quat[[3, 0, 1, 2]]
        mujoco.mj_forward(self.model, self.data)

    def detach(self):
        # Last world positions stay fixed because these geoms have world parent.
        self.rel_pos = None
        self.rel_rot = None


def read_object_targets(path):
    """Interoperable per-recording localization result (e.g., from SAM3 masks).

    Coordinates are in resized 640x480 Side-video pixel space.
    A mask-derived 2D angle is not a full 6DoF object orientation.
    """
    raw = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Compatibility with annotated JSON produced before 2026-11-07 fix:
        # the GUI mistakenly appended a *literal* backslash-n after the JSON.
        if not raw.endswith("\\n"):
            raise
        payload = json.loads(raw[:-2])
    if not isinstance(payload, dict):
        raise ValueError("Object localization JSON must contain an object")
    if payload.get("image_size") != [640, 480]:
        raise ValueError("Object localization JSON must have image_size [640, 480]")
    def center(key):
        value = payload.get(key)
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"Missing 2D center in {key}")
        u, v = map(float, value)
        if not (np.isfinite(u) and np.isfinite(v) and 0 <= u < 640 and 0 <= v < 480):
            raise ValueError(f"Invalid {key} pixel coordinates")
        return (u, v)
    yaw = float(payload["muhi_yaw_deg"])
    if not np.isfinite(yaw):
        raise ValueError("muhi_yaw_deg must be finite")
    return center("muhi_center_px"), center("cup_center_px"), yaw


def set_visual_gripper(model, data, raw_value, invert=False):
    """Provisional display mapping: recorded normalized J6 [0,100] -> URDF range.

    Assumes 0=closed and 100=open unless --invert-gripper is supplied.
    This is NOT a verified LeRobot-to-servo calibration.
    """
    import mujoco
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    if jid < 0:
        raise ValueError("URDF gripper joint not found")
    value = float(raw_value)
    if not np.isfinite(value) or value < -1 or value > 101:
        raise ValueError("Unexpected normalized J6 outside [0,100]")
    value = min(100., max(0., value))
    if invert:
        value = 100. - value
    lower, upper = [float(v) for v in model.jnt_range[jid]]
    angle = lower + value / 100. * (upper - lower)
    data.qpos[int(model.jnt_qposadr[jid])] = angle
    mujoco.mj_forward(model, data)
    return angle


def run_motion(args):
    import av
    import mujoco

    if args.output.exists():
        raise ValueError("Output exists; use a new directory")
    if args.stride <= 0 or args.max_frames <= 0:
        raise ValueError("stride and max-frames must be positive")
    dataset = args.dataset_root.resolve(strict=True)
    if args.output.resolve() == dataset or dataset in args.output.resolve().parents:
        raise ValueError("Output must be outside the raw dataset")
    ep = r2s.read_episode(dataset, args.episode, args.stride, args.max_frames)
    video_path, start = side_video_source(dataset, ep)
    frame_ids = [start + int(x["frame_index"]) for x in ep["rows"]]
    reals = read_frames_once(video_path, frame_ids)

    extrinsics = json.loads(args.extrinsics.read_text(encoding="utf-8"))
    intrinsics = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    if (intrinsics.get("width"), intrinsics.get("height")) != (640, 480):
        raise ValueError("Expected 640x480 Side candidate intrinsics")
    if args.object_layout_json is not None:
        muhi_pixel, cup_pixel, muhi_yaw = read_object_targets(args.object_layout_json)
    else:
        muhi_pixel, cup_pixel = (args.muhi_u, args.muhi_v), (args.cup_u, args.cup_v)
        muhi_yaw = args.muhi_yaw_deg
    layout = video_object_layout(extrinsics["T_base_camera"], intrinsics,
                                 muhi_pixel, cup_pixel, muhi_yaw,
                                 cup_click_height_m=args.cup_click_height_cm / 100.)
    model, camera = make_calibrated_model(
        extrinsics, intrinsics, args.urdf, layout,
        table_extra_left_m=args.table_extra_left_cm / 100.,
        table_extra_front_m=args.table_extra_front_cm / 100.)
    ids = table_geom_ids(model)
    reference_positions = model.geom_pos[ids].copy()
    reference_quats = model.geom_quat[ids].copy()
    set_table_pose_delta(
        model, reference_positions, reference_quats,
        base_xyz_m=np.array([args.table_base_x_cm, args.table_base_y_cm,
                             args.table_base_z_cm]) / 100.,
        yaw_delta_deg=args.table_yaw_delta_deg)
    robot_table_edge = None
    if args.anchor_robot_edge != "off":
        robot_table_edge = anchor_table_edge_to_robot(
            model, args.anchor_robot_edge,
            (args.robot_anchor_x_cm / 100., args.robot_anchor_y_cm / 100.),
            args.robot_edge_inset_cm / 100.)
    annotations = None
    if args.object_layout_json is not None:
        raw = args.object_layout_json.read_text(encoding="utf-8")
        try:
            annotations = json.loads(raw)
        except json.JSONDecodeError:
            if not raw.endswith("\\n"):
                raise
            annotations = json.loads(raw[:-2])
    muhi_projection = muhi_axis_reprojection(model, extrinsics["T_base_camera"],
                                            intrinsics, annotations)
    data = mujoco.MjData(model)
    config = {"joints": {"sign": {name: 1 for name in r2s.JOINTS},
                         "offset_deg": {name: 0. for name in r2s.JOINTS}}}
    if args.attach_frame is not None:
        if args.attach_frame < 0 or (args.release_frame is not None
                                     and args.release_frame <= args.attach_frame):
            raise ValueError("attach-frame must be nonnegative and precede release-frame")
    elif args.release_frame is not None:
        raise ValueError("--release-frame requires --attach-frame")
    attachment = MuhiGripperAttachment(model, data) if args.attach_frame is not None else None
    attached = False
    released = False
    source_fps = float(ep["info"]["fps"])
    fps_out = source_fps / args.stride
    if not (0 < fps_out < 121):
        raise ValueError("Invalid source FPS / stride")
    # Fixed integer frame rate is sufficient for offline visual inspection;
    # original source frame indices are retained in the overlay and report.
    encoded_fps = max(1, round(fps_out))
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "real_vs_sim_episode.mp4"
    with av.open(str(output), "w") as container:
        stream = container.add_stream("libx264", rate=encoded_fps)
        stream.width, stream.height = 1280, 508
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "veryfast", "crf": "23"}
        with mujoco.Renderer(model, width=640, height=480) as renderer:
            for row, source_index in zip(ep["rows"], frame_ids):
                r2s.set_pose(model, data, row["state"], config)
                set_visual_gripper(model, data, row["state"][5], args.invert_gripper)
                frame_index = int(row["frame_index"])
                if attachment is not None:
                    if not attached and frame_index >= args.attach_frame:
                        attachment.attach()
                        attached = True
                    if attached and not released:
                        if args.release_frame is not None and frame_index >= args.release_frame:
                            # Keep Muhi fixed in the world after the previous pose.
                            attachment.detach()
                            released = True
                        else:
                            attachment.follow()
                renderer.update_scene(data, camera=camera)
                simulation = Image.fromarray(renderer.render().copy()).convert("RGB")
                panel = comparison_panel(reals[source_index], simulation,
                                         subtitle=f"ep={args.episode} frame={row['frame_index']}")
                rgb = av.VideoFrame.from_ndarray(np.asarray(panel), format="rgb24")
                for packet in stream.encode(rgb):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    frame0 = reals[frame_ids[0]].convert("RGB").resize((640, 480))
    draw_muhi_axis_preview(frame0, muhi_projection).save(
        args.output / "muhi_axis_reprojection.png")
    (args.output / "report.json").write_text(json.dumps({
        "mode": "RECORDED_REAL_SIDE_VS_MUJOCO_FK_NOT_PHYSICS_REPLAY",
        "output": str(output), "video_source": str(video_path),
        "episode": args.episode, "stride": args.stride, "frames": len(ep["rows"]),
        "source_fps": source_fps, "video_fps": encoded_fps,
        "video_object_pixel_targets": {
            "muhi": list(muhi_pixel),
            "cup": list(cup_pixel),
        },
        "video_object_local_layout": layout,
        "cup_click_reference_height_cm": args.cup_click_height_cm,
        "muhi_axis_reprojection": muhi_projection,
        "j6_mapping": {
            "status": "provisional_normalized_to_urdf_range",
            "invert_gripper": args.invert_gripper,
            "source": "observation.state[5]",
        },
        "object_layout_json_source": str(args.object_layout_json) if args.object_layout_json else None,
        "muhi_attachment": {
            "attach_frame": args.attach_frame,
            "release_frame": args.release_frame,
            "attachment_triggered": attached,
            "release_triggered": released,
            "distance_from_gripper_at_attach_m": (
                attachment.initial_distance_m if attachment is not None else None),
            "behavior": "fixed relative pose to gripper body after attach; no grasp physics",
        },
        "video_object_alignment": "Approximate first-frame pixel matching using November camera model; not actual 3D measured pose.",
        "table_extra_left_cm": args.table_extra_left_cm,
        "table_extra_front_cm": args.table_extra_front_cm,
        "table_base_xyz_cm": [args.table_base_x_cm, args.table_base_y_cm, args.table_base_z_cm],
        "table_yaw_delta_deg": args.table_yaw_delta_deg,
        "table_yaw_deg": 15. + args.table_yaw_delta_deg,
        "robot_table_edge": robot_table_edge,
        "limitations": [
            "The Side calibration dates to 2026-11-07 and may not match the recording setup.",
            "J6 normalized [0,100] is mapped linearly to URDF gripper joint limits for visualization only.",
            "Cup stays static; Muhi follows the arm only if --attach-frame is supplied.",
            "Attachment maintains initial relative offset, does not detect contact or solve grasp mechanics.",
            "If distance_from_gripper_at_attach_m is large, the visual attachment is not physically plausible.",
            "There is no ACT inference and no hardware connection.",
        ],
    }, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="mode", required=True)
    for mode in ("poses", "motion"):
        sub = actions.add_parser(mode)
        sub.add_argument("--extrinsics", type=Path, required=True)
        sub.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
        sub.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
        sub.add_argument("--output", type=Path, required=True)
        if mode == "poses":
            sub.add_argument("--manifest", type=Path, required=True)
            sub.add_argument("--count", type=int, default=5)
        else:
            sub.add_argument("--dataset-root", type=Path, required=True)
            sub.add_argument("--episode", type=int, default=0)
            sub.add_argument("--stride", type=int, default=5)
            sub.add_argument("--max-frames", type=int, default=60)
            sub.add_argument("--muhi-u", type=float, default=255., help="Muhi center x in the 640-pixel Real video panel")
            sub.add_argument("--muhi-v", type=float, default=360., help="Muhi center y in 480-pixel Real video panel")
            sub.add_argument("--cup-click-height-cm", type=float, default=5.5,
                             help="Height of the previously clicked cup point above base z=0 in cm; 5.5 is the historical unverified default, not a measured contact point")
            sub.add_argument("--cup-u", type=float, default=375., help="Clear container center x in the 640-pixel Real video panel")
            sub.add_argument("--cup-v", type=float, default=310., help="Clear container center y in 480-pixel Real video panel")
            sub.add_argument("--muhi-yaw-deg", type=float, default=45., help="Provisional Muhi yaw around base Z; change if axis disagrees")
            sub.add_argument("--table-extra-left-cm", type=float, default=4.,
                             help="Tabletop offset camera-left in cm (default: 4)")
            sub.add_argument("--table-extra-front-cm", type=float, default=5.,
                             help="Tabletop offset toward camera in cm (negative moves farther; default: 5)")
            sub.add_argument("--anchor-robot-edge",
                             choices=("off", "auto", "x+", "x-", "y+", "y-"),
                             default="off",
                             help="Constrain tabletop edge at robot anchor; auto picks nearest baseline edge")
            sub.add_argument("--robot-anchor-x-cm", type=float, default=0.,
                             help="Base-frame X of actual clamp anchor (default origin)")
            sub.add_argument("--robot-anchor-y-cm", type=float, default=0.,
                             help="Base-frame Y of actual clamp anchor (default origin)")
            sub.add_argument("--robot-edge-inset-cm", type=float, default=0.,
                             help="Clamp anchor inset from tabletop edge, cm")
            sub.add_argument("--table-yaw-delta-deg", type=float, default=0.,
                             help="Additional tabletop yaw around own center (deg); default 0")
            sub.add_argument("--table-base-x-cm", type=float, default=0.,
                             help="Additional base X translation of table+legs, cm; no rotation")
            sub.add_argument("--table-base-y-cm", type=float, default=0.,
                             help="Additional base Y translation of table+legs, cm; no rotation")
            sub.add_argument("--table-base-z-cm", type=float, default=0.,
                             help="Additional base Z translation of table+legs, cm; no rotation")
            sub.add_argument("--object-layout-json", type=Path, default=None,
                             help="Optional 640x480 Side-frame centers and Muhi yaw from external detector")
            sub.add_argument("--invert-gripper", action="store_true",
                             help="Reverse unverified J6 normalized open/closed mapping")
            sub.add_argument("--attach-frame", type=int, default=None,
                             help="Recorded episode frame index at grasp; Muhi follows gripper after this")
            sub.add_argument("--release-frame", type=int, default=None,
                             help="Recorded episode frame index at release; Muhi stays at last position")
    args = parser.parse_args()
    print("Saved:", run_poses(args) if args.mode == "poses" else run_motion(args))


if __name__ == "__main__":
    main()
