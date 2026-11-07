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
                        cup_pixel=(375., 310.), muhi_yaw_deg=45.):
    """Approximate positions read visually from the *recorded video* first frame.

    These are display-only initial estimates, not measured 3D object poses.
    """
    shift = comparison.workcell_shift_left_front(T_base_cv, .08, .10)
    muhi = pixel_to_base_plane(T_base_cv, intrinsics, *muhi_pixel, .019) - shift
    cup = pixel_to_base_plane(T_base_cv, intrinsics, *cup_pixel, .055) - shift
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
    names = ("r2s_preview_table",) + tuple(
        f"r2s_table_leg_{x}{y}" for x in ("m", "p") for y in ("m", "p"))
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names]
    if any(index < 0 for index in ids):
        raise ValueError("Missing table geom; refusing partial translation")
    model.geom_pos[ids] += delta / 100.


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
                                 muhi_pixel, cup_pixel, muhi_yaw)
    model, camera = make_calibrated_model(
        extrinsics, intrinsics, args.urdf, layout,
        table_extra_left_m=args.table_extra_left_cm / 100.,
        table_extra_front_m=args.table_extra_front_cm / 100.)
    apply_table_base_offset(model, (args.table_base_x_cm, args.table_base_y_cm,
                                    args.table_base_z_cm))
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
            sub.add_argument("--cup-u", type=float, default=375., help="Clear container center x in the 640-pixel Real video panel")
            sub.add_argument("--cup-v", type=float, default=310., help="Clear container center y in 480-pixel Real video panel")
            sub.add_argument("--muhi-yaw-deg", type=float, default=45., help="Provisional Muhi yaw around base Z; change if axis disagrees")
            sub.add_argument("--table-extra-left-cm", type=float, default=4.,
                             help="Tabletop offset camera-left in cm (default: 4)")
            sub.add_argument("--table-extra-front-cm", type=float, default=5.,
                             help="Tabletop offset toward camera in cm (negative moves farther; default: 5)")
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
