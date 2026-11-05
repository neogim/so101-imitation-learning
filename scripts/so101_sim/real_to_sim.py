#!/usr/bin/env python3
"""Read-only SO-101 Real-to-Sim kinematic replay. NEVER accesses robot hardware.

The default workcell, camera poses and joint mapping are UNCALIBRATED examples.
This is NOT a physics-accurate grasp replay, a policy evaluation, or motor control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
STATE_NAMES = [f"{x}.pos" for x in (*JOINTS, "gripper")]
REPO = Path(__file__).resolve().parents[2]
DEFAULT_URDF = REPO / "data/urdf/so101/so101_new_calib.urdf"


def validate_config(config: dict) -> dict:
    if config.get("calibrated") is not True:
        # An explicit flag is deliberately required for any calibrated claim.
        config["calibrated"] = False
    for name in ("side", "wrist"):
        camera = config["cameras"][name]
        if len(camera["pos_m"]) != 3 or len(camera["quat_wxyz"]) != 4:
            raise ValueError(f"{name}: invalid camera pose")
        if not all(math.isfinite(float(x)) for x in (*camera["pos_m"], *camera["quat_wxyz"])):
            raise ValueError(f"{name}: nonfinite camera pose")
        if not 5 <= float(camera["fovy_deg"]) <= 160:
            raise ValueError(f"{name}: invalid fovy_deg")
    if set(config["joints"]["sign"]) != set(JOINTS) or set(config["joints"]["offset_deg"]) != set(JOINTS):
        raise ValueError("Joint calibration must provide all five arm joints")
    for name in JOINTS:
        if config["joints"]["sign"][name] not in (-1, 1):
            raise ValueError(f"{name}: sign must be +1 or -1")
        if not math.isfinite(float(config["joints"]["offset_deg"][name])):
            raise ValueError(f"{name}: nonfinite joint offset")
    return config


def load_config(path: Path) -> dict:
    return validate_config(json.loads(path.read_text(encoding="utf-8")))


def _episode_v3_metadata(root: Path, episode: int) -> dict:
    """Read one v3 per-episode metadata record; never infer boundaries from filenames."""
    import pyarrow.dataset as ds

    meta_dir = root / "meta/episodes"
    paths = sorted(meta_dir.glob("chunk-*/file-*.parquet"))
    if not paths:
        raise ValueError(f"Missing v3 episode metadata under {meta_dir}")
    dataset = ds.dataset([str(p) for p in paths], format="parquet")
    if "episode_index" not in dataset.schema.names:
        raise ValueError("v3 episode metadata lacks episode_index")
    import pyarrow.compute as pc
    subset = dataset.to_table(filter=pc.field("episode_index") == episode)
    if subset.num_rows != 1:
        raise ValueError(f"Expected one metadata row for episode {episode}; got {subset.num_rows}")
    meta = subset.to_pylist()[0]
    for key in ("data/chunk_index", "data/file_index", "length"):
        if key not in meta:
            raise ValueError(f"Missing required v3 episode metadata: {key}")
    return meta


def read_episode(root: Path, episode: int, stride: int, max_frames: int) -> dict:
    """Read one episode by metadata and episode_index; never alter the source data."""
    if episode < 0 or stride <= 0 or max_frames <= 0:
        raise ValueError("episode>=0, stride>0 and max_frames>0 are required")
    info = json.loads((root / "meta/info.json").read_text())
    features = info.get("features", {})
    if not {"observation.state", "action"}.issubset(features):
        raise ValueError("Missing state/action in dataset")
    for key in ("observation.state", "action"):
        if features[key].get("names") != STATE_NAMES:
            raise ValueError(f"Unexpected joint names/order in {key}: {features[key].get('names')}")
    import pyarrow.parquet as pq
    version = info.get("codebase_version", "")
    metadata = None
    if version.startswith("v3"):
        metadata = _episode_v3_metadata(root, episode)
        pattern = info.get("data_path")
        if not pattern:
            raise ValueError("Missing data_path in v3 info.json")
        relative = pattern.format(chunk_index=int(metadata["data/chunk_index"]),
                                  file_index=int(metadata["data/file_index"]))
        file = root / relative
        if not file.is_file():
            raise ValueError(f"v3 data shard missing: {file}")
        columns = ["episode_index", "frame_index", "observation.state", "action", "timestamp"]
        schema = pq.read_schema(file)
        if not set(columns).issubset(schema.names):
            raise ValueError(f"v3 data shard columns missing: {set(columns) - set(schema.names)}")
        import pyarrow.compute as pc
        tab = pq.read_table(file, columns=columns)
        tab = tab.filter(pc.equal(tab["episode_index"], episode))
        if tab.num_rows != int(metadata["length"]):
            raise ValueError(f"v3 episode length mismatch: metadata={metadata['length']}, data={tab.num_rows}")
        table_rows = sorted(tab.to_pylist(), key=lambda r: r["frame_index"])
        if [r["frame_index"] for r in table_rows] != list(range(len(table_rows))):
            raise ValueError("v3 frame_index is non-contiguous or duplicated")
    else:
        found = sorted(root.glob(f"data/chunk-*/episode_{episode:06d}.parquet"))
        if len(found) != 1:
            raise ValueError(f"Expected exactly one Parquet for episode {episode}; got {len(found)}")
        file = found[0]
        tab = pq.read_table(file, columns=["observation.state", "action", "timestamp"])
        table_rows = [{"frame_index": i, **row} for i, row in enumerate(tab.to_pylist())]
    rows = []
    for r in table_rows[::stride][:max_frames]:
        i = int(r["frame_index"])
        state = [float(v) for v in r["observation.state"]]
        action = [float(v) for v in r["action"]]
        if len(state) != 6 or len(action) != 6 or not all(map(math.isfinite, state + action)):
            raise ValueError(f"Nonfinite or wrong state/action shape at frame {i}")
        timestamp = float(r["timestamp"])
        if not math.isfinite(timestamp):
            raise ValueError(f"Nonfinite timestamp at frame {i}")
        rows.append({"frame_index": i, "timestamp": timestamp, "state": state, "action": action})
    if not rows:
        raise ValueError("Episode has zero rows")
    return {"source_parquet": str(file), "info": info, "episode": episode, "metadata": metadata, "rows": rows}


def joint_radians(row: list[float], config: dict) -> dict[str, float]:
    """Five joint axes only. The normalized gripper value is intentionally NOT guessed."""
    return {
        name: math.radians(
            config["joints"]["sign"][name] * row[i] + config["joints"]["offset_deg"][name]
        )
        for i, name in enumerate(JOINTS)
    }


def build_model(urdf: Path, config: dict):
    """MuJoCo MjSpec import; add illustrative STATIC props and 2 cameras.

    Static Muhi is NOT graspable; this is for camera/kinematic alignment only.
    """
    import mujoco
    spec = mujoco.MjSpec.from_file(str(urdf.resolve()))
    world = spec.worldbody
    table = config["objects"]["table"]
    world.add_geom(name="r2s_table", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=table["pos_m"], size=table["half_size_m"], rgba=[0.65, 0.45, 0.25, 1.0])
    muhi = config["objects"]["muhi"]
    world.add_geom(name="r2s_muhi_static", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                   pos=muhi["pos_m"], quat=muhi["quat_wxyz"], size=[muhi["radius_m"], muhi["half_length_m"], 0],
                   rgba=[0.92, 0.95, 0.98, 1], contype=0, conaffinity=0)
    cup = config["objects"]["cup"]
    # Schematic cup; collision disabled and transparency is only approximate.
    import numpy as np
    cx, cy, cz = cup["pos_m"]
    n = 16
    radius = float(cup["radius_m"])
    height = float(cup["height_m"])
    for i in range(n):
        theta = 2 * math.pi * i / n
        world.add_geom(name=f"r2s_cup_wall_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[cx + radius*math.cos(theta), cy + radius*math.sin(theta), cz + height/2],
                       quat=[math.cos(theta/2), 0, 0, math.sin(theta/2)],
                       size=[math.pi*radius/n, cup["wall_thickness_m"]/2, height/2],
                       rgba=[0.7, 0.86, 0.9, 0.25], contype=0, conaffinity=0)
    sc = config["cameras"]["side"]
    wc = config["cameras"]["wrist"]
    world.add_camera(name="r2s_side", pos=sc["pos_m"], quat=sc["quat_wxyz"], fovy=sc["fovy_deg"])
    parent = spec.body(config["wrist_parent_body"])
    if parent is None:
        raise ValueError(f"Missing wrist body in URDF: {config['wrist_parent_body']}")
    parent.add_camera(name="r2s_wrist", pos=wc["pos_m"], quat=wc["quat_wxyz"], fovy=wc["fovy_deg"])
    return spec.compile()


def set_pose(model, data, state: list[float], config: dict) -> None:
    import mujoco
    angles = joint_radians(state, config)
    data.qpos[:] = model.qpos0
    for name, angle in angles.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint not found in model: {name}")
        data.qpos[int(model.jnt_qposadr[jid])] = angle
    mujoco.mj_forward(model, data)


def diagnostics(model, data) -> dict:
    import mujoco
    outside = []
    for name in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        a = int(model.jnt_qposadr[jid])
        if bool(model.jnt_limited[jid]):
            lo, hi = (float(x) for x in model.jnt_range[jid])
            if not lo <= float(data.qpos[a]) <= hi:
                outside.append(name)
    return {"outside_mujoco_joint_limits": outside, "contact_count_model_only": int(data.ncon)}


def render_image(model, data, camera_name: str, width: int, height: int):
    import mujoco
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera_name)
        return renderer.render().copy()


def source_video_frame(root: Path, key: str, episode: int, frame_index: int,
                       info: dict | None = None, metadata: dict | None = None):
    """Decode one frame using v3 video shard offset, or v2 episode-local MP4."""
    if info is not None and info.get("codebase_version", "").startswith("v3"):
        if metadata is None:
            raise ValueError("v3 video decoding requires episode metadata")
        prefix = f"videos/{key}/"
        needed = (prefix + "chunk_index", prefix + "file_index",
                  prefix + "from_timestamp", prefix + "to_timestamp")
        if not all(k in metadata for k in needed):
            raise ValueError(f"Missing v3 video offsets for {key}")
        template = info.get("video_path")
        if not template:
            raise ValueError("Missing v3 video_path")
        video = root / template.format(video_key=key,
                                       chunk_index=int(metadata[needed[0]]),
                                       file_index=int(metadata[needed[1]]))
        fps = float(info["fps"])
        from_ts = float(metadata[needed[2]])
        to_ts = float(metadata[needed[3]])
        if frame_index < 0 or frame_index >= int(metadata["length"]):
            raise ValueError("Frame index outside requested episode")
        target = round(from_ts * fps) + frame_index
        if target >= round(to_ts * fps):
            raise ValueError("Frame index exceeds video episode end")
    else:
        videos = list(root.glob(f"videos/{key}/chunk-*/episode_{episode:06d}.mp4"))
        if len(videos) != 1:
            return None
        video = videos[0]
        target = frame_index
    if not video.is_file():
        raise ValueError(f"Video file missing: {video}")
    import av
    with av.open(str(video)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i == target:
                return frame.to_image()
    raise ValueError(f"Requested video frame {target} not decoded from {video}")


def run(args) -> dict:
    root = args.dataset_root.expanduser().resolve(strict=True)
    config = load_config(args.config.expanduser().resolve(strict=True))
    ep = read_episode(root, args.episode, args.stride, args.max_frames)
    import mujoco
    model = build_model(args.urdf, config)
    data = mujoco.MjData(model)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing output: {output}")
    if output == root or root in output.parents:
        raise ValueError("Output must be outside raw dataset root")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "mode": "OFFLINE_KINEMATIC_REPLAY_ONLY_NOT_PHYSICS_VALIDATION",
        "dataset": str(root), "episode": args.episode,
        "urdf": str(args.urdf.resolve()), "calibrated": config["calibrated"],
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "assumptions": ["state joint values are degrees; sign/offset are unverified until calibrated",
                        "gripper is NOT mapped (normalized 0..100 != URDF joint angle)",
                        "Muhi/cup positions and cameras are illustrative until measured",
                        "Muhi/cup are static, non-collidable geometry; no physical grasp evaluation",
                        "mj_forward sets poses directly; no actuator simulation or system identification",
                        "real timestamps are nominal; video capture skew unmeasured"],
        "frames": [],
    }
    # Save metadata first even when graphics backend is unavailable.
    from PIL import Image
    for row in ep["rows"]:
        set_pose(model, data, row["state"], config)
        item = {"frame_index": row["frame_index"], "timestamp": row["timestamp"],
                "gripper_normalized": row["state"][5],
                "command_state_delta_deg": [round(row["action"][i]-row["state"][i], 5) for i in range(5)],
                **diagnostics(model, data)}
        result["frames"].append(item)
        if args.render and (len(result["frames"]) - 1) % args.render_every == 0:
            for key, cam in (("side", "r2s_side"), ("wrist", "r2s_wrist")):
                image = render_image(model, data, cam, args.width, args.height)
                prefix = f"frame_{row['frame_index']:06d}_{key}"
                Image.fromarray(image).save(output / (prefix + "_sim.png"))
                if args.real_images:
                    real = source_video_frame(root, "observation.images." + key,
                                              args.episode, row["frame_index"], ep["info"], ep["metadata"])
                    if real is not None:
                        real = real.resize((args.width, args.height))
                        real.save(output / (prefix + "_real.png"))
                        from PIL import ImageOps
                        comparison = Image.new("RGB", (args.width*2, args.height))
                        comparison.paste(real, (0, 0))
                        comparison.paste(Image.fromarray(image), (args.width, 0))
                        comparison.save(output / (prefix + "_comparison.png"))
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--stride", type=int, default=20, help="Process every Nth frame")
    p.add_argument("--max-frames", type=int, default=30)
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("workcell.example.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--render", action="store_true", help="Save simulated camera frames")
    p.add_argument("--real-images", action="store_true", help="Decode existing Side/Wrist MP4 frames")
    p.add_argument("--render-every", type=int, default=1)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    a = p.parse_args()
    if a.render_every <= 0 or a.width <= 0 or a.height <= 0:
        p.error("Invalid rendering dimensions/step")
    if a.real_images and not a.render:
        p.error("--real-images requires --render")
    try:
        result = run(a)
    except (OSError, ValueError, RuntimeError) as exc:
        p.error(str(exc))
    print(json.dumps({"frames": len(result["frames"]), "calibrated": result["calibrated"],
                      "output": str(a.output), "mode": result["mode"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
