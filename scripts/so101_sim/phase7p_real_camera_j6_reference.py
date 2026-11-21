#!/usr/bin/env python3
"""Compare Episode 015 real Side frames with FK + provisional J6 aperture and camera.

No IK, actuation, mj_step or real hardware.  Camera is a view hypothesis, not calibration.
The LeRobot normalized gripper to URDF angle mapping is provisional and is never
allowed to be described as measured or verified.  Real frames and sim FK share timestamps.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TIMES = (14.5, 15.0, 15.5, 16.0, 17.0, 18.0)


def load_rows(path):
    rows = []
    with Path(path).open(newline="", encoding="utf8") as fp:
        for raw in csv.DictReader(fp):
            r = {"t": float(raw["timestamp"]), "frame": int(raw["frame_index"]),
                 "state": np.asarray(json.loads(raw["observation.state"]), dtype=float),
                 "action": np.asarray(json.loads(raw["action"]), dtype=float)}
            if r["state"].shape != (6,) or r["action"].shape != (6,):
                raise ValueError("Expected six-dimensional state/action")
            if not np.isfinite(np.r_[r["t"], r["state"], r["action"]]).all():
                raise ValueError("Non-finite observation or action")
            rows.append(r)
    if not rows:
        raise ValueError("No rows in episode")
    return rows


def state_to_urdf_deg(state, sign, offset):
    a = np.asarray(state, dtype=float)
    s = np.asarray(sign, dtype=float)
    o = np.asarray(offset, dtype=float)
    if a.shape != (5,) or s.shape != (5,) or o.shape != (5,) or not np.all(np.isin(s, (-1, 1))):
        raise ValueError("Five joint values, signs, offsets required")
    return a*s + o


def j6_map(norm, closed_deg, open_deg):
    # Video evidence supports lower normalized values = closing direction.
    # It does NOT establish zero, scale, hysteresis or physical aperture.
    if not (0 <= norm <= 100 and closed_deg < open_deg):
        raise ValueError("J6 0..100 or mapping direction invalid")
    return closed_deg + (open_deg-closed_deg)*norm/100


def real_frames(video, frame_idxs):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("Could not open recorded Side video: "+str(video))
    frames = {}
    try:
        for i in sorted(set(frame_idxs)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError("Could not decode recorded frame "+str(i))
            frames[i] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((640, 480))
    finally:
        cap.release()
    return frames


def camera_model(azimuth, elevation, distance, lookat):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(lookat, dtype=float)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    cam.distance = float(distance)
    return cam


def annotate(im, line1, line2):
    tile = Image.new("RGB", (640, 528), "#17212b")
    tile.paste(im, (0, 48))
    draw = ImageDraw.Draw(tile)
    draw.text((8, 8), line1, fill="white")
    draw.text((8, 27), line2, fill="#d4e2ed")
    return tile


def montage(images, columns):
    rows = math.ceil(len(images)/columns)
    canvas = Image.new("RGB", (640*columns, 528*rows), "#17212b")
    for i, tile in enumerate(images):
        canvas.paste(tile, ((i%columns)*640, (i//columns)*528))
    return canvas


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--side", type=Path, required=True)
    p.add_argument("--urdf", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--azimuth", type=float, default=90)
    p.add_argument("--elevation", type=float, default=-25)
    p.add_argument("--distance", type=float, default=.57)
    p.add_argument("--lookat", type=float, nargs=3, default=[.12, 0., .10])
    p.add_argument("--j6-closed-deg", type=float, default=-10)
    p.add_argument("--j6-open-deg", type=float, default=100)
    p.add_argument("--camera-grid", action="store_true")
    args = p.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    rows = load_rows(args.csv)
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    sign, offset = mapping["sign"], mapping["offset_deg"]
    sample = [min(rows, key=lambda r: abs(r["t"]-t)) for t in TIMES]
    recorded = real_frames(args.side, [x["frame"] for x in sample])
    spec = mujoco.MjSpec.from_file(str(args.urdf.resolve()))
    model = spec.compile()
    d = mujoco.MjData(model)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
    if min(joints) < 0:
        raise ValueError("Expected named SO-101 joints missing")
    adr = [int(model.jnt_qposadr[j]) for j in joints]
    low, high = model.jnt_range[joints[5]].tolist()
    if not(low <= math.radians(args.j6_closed_deg) < math.radians(args.j6_open_deg) <= high):
        raise ValueError("Provisional J6 endpoint mapping outside URDF limits")
    results = []
    matches = []
    renderer = mujoco.Renderer(model, width=640, height=480)
    try:
        for record in sample:
            arm_deg = state_to_urdf_deg(record["state"][:5], sign, offset)
            j6_deg = j6_map(record["state"][5], args.j6_closed_deg, args.j6_open_deg)
            qdeg = np.r_[arm_deg, j6_deg]
            bounds = np.rad2deg(model.jnt_range[joints])
            failures = [name for name, val, lr in zip(JOINTS, qdeg, bounds)
                        if not (lr[0] - 1e-4 <= val <= lr[1] + 1e-4)]
            diagnostic = dict(time_s=float(record["t"]), frame=int(record["frame"]),
                              joint_degrees=qdeg.tolist(),
                              j6_recorded_normalized=float(record["state"][5]),
                              j6_command_normalized=float(record["action"][5]),
                              urdf_limit_errors=failures)
            results.append(diagnostic)
            if failures:
                raise ValueError(f"URDF limit failure at {record['t']:.2f}s: {failures}")
            d.qpos[:] = model.qpos0
            d.qpos[adr] = np.deg2rad(qdeg)
            mujoco.mj_forward(model, d)  # FK only, no physical time integration.
            camera = camera_model(args.azimuth, args.elevation, args.distance, args.lookat)
            renderer.update_scene(d, camera=camera)
            sim = Image.fromarray(renderer.render().copy())
            real = recorded[record["frame"]]
            t = record["t"]
            matches.extend([annotate(real, f"REAL Side | t={t:.2f}s", "Original 320x240 scaled to 640x480"),
                            annotate(sim, f"MuJoCo FK | t={t:.2f}s", f"J6 hypothesis {j6_deg:.1f} deg | az={args.azimuth:.0f}, el={args.elevation:.0f}")])
        montage(matches, 2).save(args.out/"real_vs_sim_side_synchronized.png")
        if args.camera_grid:
            ref = min(sample, key=lambda r: abs(r["t"]-15.5))
            arm = state_to_urdf_deg(ref["state"][:5], sign, offset)
            j6 = j6_map(ref["state"][5], args.j6_closed_deg, args.j6_open_deg)
            d.qpos[:] = model.qpos0
            d.qpos[adr] = np.deg2rad(np.r_[arm, j6])
            mujoco.mj_forward(model, d)
            grid = []
            for elevation in (args.elevation-15, args.elevation, args.elevation+15):
                for azimuth in (args.azimuth-30, args.azimuth, args.azimuth+30):
                    renderer.update_scene(d, camera=camera_model(azimuth,elevation,args.distance,args.lookat))
                    pic = Image.fromarray(renderer.render().copy())
                    grid.append(annotate(pic, f"AZ={azimuth:.0f} | EL={elevation:.0f}", f"t={ref['t']:.2f}s | same arm pose and J6"))
            montage(grid, 3).save(args.out/"camera_3x3_view_sweep.png")
    finally:
        renderer.close()
    with (args.out/"j6_reference_trace.csv").open("w",newline="",encoding="utf8") as fp:
        fields=["time_s","frame","j6_recorded_normalized","j6_command_normalized","j6_provisional_urdf_deg"]
        writer=csv.DictWriter(fp,fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(dict(time_s=r["time_s"],frame=r["frame"],
                j6_recorded_normalized=r["j6_recorded_normalized"],
                j6_command_normalized=r["j6_command_normalized"],
                j6_provisional_urdf_deg=r["joint_degrees"][5]))
    info={"status":"FK_REFERENCE_ONLY_J6_AND_CAMERA_NOT_CALIBRATED",
          "recorded_episode":15,"reference_times_s":list(TIMES),
          "camera_view_hypothesis":dict(azimuth_deg=args.azimuth,elevation_deg=args.elevation,
              distance_m=args.distance,lookat_m=args.lookat),
          "joint_mapping_hypothesis":mapping,
          "j6_mapping_hypothesis":dict(closed_deg_at_norm0=args.j6_closed_deg,
              open_deg_at_norm100=args.j6_open_deg),
          "results":results,"mj_step_executed":False,
          "note":"Comparison of camera/kinematic hypotheses only. No object collision, friction, or grasp success measured."}
    (args.out/"summary.json").write_text(json.dumps(info,indent=2,ensure_ascii=False))
    print(json.dumps(dict(status=info["status"],out=str(args.out),frames=len(results)),ensure_ascii=False))
    return info

if __name__=="__main__":
    main()
