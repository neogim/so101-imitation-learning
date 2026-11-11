#!/usr/bin/env python3
"""Two-waypoint SO-101 kinematic demo: approach Muhi, then move above a box.

No grasp, lifting an object, collision avoidance, dynamics, or hardware motion.
Muhi and the open box are VISUAL PROXIES placed at FK-reachable targets.
DLS iteratively controls the 5 arm joints; goal-generating joint angles are not
passed to DLS. Save recorded trajectory as an MP4 or replay in a MuJoCo window.
On macOS use mjpython to open a GUI; regular python works for --validate-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

import real_to_sim as r2s
from learn_reach_kinematic import KinematicReachEnv, jacobian_reach_action

JOINT_COLUMNS = tuple(f"joint_{i}" for i in range(1, 6))
STAGES = ("muhi", "box")
STAGE_COLORS = {
    "muhi": np.array([0.95, 0.35, 0.18, 1.0]),
    "box": np.array([0.12, 0.70, 0.36, 1.0]),
}


def _tcp(env):
    return np.asarray(env.data.xpos[env.gripper_id], dtype=float).copy()


def _fk_candidate(env, q):
    previous = env.data.qpos[env.qids].copy()
    env.data.qpos[env.qids] = q
    mujoco.mj_forward(env.model, env.data)
    result = _tcp(env)
    env.data.qpos[env.qids] = previous
    mujoco.mj_forward(env.model, env.data)
    return result


def _run_segment(env, target, stage, max_steps):
    """Record one target reach without resetting joints between stages."""
    env.target_tcp = np.asarray(target, float).copy()
    env.steps = 0
    rows = []

    def capture(step):
        current = _tcp(env)
        row = {
            "stage": stage, "local_step": step,
            "tcp_x": float(current[0]), "tcp_y": float(current[1]),
            "tcp_z": float(current[2]),
            "target_x": float(target[0]), "target_y": float(target[1]),
            "target_z": float(target[2]),
            "tcp_error_mm": float(np.linalg.norm(current - target)*1000),
        }
        row.update({name: float(q) for name, q in
                    zip(JOINT_COLUMNS, env.data.qpos[env.qids])})
        return row

    rows.append(capture(0))
    success = rows[0]["tcp_error_mm"] <= env.success_radius_m*1000
    if not success:
        for step in range(1, max_steps + 1):
            action = jacobian_reach_action(env)
            _, _, done, timeout, _ = env.step(action)
            rows.append(capture(step))
            if done or timeout:
                success = bool(done)
                break
    return rows, success


def calculate_sequence(args):
    """Deterministically find two separated, DLS-solvable FK targets."""
    env = KinematicReachEnv(
        seed=args.seed, max_steps=args.max_steps, reward_scale=10.0,
        action_delta_rad=0.04, goal_joint_delta_rad=0.70,
        success_radius_m=args.success_mm/1000)
    rng = np.random.default_rng(args.seed)
    try:
        for trial in range(1, args.max_trials + 1):
            env.reset(seed=args.seed + trial)
            start_q = env.data.qpos[env.qids].copy()
            start_tcp = _tcp(env)
            low = np.maximum(env.qmin + 0.06, start_q - args.sample_delta_rad)
            high = np.minimum(env.qmax - 0.06, start_q + args.sample_delta_rad)
            a, b = rng.uniform(low, high), rng.uniform(low, high)
            muhi, box = _fk_candidate(env, a), _fk_candidate(env, b)
            start_muhi = np.linalg.norm(muhi - start_tcp)
            muhi_box = np.linalg.norm(box - muhi)
            planar_sep = np.linalg.norm(box[:2] - muhi[:2])
            if (min(muhi[2], box[2]) < args.min_goal_height_m or
                    start_muhi < args.min_start_muhi_mm / 1000 or
                    muhi_box < args.min_muhi_box_mm / 1000 or
                    planar_sep < args.min_xy_sep_mm / 1000):
                continue
            # DLS is evaluated on the same sequential joint trajectory; it
            # does not read a or b (the goal-generating joint configurations).
            env.data.qpos[env.qids] = start_q
            mujoco.mj_forward(env.model, env.data)
            first, first_ok = _run_segment(env, muhi, "muhi", args.max_steps)
            if not first_ok:
                continue
            second, second_ok = _run_segment(env, box, "box", args.max_steps)
            if not second_ok:
                continue
            frames = first + second[1:]   # no duplicate transition frame
            for i, row in enumerate(frames):
                row["frame"] = i
            summary = {
                "status": "KINEMATIC_ONLY_VISUAL_PROPS_NO_GRASP_NO_CONTACT",
                "policy": "Jacobian DLS",
                "seed": args.seed, "sample_trial": trial,
                "success_radius_mm": args.success_mm,
                "min_start_muhi_mm": args.min_start_muhi_mm,
                "min_muhi_box_mm": args.min_muhi_box_mm,
                "actual_start_to_muhi_mm": float(start_muhi*1000),
                "actual_muhi_to_box_mm": float(muhi_box*1000),
                "muhi_goal_xyz_m": muhi.tolist(),
                "box_goal_xyz_m": box.tolist(),
                "initial_joint_rad": start_q.tolist(),
                "muhi_reach_success": first_ok, "box_reach_success": second_ok,
                "muhi_steps": len(first)-1,
                "box_steps": len(second)-1,
                "muhi_final_error_mm": first[-1]["tcp_error_mm"],
                "box_final_error_mm": second[-1]["tcp_error_mm"],
                "frames": len(frames),
                "note": "The Muhi and box are schematic visual proxies positioned below FK-generated goals. "
                        "They are not a measured real-world arrangement or collision-checked objects.",
            }
            return frames, summary
        raise RuntimeError(
            f"No two-stage DLS-solvable pair found in {args.max_trials} trials. "
            "Try lowering --min-start-muhi-mm / --min-muhi-box-mm or "
            "increasing --max-trials. No result has been fabricated.")
    finally:
        env.close()


def save_data(out, frames, summary):
    if out.exists():
        raise FileExistsError(f"Output directory already exists: {out}")
    out.mkdir(parents=True)
    with (out / "rollout.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(frames[0].keys()))
        writer.writeheader()
        writer.writerows(frames)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    (out / "report.md").write_text(
        "# SO-101 Muhi to box two-waypoint Reach (kinematics only)\n\n"
        "- Muhi and open box are illustrative geometry only. No grasp or release.\n"
        "- Both TCP targets are FK-generated and separately reached by DLS.\n"
        f"- Start to Muhi: {summary['actual_start_to_muhi_mm']:.1f} mm\n"
        f"- Muhi to box: {summary['actual_muhi_to_box_mm']:.1f} mm\n"
        f"- Muhi: {summary['muhi_steps']} steps, {summary['muhi_final_error_mm']:.2f} mm TCP error\n"
        f"- Box: {summary['box_steps']} steps, {summary['box_final_error_mm']:.2f} mm TCP error\n"
        "- Reach only; this does not simulate pick, lift, transfer of object, "
        "collision avoidance or real-world hardware.\n", encoding="utf-8")


def build_visual_scene(summary):
    root = mujoco.MjSpec()
    root.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0, 0, -0.03], size=[0.9, 0.9, 0.01],
        rgba=[0.84, 0.86, 0.88, 1], contype=0, conaffinity=0)
    robot = mujoco.MjSpec.from_file(str(r2s.DEFAULT_URDF.resolve()))
    robot.compiler.discardvisual = False
    frame = root.worldbody.add_frame(name="demo_origin", pos=[0, 0, 0])
    root.attach(robot, prefix="demo_", frame=frame)

    muhi = np.array(summary["muhi_goal_xyz_m"])
    box = np.array(summary["box_goal_xyz_m"])
    # Simplified white Muhi tube with red cap; place its cap BELOW goal TCP.
    root.worldbody.add_geom(
        name="muhi_body_visual", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=muhi + [0, 0, -0.067], size=[0.018, 0.018, 0.042],
        rgba=[0.96, 0.96, 0.94, 1], contype=0, conaffinity=0)
    root.worldbody.add_geom(
        name="muhi_cap_visual", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=muhi + [0, 0, -0.020], size=[0.021, 0.021, 0.008],
        rgba=[0.95, 0.24, 0.23, 1], contype=0, conaffinity=0)
    root.worldbody.add_geom(
        name="muhi_goal_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=muhi, size=[0.008, 0, 0], rgba=STAGE_COLORS["muhi"],
        contype=0, conaffinity=0)

    # Open-top tray: bottom + four thin walls, goal is ABOVE the tray.
    x, y, z = box
    parts = [
        ("bottom", [x, y, z-0.096], [0.085, 0.070, 0.006]),
        ("wall_x_plus", [x+0.083, y, z-0.064], [0.005, 0.070, 0.030]),
        ("wall_x_minus", [x-0.083, y, z-0.064], [0.005, 0.070, 0.030]),
        ("wall_y_plus", [x, y+0.068, z-0.064], [0.085, 0.005, 0.030]),
        ("wall_y_minus", [x, y-0.068, z-0.064], [0.085, 0.005, 0.030]),
    ]
    for name, pos, size in parts:
        root.worldbody.add_geom(
            name=f"box_{name}_visual", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos, size=size, rgba=[0.19, 0.52, 0.79, 0.75],
            contype=0, conaffinity=0)
    root.worldbody.add_geom(
        name="box_goal_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=box, size=[0.008, 0, 0], rgba=STAGE_COLORS["box"],
        contype=0, conaffinity=0)

    model = root.compile()
    data = mujoco.MjData(model)
    addresses = []
    for joint in r2s.JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                "demo_" + joint)
        if jid < 0:
            raise ValueError(f"Joint {joint} not found in visual scene")
        addresses.append(int(model.jnt_qposadr[jid]))
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                               "demo_gripper_link")
    if tcp_id < 0:
        raise ValueError("Visual scene TCP body missing")
    return model, data, np.asarray(addresses, int), tcp_id


def set_frame(model, data, qpos_ids, frame):
    data.qpos[qpos_ids] = [frame[name] for name in JOINT_COLUMNS]
    mujoco.mj_forward(model, data)


def validate_visual_fk(model, data, qpos_ids, tcp_id, frames):
    error = 0.0
    for row in frames:
        set_frame(model, data, qpos_ids, row)
        actual = np.asarray(data.xpos[tcp_id])
        expected = np.array([row["tcp_x"], row["tcp_y"], row["tcp_z"]])
        error = max(error, float(np.linalg.norm(actual - expected)))
    if error > 1e-8:
        raise ValueError(f"Viewer FK mismatch: {error:.3g}m")
    set_frame(model, data, qpos_ids, frames[0])
    print(f"Viewer TCP reconstruction max error: {error:.3g} m")


def camera_for(summary):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    p = (np.array(summary["muhi_goal_xyz_m"]) +
         np.array(summary["box_goal_xyz_m"])) / 2
    camera.lookat[:] = p
    camera.distance = 0.95
    camera.azimuth = 155
    camera.elevation = -28
    return camera


def add_path_to_scene(scene, rows, frame):
    # Current path only, with distinct colors for Muhi approach and box approach.
    for a, b in zip(rows[:frame], rows[1:frame+1]):
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("Scene cannot accommodate all trajectory segments")
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE,
                            np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                            STAGE_COLORS[b["stage"]])
        start = np.array([a[f"tcp_{i}"] for i in "xyz"])
        end = np.array([b[f"tcp_{i}"] for i in "xyz"])
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.5, start, end)
        scene.ngeom += 1


def write_mp4(model, data, qpos_ids, frames, summary, destination,
              fps=30, steps_per_second=8, hold_seconds=2):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("MP4 export needs pillow (python -m pip install pillow)") from exc
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("MP4 export needs ffmpeg (macOS: brew install ffmpeg)")
    if destination.exists():
        raise FileExistsError(f"MP4 already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1280, 720
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s:v", f"{width}x{height}", "-r", str(fps),
               "-i", "-", "-an", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(destination)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            try:
                font = ImageFont.truetype(
                    "/System/Library/Fonts/Supplemental/Arial.ttf", 24)
            except OSError:
                font = ImageFont.load_default()
            n_motion = math.ceil((len(frames)-1) / steps_per_second * fps) + 1
            n_hold = int(round(hold_seconds * fps))
            for index in range(n_motion + n_hold):
                step = min(len(frames)-1, int(index*steps_per_second/fps))
                row = frames[step]
                set_frame(model, data, qpos_ids, row)
                renderer.update_scene(data, camera=camera_for(summary))
                add_path_to_scene(renderer.scene, frames, step)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 1280, 75), fill=(18, 28, 39))
                stage_label = "1: Move to Muhi" if row["stage"] == "muhi" else "2: Move above box"
                draw.text((20, 10), f"SO-101 Jacobian DLS  |  {stage_label}",
                          font=font, fill="white")
                draw.text((20, 42), f"step {step}/{len(frames)-1}  TCP error: {row['tcp_error_mm']:.1f} mm",
                          font=font, fill="white")
                draw.rectangle((0, 680, 1280, 720), fill=(18, 28, 39))
                draw.text((20, 686), "Visual proxies only | no grasp / no contact / no object transport",
                          font=font, fill="white")
                assert proc.stdin is not None
                proc.stdin.write(np.asarray(image, np.uint8).tobytes())
        assert proc.stdin is not None
        proc.stdin.close()
        proc.stdin = None
        err = proc.communicate()[1]
        if proc.returncode:
            raise RuntimeError(f"ffmpeg failed: {err.decode(errors='replace')}")
    except Exception:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
        if destination.exists():
            destination.unlink()
        raise
    print(f"Saved MP4: {destination} (1280x720, {fps}fps)")


def replay_viewer(model, data, qpos_ids, frames, summary, fps):
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data,
                                     show_left_ui=False,
                                     show_right_ui=False) as viewer:
        camera = camera_for(summary)
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.distance = camera.distance
        viewer.cam.azimuth = camera.azimuth
        viewer.cam.elevation = camera.elevation
        with viewer.lock():
            viewer.user_scn.ngeom = 0
            add_path_to_scene(viewer.user_scn, frames, len(frames)-1)
        print("MuJoCo viewer: watch Muhi → box approach; Esc to close.")
        print("Objects are VISUAL ONLY. No grasp or physics.")
        for row in frames:
            if not viewer.is_running():
                break
            with viewer.lock():
                set_frame(model, data, qpos_ids, row)
            viewer.set_texts((
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "DLS: Muhi (red) -> Box (green)",
                f"{row['stage']} | TCP error {row['tcp_error_mm']:.1f}mm",
            ))
            viewer.sync()
            time.sleep(1/fps)
        while viewer.is_running():
            viewer.sync()
            time.sleep(1/60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--success-mm", type=float, default=10)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--sample-delta-rad", type=float, default=0.85)
    parser.add_argument("--min-start-muhi-mm", type=float, default=120)
    parser.add_argument("--min-muhi-box-mm", type=float, default=180)
    parser.add_argument("--min-xy-sep-mm", type=float, default=100)
    parser.add_argument("--min-goal-height-m", type=float, default=0.10)
    parser.add_argument("--max-trials", type=int, default=300)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--save-video", type=Path, default=None)
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--fps", type=float, default=8)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.success_mm <= 0 or args.max_steps <= 0 or args.fps <= 0 or args.video_fps <= 0:
        parser.error("Success radius, max steps and fps must be positive")
    if not 0 < args.sample_delta_rad <= 1:
        parser.error("Sample delta must be within (0,1] radians")
    if args.min_start_muhi_mm < 0 or args.min_muhi_box_mm < 0 or args.min_xy_sep_mm < 0:
        parser.error("Distances must be nonnegative")
    if args.max_trials <= 0 or args.hold_seconds < 0:
        parser.error("Invalid max trials or hold duration")
    frames, summary = calculate_sequence(args)
    model, data, addresses, tip = build_visual_scene(summary)
    validate_visual_fk(model, data, addresses, tip, frames)
    save_data(args.out, frames, summary)
    print("Muhi reached:", summary["muhi_reach_success"],
          f"in {summary['muhi_steps']} steps")
    print("Box reached:", summary["box_reach_success"],
          f"in {summary['box_steps']} steps")
    print("Distances: start -> Muhi =",
          f"{summary['actual_start_to_muhi_mm']:.1f}mm, Muhi -> box =",
          f"{summary['actual_muhi_to_box_mm']:.1f}mm")
    if args.save_video:
        write_mp4(model, data, addresses, frames, summary, args.save_video,
                  args.video_fps, args.fps, args.hold_seconds)
    if args.show_viewer or (not args.validate_only and not args.save_video):
        replay_viewer(model, data, addresses, frames, summary, args.fps)


if __name__ == "__main__":
    main()
