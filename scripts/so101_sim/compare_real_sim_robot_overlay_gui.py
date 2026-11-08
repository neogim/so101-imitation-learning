#!/usr/bin/env python3
"""Offline SO-101 joint-offset inspection against recorded Side camera frames.

Real / Sim / 50% overlay of episode frames 0, 24, 40 and 52. Changes only
five *temporary* joint offset parameters in MuJoCo; never actuates hardware.
Saved JSON is an inspection candidate, not a calibrated robot model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import align_cup_gui as cup_gui
import compare_real_sim_motion as movie
import compare_id1_eye_to_hand as geometry
import real_to_sim as r2s

DEFAULT_FRAMES = (0, 24, 40, 52)


def joint_config(offsets):
    """Use the existing logged-state convention, leaving signs unchanged."""
    arr = np.asarray(offsets, dtype=float)
    if arr.shape != (len(r2s.JOINTS),) or not np.isfinite(arr).all():
        raise ValueError("Expected five finite joint offsets in degrees")
    if np.max(np.abs(arr)) > 45.:
        raise ValueError("Offsets above 45 degrees require independent calibration")
    return {
        "joints": {
            "sign": {name: 1 for name in r2s.JOINTS},
            "offset_deg": {name: float(value) for name, value in zip(r2s.JOINTS, arr)},
        }
    }


def selected_frame_rows(episode_data, frame_indices):
    by_frame = {int(row["frame_index"]): row for row in episode_data["rows"]}
    frames = [int(f) for f in frame_indices]
    if len(set(frames)) != len(frames) or not frames:
        raise ValueError("Frame indices must be nonempty and unique")
    missing = sorted(set(frames) - set(by_frame))
    if missing:
        raise ValueError(f"Missing episode frames: {missing}")
    return {i: by_frame[i] for i in frames}


def robot_pose_payload(offsets, episode, object_layout_json, cup_pose_json, table_flags,
                       frame_index=0, overlay_alpha=.5):
    cfg = joint_config(offsets)
    if not 0 <= overlay_alpha <= 1:
        raise ValueError("Overlay alpha must be between 0 and 1")
    if len(table_flags) != 6 or not np.isfinite(table_flags).all():
        raise ValueError("Expected six finite table flags")
    keys = ("table_extra_left_cm", "table_extra_front_cm",
            "table_base_x_cm", "table_base_y_cm", "table_base_z_cm",
            "table_yaw_delta_deg")
    return {
        "schema": "so101_robot_joint_offset_candidate_v1",
        "status": "VISUAL_INSPECTION_CANDIDATE_NOT_CALIBRATED",
        "episode": int(episode),
        "source_frame_index": int(frame_index),
        "overlay_alpha": round(float(overlay_alpha), 3),
        "time_shift_frames": 0,
        "joint_offset_deg": cfg["joints"]["offset_deg"],
        "joint_sign": cfg["joints"]["sign"],
        "object_layout_json": str(object_layout_json),
        "cup_pose_json": str(cup_pose_json),
        "table": {key: float(val) for key, val in zip(keys, table_flags)},
        "note": "Apply identical offsets to all four frames. Do not modify camera, base, table, objects or hardware.",
    }


def read_robot_pose(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if result.get("schema") != "so101_robot_joint_offset_candidate_v1":
        raise ValueError("Wrong SO-101 robot offset JSON schema")
    names = result.get("joint_offset_deg", {})
    if set(names) != set(r2s.JOINTS):
        raise ValueError("Joint offset JSON missing or adding joints")
    joint_config([names[name] for name in r2s.JOINTS])
    if result.get("joint_sign") != {name: 1 for name in r2s.JOINTS}:
        raise ValueError("Unexpected joint signs")
    return result


def matches_scene(saved, args):
    """Fail rather than overlay a candidate onto a different workcell."""
    if saved.get("episode") != args.episode:
        raise ValueError("Robot-offset JSON belongs to another episode")
    if Path(saved.get("object_layout_json", "")) != args.object_layout_json:
        raise ValueError("Robot-offset JSON references different object layout")
    expected_cup = str(args.cup_pose_json) if args.cup_pose_json else "None"
    if saved.get("cup_pose_json") != expected_cup:
        raise ValueError("Robot-offset JSON references different cup pose")
    fields = {
        "table_extra_left_cm": args.table_extra_left_cm,
        "table_extra_front_cm": args.table_extra_front_cm,
        "table_base_x_cm": args.table_base_x_cm,
        "table_base_y_cm": args.table_base_y_cm,
        "table_base_z_cm": args.table_base_z_cm,
        "table_yaw_delta_deg": args.table_yaw_delta_deg,
    }
    values = saved.get("table", {})
    for key, wanted in fields.items():
        if not np.isclose(float(values.get(key, float("nan"))), wanted):
            raise ValueError(f"Robot-offset JSON uses a different {key}")


def run(args):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, RadioButtons, Slider
    import mujoco

    if args.output.exists() or args.output.with_suffix(".png").exists():
        raise ValueError("Output exists, use a new filename")
    if args.dataset_root.resolve() in args.output.resolve().parents:
        raise ValueError("Output may not overwrite the source dataset")
    frames = tuple(args.frames)
    ep = r2s.read_episode(args.dataset_root.resolve(strict=True), args.episode,
                          1, max(frames) + 1)
    rows = selected_frame_rows(ep, frames)
    side, video_start = movie.side_video_source(args.dataset_root, ep)
    videos = movie.read_frames_once(side, [video_start + i for i in frames])
    reals = {
        frame: np.asarray(videos[video_start + frame].convert("RGB").resize(
            (640, 480), Image.Resampling.BILINEAR))
        for frame in frames
    }
    intr = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    extr = json.loads(args.extrinsics.read_text(encoding="utf-8"))
    if (intr.get("width"), intr.get("height")) != (640, 480):
        raise ValueError("Side camera intrinsics must be 640x480")
    muhi, cup, yaw = movie.read_object_targets(args.object_layout_json)
    layout = movie.video_object_layout(extr["T_base_camera"], intr, muhi, cup, yaw)
    model, camera = movie.make_calibrated_model(
        extr, intr, args.urdf, layout,
        table_extra_left_m=args.table_extra_left_cm / 100.,
        table_extra_front_m=args.table_extra_front_cm / 100.)
    table_ids = movie.table_geom_ids(model)
    movie.set_table_pose_delta(
        model, model.geom_pos[table_ids].copy(), model.geom_quat[table_ids].copy(),
        np.array([args.table_base_x_cm, args.table_base_y_cm,
                  args.table_base_z_cm]) / 100.,
        args.table_yaw_delta_deg)
    if args.cup_pose_json is not None:
        cup_data = cup_gui.read_cup_payload(args.cup_pose_json)
        if cup_data.get("episode") != args.episode:
            raise ValueError("Cup pose belongs to another episode")
        if Path(cup_data.get("object_layout_json", "")) != args.object_layout_json:
            raise ValueError("Cup pose references different object annotations")
        expected = {
            "table_extra_left_cm": args.table_extra_left_cm,
            "table_extra_front_cm": args.table_extra_front_cm,
            "table_base_x_cm": args.table_base_x_cm,
            "table_base_y_cm": args.table_base_y_cm,
            "table_base_z_cm": args.table_base_z_cm,
            "table_yaw_delta_deg": args.table_yaw_delta_deg,
        }
        for key, val in expected.items():
            if not np.isclose(float(cup_data.get(key, float("nan"))), val):
                raise ValueError(f"Cup pose was saved under different {key}")
        cup_ids = cup_gui.cup_geom_ids(model)
        cup_gui.set_cup_dimensions(
            model, cup_ids, model.geom_pos[cup_ids].copy(),
            model.geom_size[cup_ids].copy(),
            diameter_m=geometry.CUP_DIAMETER_M,
            height_m=geometry.CUP_HEIGHT_M,
            xyz_m=cup_data["cup_delta_base_m"])
    data = mujoco.MjData(model)

    selected = [frames[0]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 8.8))
    fig.subplots_adjust(top=.89, bottom=.42, left=.025, right=.99, wspace=.04)
    for ax, label in zip(axes, ("REAL", "SIM (MuJoCo FK)", "OVERLAY")):
        ax.set_title(label)
        ax.axis("off")
    r_artist = axes[0].imshow(reals[selected[0]])
    blank = np.zeros_like(reals[selected[0]])
    sim_artist = axes[1].imshow(blank)
    axes[2].imshow(reals[selected[0]])
    overlay_real_artist = axes[2].images[0]
    overlay_artist = axes[2].imshow(blank, alpha=.5)

    sliders = {}
    for index, name in enumerate(r2s.JOINTS):
        ax = fig.add_axes([.20, .34 - index * .055, .65, .026])
        sliders[name] = Slider(ax, name + " offset (deg)", -25., 25.,
                               valinit=0., valstep=.25)
    alpha_ax = fig.add_axes([.20, .052, .65, .022])
    alpha_slider = Slider(alpha_ax, "overlay alpha", 0., 1., valinit=.5)
    frame_ax = fig.add_axes([.025, .12, .15, .22])
    frame_radio = RadioButtons(frame_ax, [f"frame {i}" for i in frames], active=0)
    save_ax = fig.add_axes([.86, .012, .12, .042])
    save_button = Button(save_ax, "SAVE JSON + PNG")
    fig.suptitle("SO-101 joint offsets (diagnostic only): keep tabletop and objects fixed",
                 fontsize=12)
    status = fig.text(.025, .013, "Adjust J1–J5; then check other frames before saving.",
                      fontsize=9)
    renderer = mujoco.Renderer(model, width=640, height=480)

    def current_offsets():
        return np.array([sliders[name].val for name in r2s.JOINTS])

    def render(frame):
        state = rows[frame]["state"]
        r2s.set_pose(model, data, state, joint_config(current_offsets()))
        movie.set_visual_gripper(model, data, state[5])
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()

    def redraw(_=None):
        frame = selected[0]
        real = reals[frame]
        rgb = render(frame)
        r_artist.set_data(real)
        overlay_real_artist.set_data(real)
        sim_artist.set_data(rgb)
        overlay_artist.set_data(rgb)
        overlay_artist.set_alpha(alpha_slider.val)
        status.set_text(f"frame {frame}; offsets are SHARED across every frame, not optimized per frame")
        fig.canvas.draw_idle()

    def choose_frame(label):
        selected[0] = int(label.split()[-1])
        redraw()

    def save(_):
        payload = robot_pose_payload(
            current_offsets(), args.episode, args.object_layout_json.as_posix(),
            str(args.cup_pose_json) if args.cup_pose_json else "None",
            (args.table_extra_left_cm, args.table_extra_front_cm,
             args.table_base_x_cm, args.table_base_y_cm,
             args.table_base_z_cm, args.table_yaw_delta_deg),
            frame_index=selected[0], overlay_alpha=alpha_slider.val)
        # Check all four frames with the *same* offsets before accepting a candidate.
        pngs = []
        for frame in frames:
            real = reals[frame]
            sim = render(frame)
            blend = np.clip(
                (1 - alpha_slider.val) * real.astype(float)
                + alpha_slider.val * sim.astype(float), 0, 255).astype(np.uint8)
            pngs.append(np.concatenate([real, sim, blend], axis=1))
        contact = np.concatenate(pngs, axis=0)
        outpng = args.output.with_suffix(".png")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.output.open("x", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
                file.write("\n")
        except FileExistsError:
            status.set_text("Output exists: choose another name")
            fig.canvas.draw_idle()
            return
        Image.fromarray(contact).save(outpng)
        status.set_text(f"Saved {args.output} + four-frame overlay PNG")
        print("Saved:", args.output, outpng)
        redraw()

    for slider in sliders.values():
        slider.on_changed(redraw)
    alpha_slider.on_changed(redraw)
    frame_radio.on_clicked(choose_frame)
    save_button.on_clicked(save)
    try:
        redraw()
        plt.show(block=True)
    finally:
        renderer.close()
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--episode", type=int, default=4)
    p.add_argument("--frames", type=int, nargs="+", default=list(DEFAULT_FRAMES))
    p.add_argument("--extrinsics", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path, default=movie.DEFAULT_INTRINSICS)
    p.add_argument("--urdf", type=Path, default=movie.DEFAULT_URDF)
    p.add_argument("--object-layout-json", type=Path, required=True)
    p.add_argument("--cup-pose-json", type=Path)
    p.add_argument("--table-extra-left-cm", type=float, default=4.)
    p.add_argument("--table-extra-front-cm", type=float, default=5.)
    p.add_argument("--table-base-x-cm", type=float, default=12.)
    p.add_argument("--table-base-y-cm", type=float, default=7.)
    p.add_argument("--table-base-z-cm", type=float, default=0.)
    p.add_argument("--table-yaw-delta-deg", type=float, default=-9.375)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
