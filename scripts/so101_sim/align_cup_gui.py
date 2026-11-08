#!/usr/bin/env python3
"""Offline Real/Sim alignment GUI: adjust the CUP in base XYZ, never image scale.

Displays recorded Episode 4 Side frame alongside MuJoCo render and alpha
overlay. No robot connection. Save a portable, relative-path-only JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import compare_real_sim_motion as movie
import real_to_sim as r2s


def cup_geom_ids(model):
    """Cup base + wall pieces; explicitly exclude the robot, desk and Muhi."""
    import mujoco
    names = ("r2s_cup_base",) + tuple(f"r2s_cup_wall_{i:02d}" for i in range(16))
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
           for name in names]
    if any(index < 0 for index in ids):
        raise ValueError("Missing cup geometry in the MuJoCo preview")
    return ids


def set_cup_delta(model, ids, original_positions, xyz_m):
    """Always position cup from immutable reference; no cumulative drift."""
    change = np.asarray(xyz_m, dtype=float)
    original = np.asarray(original_positions, dtype=float)
    if change.shape != (3,) or original.shape != (len(ids), 3):
        raise ValueError("Cup XYZ or reference shape mismatch")
    if not np.isfinite(change).all() or not np.isfinite(original).all():
        raise ValueError("Cup coordinates must be finite")
    model.geom_pos[ids] = original + change


def alignment_payload(xyz_m, object_layout, episode, table):
    delta = np.asarray(xyz_m, dtype=float)
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise ValueError("Cup offset must contain three finite numbers")
    if len(table) != 4 or not np.isfinite(table).all():
        raise ValueError("Table pose flags must be finite")
    return {
        "schema": "so101_cup_pose_delta_v1",
        "episode": int(episode),
        "cup_delta_base_m": [round(float(n), 6) for n in delta],
        "coordinate_frame": "so101_base",
        "reference": "existing three-click layout, no manual image scaling",
        "object_layout_json": str(object_layout),
        "table_base_x_cm": float(table[0]),
        "table_base_y_cm": float(table[1]),
        "table_yaw_delta_deg": float(table[2]),
        "table_base_z_cm": float(table[3]),
        "note": "Visual fit candidate, not measured 3D pose; camera/robot/Muhi unchanged.",
    }


def read_cup_payload(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "so101_cup_pose_delta_v1":
        raise ValueError("Unsupported cup pose JSON format")
    xyz = np.asarray(payload.get("cup_delta_base_m"), dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("Invalid cup_delta_base_m")
    return payload


def run(args):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    import mujoco

    if args.output.exists():
        raise ValueError("Refusing to overwrite existing cup pose JSON")
    dataset = args.dataset_root.resolve(strict=True)
    if dataset in args.output.resolve().parents:
        raise ValueError("Output must be outside source dataset")

    ep = r2s.read_episode(dataset, args.episode, 1, 1)
    video, start = movie.side_video_source(dataset, ep)
    real = np.asarray(movie.read_frames_once(video, [start])[start]
                      .convert("RGB").resize((640, 480), Image.Resampling.BILINEAR))
    intrinsics = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    extrinsics = json.loads(args.extrinsics.read_text(encoding="utf-8"))
    if (intrinsics.get("width"), intrinsics.get("height")) != (640, 480):
        raise ValueError("Expected Side camera intrinsics for 640x480")

    muhi_pixel, cup_pixel, yaw = movie.read_object_targets(args.object_layout_json)
    layout = movie.video_object_layout(
        extrinsics["T_base_camera"], intrinsics, muhi_pixel, cup_pixel, yaw)
    model, camera = movie.make_calibrated_model(
        extrinsics, intrinsics, args.urdf, layout,
        table_extra_left_m=args.table_extra_left_cm / 100.,
        table_extra_front_m=args.table_extra_front_cm / 100.)
    table_ids = movie.table_geom_ids(model)
    movie.set_table_pose_delta(
        model, model.geom_pos[table_ids].copy(), model.geom_quat[table_ids].copy(),
        np.asarray([args.table_base_x_cm, args.table_base_y_cm,
                    args.table_base_z_cm]) / 100., args.table_yaw_delta_deg)
    cup_ids = cup_geom_ids(model)
    original_cup = model.geom_pos[cup_ids].copy()
    data = mujoco.MjData(model)
    config = {"joints": {"sign": {name: 1 for name in r2s.JOINTS},
                         "offset_deg": {name: 0. for name in r2s.JOINTS}}}
    state = ep["rows"][0]["state"]
    r2s.set_pose(model, data, state, config)
    movie.set_visual_gripper(model, data, state[5])

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=False)
    fig.subplots_adjust(bottom=.33, left=.025, right=.99, top=.91, wspace=.04)
    for ax, title in zip(axes, ["REAL / Episode 4", "SIM / MuJoCo", "OVERLAY / alpha"]):
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    axes[0].imshow(real)
    blank = np.zeros_like(real)
    sim_artist = axes[1].imshow(blank)
    axes[2].imshow(real)
    overlay_artist = axes[2].imshow(blank, alpha=.55)
    controls = {}
    for key, y in (("x", .235), ("y", .185), ("z", .135)):
        axis = fig.add_axes([.18, y, .65, .032])
        controls[key] = Slider(axis, f"base {key.upper()} delta (mm)",
                               -300 if key != "z" else -50,
                               300 if key != "z" else 100,
                               valinit=0., valstep=1.)
    alpha_ax = fig.add_axes([.18, .08, .65, .025])
    alpha_slider = Slider(alpha_ax, "overlay alpha", 0, 1, valinit=.55)
    save_ax = fig.add_axes([.85, .01, .12, .05])
    save_btn = Button(save_ax, "SAVE JSON")
    status = fig.text(.025, .02,
                      "Cup only. X/Y/Z are SO-101 BASE axes. Perspective is automatic.",
                      fontsize=9)
    renderer = mujoco.Renderer(model, width=640, height=480)

    def current_xyz():
        return np.asarray([controls[k].val / 1000. for k in ("x", "y", "z")])

    def redraw(_=None):
        set_cup_delta(model, cup_ids, original_cup, current_xyz())
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render().copy()
        sim_artist.set_data(rgb)
        overlay_artist.set_data(rgb)
        overlay_artist.set_alpha(alpha_slider.val)
        fig.canvas.draw_idle()

    def save(_):
        payload = alignment_payload(
            current_xyz(), args.object_layout_json.as_posix(), args.episode,
            (args.table_base_x_cm, args.table_base_y_cm,
             args.table_yaw_delta_deg, args.table_base_z_cm))
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
                file.write("\n")
        except FileExistsError:
            status.set_text("File already exists — refusing overwrite")
            fig.canvas.draw_idle()
            return
        status.set_text(f"Saved: {args.output}")
        print("Saved:", args.output)
        fig.canvas.draw_idle()

    for slider in controls.values():
        slider.on_changed(redraw)
    alpha_slider.on_changed(lambda _: (overlay_artist.set_alpha(alpha_slider.val),
                                       fig.canvas.draw_idle()))
    save_btn.on_clicked(save)
    try:
        redraw()
        plt.show(block=True)
    finally:
        renderer.close()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=4)
    parser.add_argument("--extrinsics", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, default=movie.DEFAULT_INTRINSICS)
    parser.add_argument("--urdf", type=Path, default=movie.DEFAULT_URDF)
    parser.add_argument("--object-layout-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table-extra-left-cm", type=float, default=4.)
    parser.add_argument("--table-extra-front-cm", type=float, default=5.)
    parser.add_argument("--table-base-x-cm", type=float, default=12.)
    parser.add_argument("--table-base-y-cm", type=float, default=7.)
    parser.add_argument("--table-base-z-cm", type=float, default=0.)
    parser.add_argument("--table-yaw-delta-deg", type=float, default=-9.375)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
