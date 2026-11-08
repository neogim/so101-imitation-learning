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



def set_cup_dimensions(model, ids, reference_positions, reference_sizes,
                       diameter_m=.100, height_m=.090, xyz_m=(0., 0., 0.)):
    """Resize diameter/height of the cup in *3D* about its bottom-center.

    Preserve the existing 16 radial wall orientations, base thickness,
    and baseline cup center. Bottom stays at the original z position.
    Every redraw starts from immutable positions/sizes (no cumulative drift).
    """
    xyz = np.asarray(xyz_m, dtype=float)
    orig = np.asarray(reference_positions, dtype=float)
    sizes = np.asarray(reference_sizes, dtype=float)
    if (orig.shape != (17, 3) or sizes.shape != (17, 3)
            or np.asarray(ids).shape != (17,) or xyz.shape != (3,)
            or not np.isfinite(orig).all() or not np.isfinite(sizes).all()
            or not np.isfinite(xyz).all() or not np.isfinite([diameter_m, height_m]).all()
            or not (.04 <= diameter_m <= .20 and .04 <= height_m <= .20)):
        raise ValueError("Invalid cup geometry or dimensions")
    radial_scale = diameter_m / .100
    base_center = orig[0]
    centers = orig.copy()
    centers[:, :2] = base_center[:2] + radial_scale * (orig[:, :2] - base_center[:2])
    # Bottom of wall is baseline original wall center - original half height.
    bottom_z = orig[1:, 2] - sizes[1:, 2]
    centers[1:, 2] = bottom_z + height_m / 2
    model.geom_pos[ids] = centers + xyz
    adjusted = sizes.copy()
    adjusted[0, 0] = diameter_m / 2
    adjusted[1:, 0] = sizes[1:, 0] * radial_scale  # tangential coverage
    adjusted[1:, 2] = height_m / 2
    model.geom_size[ids] = adjusted


def validate_cup_dimensions(diameter_m, height_m):
    if (not np.isfinite([diameter_m, height_m]).all()
            or not .04 <= diameter_m <= .20 or not .04 <= height_m <= .20):
        raise ValueError("Cup dimensions must be 40-200 mm")
    return diameter_m, height_m


def alignment_payload(xyz_m, object_layout, episode, table,\n                      diameter_m=.100, height_m=.090):
    delta = np.asarray(xyz_m, dtype=float)
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise ValueError("Cup offset must contain three finite numbers")
    if len(table) != 4 or not np.isfinite(table).all():
        raise ValueError("Table pose flags must be finite")
    return {
        "schema": "so101_cup_pose_delta_v1",
        "episode": int(episode),
        "cup_delta_base_m": [round(float(n), 6) for n in delta],\n        "cup_diameter_m": round(float(diameter_m), 6),\n        "cup_height_m": round(float(height_m), 6),
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
    original_cup = model.geom_pos[cup_ids].copy()\n    original_cup_size = model.geom_size[cup_ids].copy()
    data = mujoco.MjData(model)
    config = {"joints": {"sign": {name: 1 for name in r2s.JOINTS},
                         "offset_deg": {name: 0. for name in r2s.JOINTS}}}
    state = ep["rows"][0]["state"]
    r2s.set_pose(model, data, state, config)
    movie.set_visual_gripper(model, data, state[5])

    start_xyz = np.zeros(3)\n    start_diameter_m, start_height_m = .100, .090\n    if args.initial_pose_json is not None:\n        previous = read_cup_payload(args.initial_pose_json)\n        if previous.get("episode") != args.episode:\n            raise ValueError("Previous pose belongs to another episode")\n        start_xyz = np.asarray(previous["cup_delta_base_m"], dtype=float)\n        start_diameter_m = float(previous.get("cup_diameter_m", .100))\n        start_height_m = float(previous.get("cup_height_m", .090))\n\n    fig, axes = plt.subplots(1, 3, figsize=(16, 7.6), constrained_layout=False)
    fig.subplots_adjust(bottom=.43, left=.025, right=.99, top=.94, wspace=.04)
    for ax, title in zip(axes, ["REAL / Episode 4", "SIM / MuJoCo", "OVERLAY / alpha"]):
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    axes[0].imshow(real)
    blank = np.zeros_like(real)
    sim_artist = axes[1].imshow(blank)
    axes[2].imshow(real)
    overlay_artist = axes[2].imshow(blank, alpha=.55)
    controls = {}
    for key, y in (("x", .34), ("y", .29), ("z", .24)):
        axis = fig.add_axes([.18, y, .65, .032])
        controls[key] = Slider(axis, f"base {key.upper()} delta (mm)",
                               -300 if key != "z" else -50,
                               300 if key != "z" else 100,
                               valinit=1000 * start_xyz[("x", "y", "z").index(key)], valstep=1.)
    dim_sliders = {}\n    for name, y, min_value, max_value, init in (\n        ("diameter", .19, 60., 130., start_diameter_m * 1000),\n        ("height", .14, 60., 140., start_height_m * 1000)):\n        axis = fig.add_axes([.18, y, .65, .032])\n        dim_sliders[name] = Slider(axis, f"cup {name} (mm)", min_value, max_value,\n                                   valinit=init, valstep=1.)\n    alpha_ax = fig.add_axes([.18, .085, .65, .025])
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
        set_cup_dimensions(\n            model, cup_ids, original_cup, original_cup_size,\n            diameter_m=dim_sliders["diameter"].val / 1000.,\n            height_m=dim_sliders["height"].val / 1000., xyz_m=current_xyz())
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
             args.table_yaw_delta_deg, args.table_base_z_cm),\n            diameter_m=dim_sliders["diameter"].val / 1000.,\n            height_m=dim_sliders["height"].val / 1000.)
        payload["table_extra_left_cm"] = float(args.table_extra_left_cm)
        payload["table_extra_front_cm"] = float(args.table_extra_front_cm)
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
                file.write("\n")
        except FileExistsError:
            status.set_text("File already exists — refusing overwrite")
            fig.canvas.draw_idle()
            return
        # Export a human-readable reproducible PNG with the exact current
        # render so a GUI screenshot is unnecessary when seeking feedback.
        sim_rgb = np.asarray(sim_artist.get_array(), dtype=np.uint8)
        overlay_rgb = np.clip(
            (1 - alpha_slider.val) * real.astype(float)
            + alpha_slider.val * sim_rgb.astype(float), 0, 255).astype(np.uint8)
        preview = np.concatenate([real, sim_rgb, overlay_rgb], axis=1)
        preview_file = args.output.with_suffix(".png")
        Image.fromarray(preview).save(preview_file)
        print("Preview:", preview_file)
        status.set_text(f"Saved: {args.output} and {preview_file}")
        print("Saved:", args.output)
        fig.canvas.draw_idle()

    for slider in list(controls.values()) + list(dim_sliders.values()):
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
    parser.add_argument("--output", type=Path, required=True)\n    parser.add_argument("--initial-pose-json", type=Path, default=None,\n                        help="Reload previous saved GUI XYZ and dimensions")
    parser.add_argument("--table-extra-left-cm", type=float, default=4.)
    parser.add_argument("--table-extra-front-cm", type=float, default=5.)
    parser.add_argument("--table-base-x-cm", type=float, default=12.)
    parser.add_argument("--table-base-y-cm", type=float, default=7.)
    parser.add_argument("--table-base-z-cm", type=float, default=0.)
    parser.add_argument("--table-yaw-delta-deg", type=float, default=-9.375)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
