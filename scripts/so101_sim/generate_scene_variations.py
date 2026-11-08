#!/usr/bin/env python3
"""Offline Phase 1.25: randomize Muhi/cup layout and render MuJoCo previews.

This produces *scene candidates*, NOT training demonstrations, valid IK
trajectories, grasp simulations or physical success claims. No robot access.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import align_cup_gui as cup_gui
import compare_real_sim_motion as movie
import compare_id1_eye_to_hand as geometry
import real_to_sim as r2s


def generate_offsets(count=20, seed=101, muhi_range_mm=20.,
                     cup_range_mm=20., yaw_range_deg=30.):
    """Generate deterministic baseline + separated object perturbation candidates.

    The ranges are maximum absolute deltas in BASE X/Y and Muhi's table yaw.
    The values are *not* reachability or collision guarantees.
    """
    if not isinstance(count, int) or not 1 <= count <= 10000:
        raise ValueError("count must be an integer in [1, 10000]")
    limits = np.asarray([muhi_range_mm, cup_range_mm, yaw_range_deg], float)
    if not np.isfinite(limits).all() or (limits < 0).any():
        raise ValueError("Ranges must be finite nonnegative numbers")
    rng = np.random.default_rng(seed)
    samples = [{
        "candidate": 0,
        "muhi_xy_delta_mm": [0., 0.],
        "muhi_yaw_delta_deg": 0.,
        "cup_xy_delta_mm": [0., 0.],
    }]
    for index in range(1, count):
        # Cycle single-variable types before mixing them. Deterministic for seed.
        mode = (index - 1) % 4
        mu = rng.uniform(-muhi_range_mm, muhi_range_mm, 2) if mode in (0, 3) else np.zeros(2)
        yaw = float(rng.uniform(-yaw_range_deg, yaw_range_deg)) if mode in (1, 3) else 0.
        cup = rng.uniform(-cup_range_mm, cup_range_mm, 2) if mode in (2, 3) else np.zeros(2)
        samples.append({
            "candidate": index,
            "muhi_xy_delta_mm": np.round(mu, 3).tolist(),
            "muhi_yaw_delta_deg": round(yaw, 3),
            "cup_xy_delta_mm": np.round(cup, 3).tolist(),
        })
    return samples


def rotate_translate_xy(points, origin_xy, angle_deg, delta_xy_m):
    """Rigid planar transform around the original object center."""
    points = np.asarray(points, float)
    origin = np.asarray(origin_xy, float)
    delta = np.asarray(delta_xy_m, float)
    if points.ndim != 2 or points.shape[1] < 2 or origin.shape != (2,) or delta.shape != (2,):
        raise ValueError("Expected Nx2 or Nx3 points, XY origin and XY delta")
    if not np.isfinite(points).all() or not np.isfinite(origin).all() or not np.isfinite(delta).all():
        raise ValueError("Non-finite object transform")
    if not np.isfinite(angle_deg):
        raise ValueError("Non-finite yaw")
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    rotation = np.array([[c, -s], [s, c]])
    output = points.copy()
    output[:, :2] = (points[:, :2] - origin) @ rotation.T + origin + delta
    return output


def z_yaw_quat(angle_deg):
    rad = np.deg2rad(angle_deg) / 2.
    return np.array([np.cos(rad), 0., 0., np.sin(rad)], dtype=float)


def quat_product(a, b):
    """Hamilton product for MuJoCo [w,x,y,z] quaternions."""
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def scene_geom_ids(model):
    import mujoco
    mu = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
          for n in ("r2s_muhi_body", "r2s_muhi_cap")]
    cup = cup_gui.cup_geom_ids(model)
    if any(i < 0 for i in mu) or not len(cup):
        raise ValueError("Muhi or cup preview geometry not found")
    return mu, cup


def capture_reference(model):
    """Read geometry positions/quaternions once; never accumulate edit drift."""
    mu, cup = scene_geom_ids(model)
    weights = model.geom_size[mu, 1].astype(float)
    center = np.average(model.geom_pos[mu, :2], axis=0, weights=weights)
    return {
        "muhi_ids": mu,
        "cup_ids": cup,
        "muhi_pos": model.geom_pos[mu].copy(),
        "muhi_quat": model.geom_quat[mu].copy(),
        "muhi_xy": center,
        "cup_pos": model.geom_pos[cup].copy(),
        "cup_quat": model.geom_quat[cup].copy(),
        "cup_xy": model.geom_pos[cup[0], :2].copy(),
    }


def apply_candidate(model, reference, candidate):
    """Move only independent world geoms for Muhi and cup in base coordinates."""
    mid, cid = reference["muhi_ids"], reference["cup_ids"]
    delta_mu = np.asarray(candidate["muhi_xy_delta_mm"], float) / 1000.
    delta_cup = np.asarray(candidate["cup_xy_delta_mm"], float) / 1000.
    yaw = float(candidate["muhi_yaw_delta_deg"])
    model.geom_pos[mid] = rotate_translate_xy(
        reference["muhi_pos"], reference["muhi_xy"], yaw, delta_mu)
    quat = z_yaw_quat(yaw)
    for j, gid in enumerate(mid):
        model.geom_quat[gid] = quat_product(quat, reference["muhi_quat"][j])
    model.geom_pos[cid] = rotate_translate_xy(
        reference["cup_pos"], reference["cup_xy"], 0., delta_cup)
    model.geom_quat[cid] = reference["cup_quat"]


def run(args):
    import mujoco

    if args.output.exists():
        raise ValueError("Output directory already exists")
    dataset = args.dataset_root.resolve(strict=True)
    if dataset == args.output.resolve() or dataset in args.output.resolve().parents:
        raise ValueError("Output must be outside source dataset")

    ep = r2s.read_episode(dataset, args.episode, 1, 1)
    intr = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    extr = json.loads(args.extrinsics.read_text(encoding="utf-8"))
    if (intr.get("width"), intr.get("height")) != (640, 480):
        raise ValueError("Expected 640x480 calibrated Side intrinsics")
    muhi, cup, yaw = movie.read_object_targets(args.object_layout_json)
    layout = movie.video_object_layout(
        extr["T_base_camera"], intr, muhi, cup, yaw)
    model, camera = movie.make_calibrated_model(
        extr, intr, args.urdf, layout,
        table_extra_left_m=args.table_extra_left_cm / 100.,
        table_extra_front_m=args.table_extra_front_cm / 100.)
    table_ids = movie.table_geom_ids(model)
    movie.set_table_pose_delta(
        model, model.geom_pos[table_ids].copy(),
        model.geom_quat[table_ids].copy(),
        np.asarray([args.table_base_x_cm, args.table_base_y_cm,
                    args.table_base_z_cm]) / 100.,
        args.table_yaw_delta_deg)

    # Use the *saved* cup registration, not the pre-GUI default.
    saved = cup_gui.read_cup_payload(args.cup_pose_json)
    if saved.get("episode") != args.episode:
        raise ValueError("Cup pose episode mismatch")
    if Path(saved.get("object_layout_json", "")) != args.object_layout_json:
        raise ValueError("Cup pose object-layout mismatch")
    for name in ("table_extra_left_cm", "table_extra_front_cm",
                 "table_base_x_cm", "table_base_y_cm", "table_base_z_cm",
                 "table_yaw_delta_deg"):
        if not np.isclose(float(saved.get(name, np.nan)), getattr(args, name)):
            raise ValueError(f"Cup pose table scene mismatch: {name}")
    cup_ids = cup_gui.cup_geom_ids(model)
    cup_gui.set_cup_dimensions(
        model, cup_ids, model.geom_pos[cup_ids].copy(),
        model.geom_size[cup_ids].copy(),
        diameter_m=geometry.CUP_DIAMETER_M, height_m=geometry.CUP_HEIGHT_M,
        xyz_m=saved["cup_delta_base_m"])

    data = mujoco.MjData(model)
    config = {"joints": {"sign": {name: 1 for name in r2s.JOINTS},
                         "offset_deg": {name: 0. for name in r2s.JOINTS}}}
    initial_state = ep["rows"][0]["state"]
    r2s.set_pose(model, data, initial_state, config)
    movie.set_visual_gripper(model, data, initial_state[5])
    reference = capture_reference(model)

    candidates = generate_offsets(args.count, args.seed, args.muhi_range_mm,
                                  args.cup_range_mm, args.yaw_range_deg)
    args.output.mkdir(parents=True, exist_ok=False)
    tiles = []
    with mujoco.Renderer(model, width=640, height=480) as renderer:
        for c in candidates:
            apply_candidate(model, reference, c)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            arr = renderer.render().copy()
            image = Image.fromarray(arr).resize((320, 240))
            ImageDraw.Draw(image).text((8, 8), f"Scene {c['candidate']:02d}", fill=(255, 255, 0))
            tiles.append(image)
            image.save(args.output / f"scene_{c['candidate']:03d}.png")
            c["muhi_center_base_m"] = (
                reference["muhi_xy"] + np.asarray(c["muhi_xy_delta_mm"]) / 1000.).round(5).tolist()
            c["cup_center_base_m"] = (
                reference["cup_xy"] + np.asarray(c["cup_xy_delta_mm"]) / 1000.).round(5).tolist()
            c["trajectory_status"] = "NOT_GENERATED"
            c["ik_status"] = "NOT_RUN"
            c["grasp_success"] = "NOT_TESTED"
    columns = min(4, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    contact = Image.new("RGB", (320 * columns, 240 * rows), (30, 30, 30))
    for i, tile in enumerate(tiles):
        contact.paste(tile, ((i % columns) * 320, (i // columns) * 240))
    contact.save(args.output / "contact_sheet.png")
    manifest = {
        "schema": "so101_phase125_scene_candidates_v1",
        "status": "SCENE_PREVIEWS_ONLY_NO_IK_OR_ACT_TRAINING_DATA",
        "episode": args.episode,
        "seed": args.seed,
        "reference_frame": 0,
        "object_layout_json": str(args.object_layout_json),
        "cup_pose_json": str(args.cup_pose_json),
        "source_dataset": str(args.dataset_root),
        "robot_joint_status": "RECORDED_FRAME_0_ZERO_OFFSET_UNVERIFIED",
        "table_flags_cm_deg": {
            "table_extra_left_cm": args.table_extra_left_cm,
            "table_extra_front_cm": args.table_extra_front_cm,
            "table_base_x_cm": args.table_base_x_cm,
            "table_base_y_cm": args.table_base_y_cm,
            "table_base_z_cm": args.table_base_z_cm,
            "table_yaw_delta_deg": args.table_yaw_delta_deg,
        },
        "candidates": candidates,
        "next_step": "Extract object-relative demonstration TCP path and solve IK; reject unsafe candidates before dataset export.",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Created {len(candidates)} OFFLINE scene previews in {args.output}")
    print("No IK, trajectories, grasp success or training episodes were generated.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--episode", type=int, default=4)
    p.add_argument("--extrinsics", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path, default=movie.DEFAULT_INTRINSICS)
    p.add_argument("--urdf", type=Path, default=movie.DEFAULT_URDF)
    p.add_argument("--object-layout-json", type=Path, required=True)
    p.add_argument("--cup-pose-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--muhi-range-mm", type=float, default=20.)
    p.add_argument("--cup-range-mm", type=float, default=20.)
    p.add_argument("--yaw-range-deg", type=float, default=30.)
    p.add_argument("--table-extra-left-cm", type=float, default=4.)
    p.add_argument("--table-extra-front-cm", type=float, default=5.)
    p.add_argument("--table-base-x-cm", type=float, default=12.)
    p.add_argument("--table-base-y-cm", type=float, default=7.)
    p.add_argument("--table-base-z-cm", type=float, default=0.)
    p.add_argument("--table-yaw-delta-deg", type=float, default=-9.375)
    run(p.parse_args())


if __name__ == "__main__":
    main()
