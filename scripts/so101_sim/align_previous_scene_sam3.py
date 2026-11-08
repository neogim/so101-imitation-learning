#!/usr/bin/env python3
"""Offline SAM3 tabletop alignment for the *calibrated-candidate* video renderer.

Uses compare_real_sim_motion.make_calibrated_model, NOT the uncalibrated
workcell.example.json renderer. All geometry is static; no hardware access.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import compare_id1_eye_to_hand as scene
import compare_real_sim_motion as movie


TABLE_PART_NAMES = ("r2s_preview_table",) + tuple(
    f"r2s_table_leg_{x}{y}" for x in ("m", "p") for y in ("m", "p")
)


def overlap_metrics(real, sim):
    real, sim = np.asarray(real, bool), np.asarray(sim, bool)
    if real.ndim != 2 or real.shape != sim.shape:
        raise ValueError("Real and simulated binary masks must have identical HxW")
    if not real.any() or not sim.any():
        raise ValueError("Target is empty in real or simulated mask")
    intersection = int(np.logical_and(real, sim).sum())
    union = int(np.logical_or(real, sim).sum())
    yr, xr = np.argwhere(real).mean(axis=0)
    ys, xs = np.argwhere(sim).mean(axis=0)
    return {"iou": intersection / union,
            "area_ratio_sim_over_real": float(sim.sum() / real.sum()),
            "centroid_delta_sim_minus_real_px": [float(xs-xr), float(ys-yr)]}


def load_mask(image_path, mask_path=None, sam3_model=None, bbox=None, prompt="wooden table"):
    if mask_path is not None:
        with Image.open(mask_path) as im:
            mask = np.asarray(im.convert("L")) > 127
    else:
        if not sam3_model or not Path(sam3_model).is_file():
            raise ValueError("SAM3 weights file is missing. Supply a real --sam3-model path "
                             "or --real-mask from an existing segmentation.")
        # Import only when model inference is explicitly requested.
        if bbox is not None:
            from ultralytics import SAM
            pred = SAM(str(sam3_model)).predict(
                source=str(image_path), bboxes=list(bbox), verbose=False)
        else:
            from ultralytics.models.sam import SAM3SemanticPredictor
            predictor = SAM3SemanticPredictor(overrides={
                "model": str(sam3_model), "task": "segment", "mode": "predict",
                "conf": .25, "verbose": False})
            predictor.set_image(str(image_path))
            pred = predictor(text=[prompt])
        if not pred or pred[0].masks is None:
            raise ValueError("SAM3 returned no mask")
        masks = pred[0].masks.data.cpu().numpy() > .5
        if len(masks) != 1:
            raise ValueError(f"Expected exactly one tabletop mask, got {len(masks)}")
        mask = masks[0]
    with Image.open(image_path) as im:
        expected = (im.height, im.width)
    if mask.shape != expected:
        raise ValueError(f"Mask shape {mask.shape} != original image {expected}")
    if not mask.any() or mask.all():
        raise ValueError("Mask must contain both foreground and background")
    return mask


def segmentation_mask(renderer, model, data, camera, table_geom_id):
    import mujoco

    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
    seg = renderer.render()
    return np.logical_and(
        seg[:, :, 0] == table_geom_id,
        seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM),
    )


def mask_bbox(mask):
    """Outer visible mask bounding rectangle, not 3D corners."""
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError("Empty tabletop mask")
    return [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]


def silhouette_diagnostics(real, sim):
    """Include envelope residuals; IoU alone conceals oversize and misplacement."""
    metrics = overlap_metrics(real, sim)
    rb, sb = mask_bbox(real), mask_bbox(sim)
    metrics["real_bbox_xyxy_px"] = rb
    metrics["sim_bbox_xyxy_px"] = sb
    metrics["bbox_delta_sim_minus_real_px"] = [s - r for s, r in zip(sb, rb)]
    metrics["bbox_max_abs_error_px"] = max(abs(x) for x in metrics["bbox_delta_sim_minus_real_px"])
    return metrics


def visible_upper_edge_residual(real, sim, x_min=440, x_max=600):
    """Compare tabletop topmost visible pixels in a selected unobstructed strip.

    This is a 2D image boundary diagnostic, NOT verified 3D tabletop corners.
    A user-selected strip must avoid the arm, Muhi/cup and mask ambiguity.
    """
    real, sim = np.asarray(real, bool), np.asarray(sim, bool)
    if real.shape != sim.shape or real.ndim != 2:
        raise ValueError("Incompatible tabletop masks")
    height, width = real.shape
    if not (0 <= x_min < x_max <= width):
        raise ValueError("Invalid edge X interval")
    errors = []
    for x in range(x_min, x_max):
        r = np.flatnonzero(real[:, x])
        s = np.flatnonzero(sim[:, x])
        if not r.size or not s.size:
            continue
        # Columns with foreground touching the top border cannot give a
        # meaningful work surface far-edge measurement.
        if r[0] <= 2 or s[0] <= 2:
            continue
        errors.append(int(s[0]) - int(r[0]))
    if not errors:
        return {"sample_columns": 0, "mean_abs_error_px": None,
                "median_sim_minus_real_px": None}
    e = np.asarray(errors, dtype=float)
    return {"sample_columns": len(errors), "mean_abs_error_px": float(np.abs(e).mean()),
            "median_sim_minus_real_px": float(np.median(e))}


def alignment_score(real, sim, edge_x_min=440, edge_x_max=600):
    """Image-space heuristic; IoU plus scale and visible envelope agreement.

    A high score is not proof of metric 3D alignment or physical corners.
    """
    if not sim.any():
        return -1.
    m = silhouette_diagnostics(real, sim)
    import math
    area_penalty = abs(math.log(m["area_ratio_sim_over_real"]))
    # Normalize bbox differences by image dimensions, not by world axes.
    height, width = real.shape
    edges = m["bbox_delta_sim_minus_real_px"]
    border_penalty = sum(abs(v) / (width if i % 2 == 0 else height)
                         for i, v in enumerate(edges)) / 4.
    # Synthetic unit-test masks may be smaller than the 640px Side image.
    # For such images, disable only the optional edge term.
    if 0 <= edge_x_min < edge_x_max <= width:
        boundary = visible_upper_edge_residual(real, sim, edge_x_min, edge_x_max)
        edge_penalty = ((boundary["mean_abs_error_px"] or 0.) / height
                        if boundary["sample_columns"] >= 20 else 0.)
    else:
        edge_penalty = 0.
    return m["iou"] - 0.20 * area_penalty - 0.35 * border_penalty - 0.50 * edge_penalty


def offset_xy(T_base_camera, left_m, front_m):
    """Legacy camera-relative offset, retained only for compatibility/tests."""
    return scene.workcell_shift_left_front(T_base_camera, left_m, front_m)[:2]


def apply_base_offset(model, xyz_cm):
    """World/base XYZ only: never touch camera, robot, props or table rotation."""
    movie.apply_table_base_offset(model, xyz_cm)


def optimize(model, data, renderer, camera, table_geom_id, real_mask,
             axes="xy", max_delta_m=.12, initial_step_m=.02, levels=4,
             edge_x_min=440, edge_x_max=600, fit_yaw=False,
             max_yaw_delta_deg=30., initial_yaw_step_deg=5.,
             robot_edge="off", robot_anchor_xy_m=(0., 0.), robot_edge_inset_m=0.,
             initial_delta_xyz_m=(0., 0., 0.), initial_yaw_delta_deg=0.):
    """Fit table XY and optionally yaw against visible mask and far-edge strip.

    Only tabletop and legs are transformed. Their physical 800x510mm
    footprint, camera, robot, Muhi and cup are fixed. No guessed hidden clicks.
    """
    import mujoco

    if max_delta_m <= 0 or initial_step_m <= 0 or levels <= 0:
        raise ValueError("Search parameters must be positive")
    if fit_yaw and (max_yaw_delta_deg <= 0 or initial_yaw_step_deg <= 0):
        raise ValueError("Yaw search limits must be positive")
    if axes not in ("x", "y", "z", "xy", "xyz"):
        raise ValueError("axes must be x, y, z, xy or xyz")
    ids = movie.table_geom_ids(model)
    original = model.geom_pos[ids].copy()
    orientations = model.geom_quat[ids].copy()
    if robot_edge not in ("off", "auto", "x+", "x-", "y+", "y-"):
        raise ValueError("Invalid robot anchor edge")
    if robot_edge == "auto":
        raise ValueError("Automatic edge choice is unsafe without measured clamp geometry; select edge explicitly or off")
    selected_edge = robot_edge
    active = tuple("xyz".index(a) for a in axes)
    selected = np.asarray(initial_delta_xyz_m, dtype=float).copy()
    if selected.shape != (3,) or not np.isfinite(selected).all():
        raise ValueError("Initial XYZ must contain 3 finite offsets")
    selected_yaw = float(initial_yaw_delta_deg)
    if not np.isfinite(selected_yaw):
        raise ValueError("Invalid initial yaw")
    selected_start = selected.copy()
    yaw_start = selected_yaw

    def trial(delta, yaw_deg):
        movie.set_table_pose_delta(model, original, orientations, delta, yaw_deg)
        if selected_edge != "off":
            movie.anchor_table_edge_to_robot(
                model, selected_edge, robot_anchor_xy_m, robot_edge_inset_m)
        candidate = segmentation_mask(renderer, model, data, camera, table_geom_id)
        if not candidate.any():
            return -1.
        return alignment_score(real_mask, candidate, edge_x_min, edge_x_max)

    before = trial(selected, selected_yaw)
    best = before
    # Fixed finite iterations. Keep the selection axes independent while
    # allowing yaw and translation to re-adjust each other twice per level.
    for level in range(levels):
        step = initial_step_m / (2 ** level)
        yaw_step = initial_yaw_step_deg / (2 ** level)
        for _ in range(2):
            for axis in active:
                candidate, candidate_score = selected.copy(), best
                for d in (-2, -1, 1, 2):
                    new_delta = selected.copy()
                    new_delta[axis] += step * d
                    if abs(new_delta[axis] - selected_start[axis]) > max_delta_m + 1e-10:
                        continue
                    score = trial(new_delta, selected_yaw)
                    if score > candidate_score + 1e-7:
                        candidate, candidate_score = new_delta, score
                selected, best = candidate, candidate_score
            if fit_yaw:
                candidate_yaw, candidate_score = selected_yaw, best
                for d in (-2, -1, 1, 2):
                    new_yaw = selected_yaw + yaw_step * d
                    if abs(new_yaw - yaw_start) > max_yaw_delta_deg + 1e-10:
                        continue
                    score = trial(selected, new_yaw)
                    if score > candidate_score + 1e-7:
                        candidate_yaw, candidate_score = new_yaw, score
                selected_yaw, best = candidate_yaw, candidate_score
    trial(selected, selected_yaw)
    return {
        "axes": axes,
        "initial_base_delta_xyz_m": selected_start.tolist(),
        "initial_yaw_delta_deg": yaw_start,
        "base_delta_xyz_m": selected.tolist(),
        "yaw_delta_deg": float(selected_yaw),
        "table_yaw_deg": float(15. + selected_yaw),
        "objective_before": before,
        "objective_after": best,
        "robot_anchor_edge": selected_edge,
        "robot_anchor_xy_m": list(robot_anchor_xy_m),
        "robot_edge_inset_m": robot_edge_inset_m,
        "robot_table_edge": (
            movie.anchor_table_edge_to_robot(
                model, selected_edge, robot_anchor_xy_m, robot_edge_inset_m)
            if selected_edge != "off" else None
        ),
        "search_bound_hit": {
            "xyz": any(abs(selected[j] - selected_start[j]) >= max_delta_m - 1e-8 for j in active),
            "yaw": bool(fit_yaw and abs(selected_yaw - yaw_start) >= max_yaw_delta_deg - 1e-8),
        },
    }


def overlay(image, real_mask, sim_mask):
    rgb = np.asarray(image.convert("RGB")).copy()
    rgb[np.logical_and(real_mask, sim_mask)] = (45, 185, 90)
    rgb[np.logical_and(real_mask, ~sim_mask)] = (230, 60, 60)
    rgb[np.logical_and(~real_mask, sim_mask)] = (60, 100, 230)
    return Image.fromarray(rgb)


def main():
    import mujoco

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-image", type=Path, required=True,
                   help="Original Side-camera frame (not combined comparison)")
    p.add_argument("--real-mask", type=Path, help="Reviewed binary mask, skips SAM3")
    p.add_argument("--sam3-model", type=Path)
    p.add_argument("--bbox", nargs=4, type=float)
    p.add_argument("--prompt", default="wooden table")
    p.add_argument("--extrinsics", type=Path, required=True,
                   help="Same extrinsics JSON used for successful comparison video")
    p.add_argument("--intrinsics", type=Path, required=True,
                   help="Same 640x480 intrinsics JSON used for successful comparison video")
    p.add_argument("--urdf", type=Path, default=movie.DEFAULT_URDF)
    p.add_argument("--dataset-root", type=Path,
                   help="Use the recorded episode first joint pose in the mask scene")
    p.add_argument("--episode", type=int, default=4)
    p.add_argument("--object-layout-json", type=Path,
                   help="Use the same Muhi/cup 3-click JSON as the comparison video")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-delta-m", type=float, default=.12)
    p.add_argument("--anchor-robot-edge", choices=("off", "auto", "x+", "x-", "y+", "y-"),
                   default="off",
                   help="Align tabletop border to the robot clamp anchor while fitting")
    p.add_argument("--robot-anchor-x-cm", type=float, default=0.)
    p.add_argument("--robot-anchor-y-cm", type=float, default=0.)
    p.add_argument("--robot-edge-inset-cm", type=float, default=0.)
    p.add_argument("--initial-base-x-cm", type=float, default=0.)
    p.add_argument("--initial-base-y-cm", type=float, default=0.)
    p.add_argument("--initial-base-z-cm", type=float, default=0.)
    p.add_argument("--initial-yaw-delta-deg", type=float, default=0.)
    p.add_argument("--fit-yaw", action="store_true",
                   help="Optimize tabletop yaw about its center alongside base XY")
    p.add_argument("--max-yaw-delta-deg", type=float, default=30.)
    p.add_argument("--initial-yaw-step-deg", type=float, default=5.)
    p.add_argument("--edge-x-min", type=int, default=440)
    p.add_argument("--edge-x-max", type=int, default=600)
    p.add_argument("--axis", choices=("x", "y", "z", "xy", "xyz"), default="xy",
                   help="SO-101 base-frame axis to optimize; default XY (Z remains fixed)")
    args = p.parse_args()

    if args.output.exists():
        p.error("Output directory already exists; refusing overwrite")
    with Image.open(args.real_image) as im:
        original = im.convert("RGB")
    if original.size != (640, 480):
        p.error("This renderer expects a 640x480 Side image; use the original "
                "recorded frame resized to 640x480, not the 1280x508 composite")
    real_mask = load_mask(args.real_image, args.real_mask,
                          args.sam3_model, args.bbox, args.prompt)
    extrinsics = json.loads(args.extrinsics.read_text())
    intrinsics = json.loads(args.intrinsics.read_text())
    if (intrinsics.get("width"), intrinsics.get("height")) != (640, 480):
        p.error("Side intrinsics must be calibrated for 640x480 pixels")

    # Use the SAME approximate object layout and first recorded joint state
    # as the comparison movie. Otherwise differently placed Muhi/cup and
    # default robot pose distort SAM3-vs-Sim occlusion comparisons.
    if bool(args.dataset_root) != bool(args.object_layout_json):
        p.error("--dataset-root and --object-layout-json are required together")
    layout = None
    recorded_first = None
    if args.dataset_root is not None:
        recorded_first = movie.r2s.read_episode(
            args.dataset_root.resolve(strict=True), args.episode, 1, 1)["rows"][0]
        muhi_px, cup_px, yaw = movie.read_object_targets(args.object_layout_json)
        layout = movie.video_object_layout(extrinsics["T_base_camera"], intrinsics,
                                           muhi_px, cup_px, yaw)
    model, camera = movie.make_calibrated_model(
        extrinsics, intrinsics, args.urdf, object_layout=layout)
    data = mujoco.MjData(model)
    if recorded_first is not None:
        joints_config = {
            "joints": {
                "sign": {name: 1 for name in movie.r2s.JOINTS},
                "offset_deg": {name: 0. for name in movie.r2s.JOINTS},
            }
        }
        movie.r2s.set_pose(model, data, recorded_first["state"], joints_config)
        movie.set_visual_gripper(model, data, recorded_first["state"][5])
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r2s_preview_table")
    if geom < 0:
        raise ValueError("No r2s_preview_table geom")
    with mujoco.Renderer(model, width=640, height=480) as renderer:
        renderer.enable_segmentation_rendering()
        first = segmentation_mask(renderer, model, data, camera, geom)
        if not first.any():
            raise ValueError("Simulated tabletop not visible with existing camera")
        result = optimize(model, data, renderer, camera, geom, real_mask,
                          axes=args.axis, max_delta_m=args.max_delta_m,
                          edge_x_min=args.edge_x_min, edge_x_max=args.edge_x_max,
                          fit_yaw=args.fit_yaw,
                          max_yaw_delta_deg=args.max_yaw_delta_deg,
                          initial_yaw_step_deg=args.initial_yaw_step_deg,
                          robot_edge=args.anchor_robot_edge,
                          robot_anchor_xy_m=(args.robot_anchor_x_cm / 100.,
                                             args.robot_anchor_y_cm / 100.),
                          robot_edge_inset_m=args.robot_edge_inset_cm / 100.,
                          initial_delta_xyz_m=np.array([
                              args.initial_base_x_cm, args.initial_base_y_cm,
                              args.initial_base_z_cm]) / 100.,
                          initial_yaw_delta_deg=args.initial_yaw_delta_deg)
        last = segmentation_mask(renderer, model, data, camera, geom)
    result["before"] = silhouette_diagnostics(real_mask, first)
    result["after"] = silhouette_diagnostics(real_mask, last)
    result["iou_before"] = result["before"]["iou"]
    result["iou_after"] = result["after"]["iou"]
    result["edge_strip_x_px"] = [args.edge_x_min, args.edge_x_max]
    result["upper_edge_before"] = visible_upper_edge_residual(real_mask, first, args.edge_x_min, args.edge_x_max)
    result["upper_edge_after"] = visible_upper_edge_residual(real_mask, last, args.edge_x_min, args.edge_x_max)
    result["status"] = "VISUAL_CANDIDATE_NOT_3D_CALIBRATION"
    result["model_target"] = "r2s_preview_table_and_four_legs"
    result["matching_episode_pose"] = args.episode if recorded_first is not None else None
    result["matching_object_layout_json"] = str(args.object_layout_json) if layout is not None else None
    result["table_yaw_deg_unchanged"] = not args.fit_yaw
    result["table_dimensions_unchanged"] = True
    result["recommended_video_flags"] = {
        "--table-extra-left-cm": 4.0,
        "--table-extra-front-cm": 5.0,
        "--table-base-x-cm": round(result["base_delta_xyz_m"][0] * 100., 3),
        "--table-base-y-cm": round(result["base_delta_xyz_m"][1] * 100., 3),
        "--table-base-z-cm": round(result["base_delta_xyz_m"][2] * 100., 3),
        "--table-yaw-delta-deg": round(result["yaw_delta_deg"], 3),
        "--anchor-robot-edge": result["robot_anchor_edge"],
        "--robot-anchor-x-cm": args.robot_anchor_x_cm,
        "--robot-anchor-y-cm": args.robot_anchor_y_cm,
        "--robot-edge-inset-cm": args.robot_edge_inset_cm,
    }
    result["warnings"] = []
    if result["robot_anchor_edge"] == "off":
        result["warnings"] = ["Robot/table edge constraint is disabled; the simulated robot may float away from the table edge."]
    else:
        result["warnings"] = ["Robot anchor is assumed at configured SO-101 base XY; verify actual clamp reference before treating pose as calibrated."]
    if result["search_bound_hit"]["xyz"] or result["search_bound_hit"]["yaw"]:
        result["warnings"].append("Optimizer reached search boundary; pose is NOT validated.")
    if abs(result["after"]["area_ratio_sim_over_real"] - 1.) > .10:
        result["warnings"].append("Projected tabletop area still differs from SAM3 by >10%.")
    if result["after"]["bbox_max_abs_error_px"] > 15:
        result["warnings"].append("Projected tabletop bounding edges still differ by >15 px.")
    if (result["upper_edge_after"]["sample_columns"] < 20):
        result["warnings"].append("Insufficient shared columns in selected upper-edge strip; change --edge-x-min/max.")
    elif result["upper_edge_after"]["mean_abs_error_px"] > 8:
        result["warnings"].append("Visible upper tabletop edge differs by >8px in the selected strip.")
    if result["warnings"]:
        result["status"] = "NOT_ALIGNED_REVIEW_MASK_AND_SCENE"
    else:
        result["status"] = "VISUAL_ENVELOPE_MATCH_CANDIDATE_UNVERIFIED"
    result["warnings"].append(
        "SAM3 mask has occlusion holes and simulated robot pose differs; IoU is NOT a reliable 3D depth/roll estimator.")
    result["warnings"].append(
        "Only selected base axes, optional tabletop yaw and optional robot-edge constraint are fitted; camera, table dimensions, robot, and props stay fixed.")
    args.output.mkdir(parents=True)
    Image.fromarray(real_mask.astype("uint8") * 255).save(args.output/"real_mask.png")
    overlay(original, real_mask, first).save(args.output/"before.png")
    overlay(original, real_mask, last).save(args.output/"after.png")
    (args.output/"report.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
