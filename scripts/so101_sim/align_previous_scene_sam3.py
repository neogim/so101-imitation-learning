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


def offset_xy(T_base_camera, left_m, front_m):
    """Camera-relative ground-plane motion; negative front means away."""
    return scene.workcell_shift_left_front(T_base_camera, left_m, front_m)[:2]


def optimize(model, data, renderer, camera, table_geom_id, real_mask,
             T_base_camera, max_delta_m=.12, initial_step_m=.02, levels=4):
    import mujoco

    if max_delta_m <= 0 or initial_step_m <= 0 or levels <= 0:
        raise ValueError("Search parameters must be positive")
    part_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                for n in TABLE_PART_NAMES]
    if any(x < 0 for x in part_ids):
        raise ValueError("Previous preview tabletop/leg geometry not found")
    original = model.geom_pos[part_ids].copy()

    def trial(left, front):
        delta = offset_xy(T_base_camera, left, front)
        model.geom_pos[part_ids, :2] = original[:, :2] + delta
        candidate = segmentation_mask(renderer, model, data, camera, table_geom_id)
        if not candidate.any():
            return -1.
        return float((real_mask & candidate).sum() / (real_mask | candidate).sum())

    best_left, best_front = 0., 0.
    before = trial(0., 0.)
    best_score = before
    for level in range(levels):
        step = initial_step_m / (2 ** level)
        winner = (best_left, best_front)
        score = best_score
        for dl in (-2, -1, 0, 1, 2):
            for df in (-2, -1, 0, 1, 2):
                left, front = best_left + dl*step, best_front + df*step
                if abs(left) > max_delta_m or abs(front) > max_delta_m:
                    continue
                value = trial(left, front)
                if value > score:
                    winner, score = (left, front), value
        best_left, best_front = winner
        best_score = score
    trial(best_left, best_front)
    return {"left_m": best_left, "front_m": best_front,
            "iou_before": before, "iou_after": best_score}


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
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-delta-m", type=float, default=.12)
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

    # Keep camera pose, perspective, white follower, backdrop, and fill light
    # exactly as in the prior successful comparison branch.
    model, camera = movie.make_calibrated_model(
        extrinsics, intrinsics, args.urdf)
    data = mujoco.MjData(model)
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r2s_preview_table")
    if geom < 0:
        raise ValueError("No r2s_preview_table geom")
    with mujoco.Renderer(model, width=640, height=480) as renderer:
        renderer.enable_segmentation_rendering()
        first = segmentation_mask(renderer, model, data, camera, geom)
        if not first.any():
            raise ValueError("Simulated tabletop not visible with existing camera")
        result = optimize(model, data, renderer, camera, geom, real_mask,
                          extrinsics["T_base_camera"], max_delta_m=args.max_delta_m)
        last = segmentation_mask(renderer, model, data, camera, geom)
    result["before"] = overlap_metrics(real_mask, first)
    result["after"] = overlap_metrics(real_mask, last)
    result["existing_table_extra_left_cm"] = 4.
    result["existing_table_extra_front_cm"] = 5.
    result["suggested_table_extra_left_cm"] = 4. + result["left_m"] * 100
    result["suggested_table_extra_front_cm"] = 5. + result["front_m"] * 100
    result["interpretation"] = (
        "Negative front_m moves the tabletop AWAY from the camera; "
        "perspective makes it smaller. Fixed camera, fixed table size/yaw, "
        "same lighting and backdrop as previous successful video."
    )
    result["status"] = "UNVERIFIED_OFFLINE_SILHOUETTE_CANDIDATE_NOT_CALIBRATION"
    args.output.mkdir(parents=True)
    Image.fromarray(real_mask.astype("uint8") * 255).save(args.output/"real_mask.png")
    overlay(original, real_mask, first).save(args.output/"before.png")
    overlay(original, real_mask, last).save(args.output/"after.png")
    (args.output/"report.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
