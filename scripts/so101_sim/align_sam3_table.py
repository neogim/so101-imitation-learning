#!/usr/bin/env python3
"""Offline SAM3-to-MuJoCo table alignment. Never accesses robot hardware.

Only adjusts the existing r2s_table world geometry XY position, retaining
physical dimensions and the camera. Re-rendering preserves perspective scale.
SAM3's real mask and MuJoCo's geometry mask must describe the SAME surface.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def mask_metrics(actual, projected):
    """IoU plus interpretable area/centroid residuals. Arrays are boolean."""
    import numpy as np

    actual = np.asarray(actual, dtype=bool)
    projected = np.asarray(projected, dtype=bool)
    if actual.shape != projected.shape or actual.ndim != 2:
        raise ValueError("Real and rendered masks must be equally sized 2D images")
    na, nb = int(actual.sum()), int(projected.sum())
    if not na or not nb:
        raise ValueError("Empty real or projected target mask")
    inter = int(np.logical_and(actual, projected).sum())
    union = int(np.logical_or(actual, projected).sum())
    ay, ax = np.argwhere(actual).mean(axis=0)
    by, bx = np.argwhere(projected).mean(axis=0)
    return {
        "iou": inter / union,
        "real_area_px": na,
        "sim_area_px": nb,
        "area_ratio_sim_over_real": nb / na,
        "centroid_delta_sim_minus_real_px": [float(bx - ax), float(by - ay)],
    }


def sam3_mask(image_path: Path, model_path: str, text_prompt: str, bbox):
    """Predict one mask. An explicit bbox is preferable if text is ambiguous."""
    import numpy as np

    if bbox:
        from ultralytics import SAM

        model = SAM(model_path)
        results = model.predict(source=str(image_path), bboxes=list(bbox), verbose=False)
    else:
        from ultralytics.models.sam import SAM3SemanticPredictor

        predictor = SAM3SemanticPredictor(
            overrides={"model": model_path, "task": "segment",
                       "mode": "predict", "conf": 0.25, "verbose": False}
        )
        predictor.set_image(str(image_path))
        results = predictor(text=[text_prompt])
    if not results or results[0].masks is None:
        raise ValueError("SAM3 found no mask; supply --bbox or a clearer prompt")
    masks = results[0].masks.data.cpu().numpy() > 0.5
    if masks.shape[0] != 1:
        raise ValueError(
            f"SAM3 returned {masks.shape[0]} masks: use --bbox to specify ONE target"
        )
    return np.asarray(masks[0], dtype=bool)


def render_target(renderer, model, data, geom_id, camera):
    import mujoco
    import numpy as np

    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
    segmentation = renderer.render()
    # MuJoCo segmentation channels: [object_id, object_type].
    return np.logical_and(
        segmentation[:, :, 0] == geom_id,
        segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM),
    )


def overlay(real_image, real_mask, sim_mask):
    """Green: shared area; red: real only; blue: simulation only."""
    import numpy as np
    from PIL import Image

    rgb = np.asarray(real_image.convert("RGB")).copy()
    real_only = real_mask & ~sim_mask
    sim_only = sim_mask & ~real_mask
    both = real_mask & sim_mask
    rgb[both] = [45, 185, 90]
    rgb[real_only] = [235, 65, 65]
    rgb[sim_only] = [60, 105, 235]
    return Image.fromarray(rgb)


def optimize_xy(model, data, renderer, real_mask, geom_id, camera,
                initial_step_m=0.04, rounds=4, max_shift_m=0.20):
    """Deterministic coarse-to-fine search with fixed size/camera and XY bounds."""
    import numpy as np

    if initial_step_m <= 0 or rounds < 1 or max_shift_m <= 0:
        raise ValueError("Invalid search bounds")
    start = np.asarray(model.geom_pos[geom_id, :2], dtype=float).copy()
    best = start.copy()

    def evaluate(xy):
        model.geom_pos[geom_id, :2] = xy
        mask = render_target(renderer, model, data, geom_id, camera)
        if not mask.any():
            return float("-inf")
        # IoU rewards BOTH alignment and perspective-induced apparent size.
        intersect = np.logical_and(real_mask, mask).sum()
        union = np.logical_or(real_mask, mask).sum()
        return float(intersect / union) if union else float("-inf")

    best_score = evaluate(best)
    before_score = best_score
    for level in range(rounds):
        step = initial_step_m / (2 ** level)
        candidates = [best + np.array([dx, dy]) * step
                      for dx in (-2, -1, 0, 1, 2)
                      for dy in (-2, -1, 0, 1, 2)]
        level_best, level_score = best.copy(), best_score
        for candidate in candidates:
            if np.any(np.abs(candidate - start) > max_shift_m + 1e-9):
                continue
            score = evaluate(candidate)
            if score > level_score:
                level_best, level_score = candidate.copy(), score
        best, best_score = level_best, level_score

    model.geom_pos[geom_id, :2] = best
    return start.tolist(), best.tolist(), before_score, best_score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-image", required=True, type=Path,
                   help="Original SIDE-camera image, not the side-by-side composite")
    p.add_argument("--real-mask", type=Path,
                   help="Optional pre-reviewed SAM3 binary mask PNG; skips SAM3 inference")
    p.add_argument("--sam3-model", default="sam3.pt")
    p.add_argument("--prompt", default="wooden table")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
                   help="SAM3 image-space box prompt for exactly one table")
    p.add_argument("--config", type=Path,
                   default=Path(__file__).with_name("workcell.example.json"))
    p.add_argument("--urdf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--initial-step-m", type=float, default=0.04)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--max-shift-m", type=float, default=0.20)
    args = p.parse_args()

    from PIL import Image
    import numpy as np
    import mujoco

    if args.output.exists():
        p.error("Refusing to overwrite existing output directory")
    real_image = Image.open(args.real_image).convert("RGB")
    width, height = real_image.size
    if args.real_mask:
        mask_image = Image.open(args.real_mask).convert("L")
        if mask_image.size != real_image.size:
            p.error("Mask and original image dimensions must match exactly")
        real_mask = np.asarray(mask_image) > 127
    else:
        real_mask = sam3_mask(args.real_image, args.sam3_model, args.prompt, args.bbox)
        if real_mask.shape != (height, width):
            p.error("SAM3 output dimensions must match original image; no implicit resize")
    if not real_mask.any() or real_mask.all():
        p.error("Real mask must contain foreground and background")

    # Load the same model and workcell as the existing read-only replay workflow.
    from real_to_sim import build_model, load_config

    config = load_config(args.config)
    model = build_model(args.urdf, config)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r2s_table")
    if geom_id < 0:
        p.error("No r2s_table geometry in MuJoCo model")
    camera = "r2s_side"

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.enable_segmentation_rendering()
        before_mask = render_target(renderer, model, data, geom_id, camera)
        if not before_mask.any():
            p.error("Target table is not visible in side camera; review camera configuration")
        before = mask_metrics(real_mask, before_mask)
        original, adjusted, _, _ = optimize_xy(
            model, data, renderer, real_mask, geom_id, camera,
            args.initial_step_m, args.rounds, args.max_shift_m
        )
        after_mask = render_target(renderer, model, data, geom_id, camera)
        after = mask_metrics(real_mask, after_mask)

    revised = copy.deepcopy(config)
    revised["objects"]["table"]["pos_m"][:2] = adjusted
    revised["calibrated"] = False  # mask overlap does not establish 3D calibration
    revised["objects"]["table"]["calibration"] = "SAM3_SILHOUETTE_FIT_UNVERIFIED"
    report = {
        "status": "OFFLINE_IMAGE_ALIGNMENT_NOT_METRIC_CALIBRATION",
        "target": "r2s_table",
        "camera": camera,
        "original_xy_m": original,
        "adjusted_xy_m": adjusted,
        "shift_xy_m": [adjusted[i] - original[i] for i in (0, 1)],
        "before": before,
        "after": after,
        "warning": "Camera remains fixed; perspective scaling emerges from real 3D projection. "
                   "Original tabletop mask MUST correspond to r2s_table. "
                   "Do not use this result for robot motion or grasp safety.",
    }
    args.output.mkdir(parents=True)
    Image.fromarray(real_mask.astype("uint8") * 255).save(args.output / "real_sam3_mask.png")
    Image.fromarray(after_mask.astype("uint8") * 255).save(args.output / "sim_table_mask.png")
    overlay(real_image, real_mask, before_mask).save(args.output / "overlay_before.png")
    overlay(real_image, real_mask, after_mask).save(args.output / "overlay_after.png")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "workcell.candidate.json").write_text(json.dumps(revised, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
