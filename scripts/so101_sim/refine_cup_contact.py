#!/usr/bin/env python3
"""Refine only the clear cup's tabletop contact center in Episode 4 Side image.

Preserves all original Muhi 3-click data, records one new cup contact pixel.
Offline image annotation: does not move robot or alter raw dataset.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import compare_real_sim_motion as replay
import real_to_sim as r2s


def add_cup_contact(original, point):
    """Preserve Muhi and original cup visual center; add measured pixel."""
    if not isinstance(original, dict) or original.get("image_size") != [640, 480]:
        raise ValueError("Expected previous 640x480 three-click JSON")
    for key in ("muhi_center_px", "muhi_yaw_deg", "muhi_body_end_px",
                "muhi_cap_end_px", "cup_center_px"):
        if key not in original:
            raise ValueError(f"Missing original annotation {key}")
    p = np.asarray(point, dtype=float)
    if p.shape != (2,) or not np.isfinite(p).all() or not (0 <= p[0] < 640 and 0 <= p[1] < 480):
        raise ValueError("Invalid 640x480 cup contact point")
    result = dict(original)
    result["cup_contact_px"] = [round(float(x), 2) for x in p]
    result["cup_contact_point_description"] = "bottom-center contact with tabletop, NOT opening/rim center"
    result["cup_contact_reference_height_cm"] = 0.1
    result["method"] = "three_click_muhi_preserved_plus_one_cup_contact_click"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=4)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists, refusing overwrite")
    raw = args.source_json.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        if not raw.endswith("\\n"):
            raise
        payload = json.loads(raw[:-2])
    dataset = args.dataset_root.resolve(strict=True)
    if dataset == args.output.resolve() or dataset in args.output.resolve().parents:
        parser.error("Output must be outside dataset")
    episode = r2s.read_episode(dataset, args.episode, 1, 1)
    video, start = replay.side_video_source(dataset, episode)
    image = replay.read_frames_once(video, [start])[start].convert("RGB")
    image = image.resize((640, 480), Image.Resampling.BILINEAR)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(image, extent=(-.5, 639.5, 479.5, -.5))
    ax.set_xlim(-.5, 639.5)
    ax.set_ylim(479.5, -.5)
    ax.set_title("Click cup BOTTOM center where it contacts the desk (Esc: cancel)")
    saved = [False]
    def click(event):
        if event.inaxes is not ax or event.button != 1:
            return
        toolbar = getattr(getattr(event.canvas, "manager", None), "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return
        if event.xdata is None or event.ydata is None:
            return
        try:
            result = add_cup_contact(payload, (event.xdata, event.ydata))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        except ValueError as exc:
            ax.set_title(str(exc))
            fig.canvas.draw_idle()
            return
        print("Saved:", args.output)
        print("Cup contact point:", result["cup_contact_px"])
        saved[0] = True
        plt.close(fig)
    fig.canvas.mpl_connect("button_press_event", click)
    plt.show(block=True)
    if not saved[0]:
        print("Cancelled; no file written")


if __name__ == "__main__":
    main()
