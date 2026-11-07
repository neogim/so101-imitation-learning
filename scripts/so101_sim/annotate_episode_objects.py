#!/usr/bin/env python3
"""Manually mark Muhi endpoints and clear cup center in one recorded Side frame.

Three clicks, one JSON; no SAM3, CUDA, hardware control or dataset modification.
The saved values use the same 640x480 resize as compare_real_sim_motion.py.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import compare_real_sim_motion as replay
import real_to_sim as r2s


def targets_from_clicks(points, T_base_cv, intrinsics):
    """(body-end, cap-end, cup-center) 2D clicks -> yaw in robot base XY."""
    p = np.asarray(points, dtype=float)
    if p.shape != (3, 2) or not np.isfinite(p).all():
        raise ValueError("Three finite 2D points required")
    if not (np.all((p[:, 0] >= 0) & (p[:, 0] < 640)) and
            np.all((p[:, 1] >= 0) & (p[:, 1] < 480))):
        raise ValueError("Click positions outside 640x480 Side frame")
    body = replay.pixel_to_base_plane(T_base_cv, intrinsics, *p[0], .019)
    cap = replay.pixel_to_base_plane(T_base_cv, intrinsics, *p[1], .019)
    axis = cap[:2] - body[:2]
    if float(np.linalg.norm(axis)) < .003:
        raise ValueError("Muhi endpoints too close")
    # Model long axis for 0-deg yaw is +base Y (toward cap).
    yaw = math.degrees(math.atan2(-axis[0], axis[1]))
    center = (p[0] + p[1]) / 2
    return {
        "image_size": [640, 480],
        "muhi_center_px": [round(float(x), 2) for x in center],
        "cup_center_px": [round(float(x), 2) for x in p[2]],
        "muhi_yaw_deg": round(yaw, 3),
        "muhi_body_end_px": [round(float(x), 2) for x in p[0]],
        "muhi_cap_end_px": [round(float(x), 2) for x in p[1]],
        "method": "three_manual_clicks_on_recorded_side_video",
        "note": "Approximate projection at fixed heights using candidate 2026-11-07 camera for an older recording",
    }



def valid_image_click(event, ax):
    """Accept only valid in-image left clicks, ignoring active zoom/pan tools."""
    from matplotlib.backend_bases import MouseButton
    if event.inaxes is not ax or event.button != MouseButton.LEFT:
        return False
    toolbar = getattr(getattr(event.canvas, "manager", None), "toolbar", None)
    if toolbar is not None and getattr(toolbar, "mode", ""):
        return False
    if event.xdata is None or event.ydata is None:
        return False
    return (np.isfinite(event.xdata) and np.isfinite(event.ydata)
            and 0 <= event.xdata < 640 and 0 <= event.ydata < 480)


def select_three_points(image, T_base_cv, intrinsics, output):
    """Click three in-image points; save on 3rd click, then close the window."""
    import matplotlib.pyplot as plt

    labels = ("Muhi BODY end", "Muhi CAP end", "clear CUP center")
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(image, extent=(-.5, 639.5, 479.5, -.5))
    ax.set_xlim(-.5, 639.5)
    ax.set_ylim(479.5, -.5)
    points, artists = [], []
    saved = [False]

    def status(message):
        ax.set_title(message)
        fig.canvas.draw_idle()

    def redraw():
        for artist in artists:
            artist.remove()
        artists.clear()
        for index, (x, y) in enumerate(points):
            artists.extend(ax.plot(x, y, marker="o", color="lime", markersize=8,
                                   markerfacecolor="none", markeredgewidth=2))
            artists.append(ax.text(x+5, y+5, str(index+1), color="lime"))
        if not saved[0]:
            status(f"Point {len(points)+1}/3: {labels[len(points)]} | Backspace: undo | Esc: cancel")

    def on_click(event):
        if saved[0] or not valid_image_click(event, ax):
            return
        points.append((float(event.xdata), float(event.ydata)))
        if len(points) == 3:
            try:
                result = targets_from_clicks(points, T_base_cv, intrinsics)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\\n",
                                  encoding="utf-8")
            except (ValueError, OSError) as exc:
                points.pop()
                redraw()
                status(f"Invalid third point: {exc} | Click again or Backspace")
                print(f"Could not save selection: {exc}")
                return
            saved[0] = True
            print(f"Saved: {output}")
            print("Muhi yaw in base XY:", result["muhi_yaw_deg"])
            plt.close(fig)
        else:
            redraw()

    def on_key(event):
        if saved[0]:
            return
        if event.key in ("backspace", "delete"):
            if points:
                points.pop()
            redraw()
        elif event.key in ("escape", "esc"):
            plt.close(fig)

    redraw()
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    print("Left-click on IMAGE: Muhi BODY end -> Muhi CAP end -> clear CUP center.")
    print("Only 3 valid image clicks count. Backspace=undo; Esc=cancel.")
    plt.show(block=True)
    if not saved[0]:
        print("Cancelled. No output was written.")
    return saved[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--extrinsics", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, default=replay.DEFAULT_INTRINSICS)
    parser.add_argument("--output", type=Path, required=True,
                        help="New path ending in .json; refuses overwrite")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output JSON exists; choose a new filename")
    dataset = args.dataset_root.resolve(strict=True)
    if dataset == args.output.resolve() or dataset in args.output.resolve().parents:
        parser.error("Output must be outside raw dataset root")
    episode = r2s.read_episode(dataset, args.episode, 1, 1)
    video, start = replay.side_video_source(dataset, episode)
    image = replay.read_frames_once(video, [start])[start].convert("RGB")
    image = image.resize((640, 480), Image.Resampling.BILINEAR)
    intrinsics = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    extrinsics = json.loads(args.extrinsics.read_text(encoding="utf-8"))
    if (intrinsics.get("width"), intrinsics.get("height")) != (640, 480):
        raise ValueError("Intrinsics must be for 640x480")
    select_three_points(image, extrinsics["T_base_camera"], intrinsics, args.output)

if __name__ == "__main__":
    main()
