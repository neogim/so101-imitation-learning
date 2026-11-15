#!/usr/bin/env python3
"""Overlay saved Real/Sim candidate PNGs, offline; does not drive robot or render physics.

Input: directory from compare_side_camera_pose_candidates.py, containing
summary.json and candidate_NN_frame_FFFF.png. The saved panel format is
two native-resolution images with a 78px header and 8px/16px offsets.

Outputs REAL, REAL with approximate sim-white-surface outline, and 50% blend.
Neither geometry alignment nor camera registration is automatically validated.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_overlay(source: Path, output: Path, alpha: float = 0.5) -> dict:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    metadata = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    w, h = (int(x) for x in metadata["video_resolution"])
    paths = sorted(source.glob("candidate_*_frame_*.png"))
    if not paths:
        raise FileNotFoundError("No candidate_NN_frame_FFFF.png in source")
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    output.mkdir(parents=True)
    panels, rows = [], []
    for path in paths:
        match = re.fullmatch(r"candidate_(\d+)_frame_(\d+)\.png", path.name)
        if not match:
            continue
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (2*w+24, h+78):
                raise ValueError(f"Unexpected panel layout {path}: {image.size}")
            real = np.asarray(image.crop((8, 70, 8+w, 70+h))).copy()
            sim = np.asarray(image.crop((w+16, 70, 2*w+16, 70+h))).copy()
        blended = np.uint8(np.clip(real*(1-alpha)+sim*alpha, 0, 255))
        # Visual-only heuristic for bright, approximately neutral robot meshes.
        rgb = sim.astype(np.int16)
        lo, hi = rgb.min(axis=2), rgb.max(axis=2)
        mask = (lo > 150) & ((hi-lo) < 55)
        edge = np.zeros(mask.shape, dtype=bool)
        edge[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]
        edge[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]
        edge[1:, :] |= mask[1:, :] & ~mask[:-1, :]
        edge[:-1, :] |= mask[:-1, :] & ~mask[1:, :]
        outlined = real.copy()
        outlined[edge] = [238, 35, 132]
        canvas = Image.new("RGB", (w*3+48, h+70), "white")
        for idx, arr in enumerate((real, outlined, blended)):
            canvas.paste(Image.fromarray(arr), (12 + idx*(w+12), 58))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 6), f"{path.name} | uncalibrated image alignment", fill="#283648")
        for idx, label in enumerate(("Real", "Real + simulated outline", "50% Real + Sim")):
            draw.text((12+idx*(w+12), 30), label, fill="#34465b")
        name = f"{path.stem}_overlay.png"
        canvas.save(output / name)
        rows.append({"candidate": int(match.group(1)), "frame": int(match.group(2)), "file": name})
        panels.append(canvas)
    if not panels:
        raise ValueError("No valid candidate panels")
    sheet = Image.new("RGB", (panels[0].width, sum(p.height for p in panels)+10*(len(panels)-1)), "#e9edf1")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height+10
    sheet.save(output / "all_frames_overlay.png")
    report = {
        "status": "UNREGISTERED_VISUAL_OVERLAY_NOT_CAMERA_OR_JOINT_CALIBRATION",
        "source": str(source), "video_resolution": [w, h], "alpha": alpha, "panels": rows,
        "limitations": [
            "Bright white foreground mask is approximate; tabletop/object edges may be omitted.",
            "No camera optimization or landmark measurement was performed.",
            "The historical candidate side-camera transform is not validated against the original recording.",
            "Do not infer J1/J2/J3 signs, TCP height or physical collision from this image alone."
        ]
    }
    (output / "overlay_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(f"Saved {len(panels)} overlay panels to {output}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.5)
    a = p.parse_args()
    make_overlay(a.input, a.output, a.alpha)


if __name__ == "__main__":
    main()
