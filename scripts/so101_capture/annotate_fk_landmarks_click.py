#!/usr/bin/env python3
"""Manual SO-101 landmark annotator. Read-only images; writes only a new JSON.

Click only landmark positions that genuinely match the URDF link origins.
Skip invisible/ambiguous landmarks. No robot connection or motor commands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

LANDMARKS = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
    "gripper_frame_link",
)


def annotate(images: list[tuple[int, Path]], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    sizes = [Image.open(path).size for _, path in images]
    if len(set(sizes)) != 1:
        raise ValueError(f"All images must have identical dimensions: {sizes}")
    width, height = sizes[0]
    result = {"width": width, "height": height, "frames": []}

    print("Controls: LEFT CLICK to set current point; s=skip point; u=undo last point; "
          "ENTER=next frame; q=quit without saving.")
    print("Warning: visible motor casing centers are not necessarily URDF link origins.")
    for frame, path in images:
        pixels: dict[str, list[float]] = {}
        history: list[str] = []
        current = [0]
        quit_ = [False]
        fig, ax = plt.subplots(figsize=(12, 8))
        im = Image.open(path).convert("RGB")
        ax.imshow(im)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_title(f"Frame {frame}")
        annotation_artists = []

        def refresh() -> None:
            for artist in annotation_artists:
                artist.remove()
            annotation_artists.clear()
            for name, pt in pixels.items():
                annotation_artists.extend([
                    ax.scatter([pt[0]], [pt[1]], color="lime", marker="+", s=90),
                    ax.text(pt[0] + 5, pt[1] - 5, name, color="lime", fontsize=8,
                            bbox={"facecolor": "black", "alpha": 0.5}),
                ])
            if current[0] < len(LANDMARKS):
                target = LANDMARKS[current[0]]
                ax.set_title(f"Frame {frame}: click {target}  |  s skip / u undo / Enter next / q cancel")
            else:
                ax.set_title(f"Frame {frame}: completed | Enter next / u undo / q cancel")
            fig.canvas.draw_idle()

        def advance() -> None:
            if current[0] < len(LANDMARKS):
                current[0] += 1

        def click(event) -> None:
            if event.inaxes != ax or event.button != 1 or current[0] >= len(LANDMARKS):
                return
            if event.xdata is None or event.ydata is None:
                return
            name = LANDMARKS[current[0]]
            pixels[name] = [round(float(event.xdata), 2), round(float(event.ydata), 2)]
            history.append(name)
            advance()
            refresh()

        def key(event) -> None:
            if event.key == "s" and current[0] < len(LANDMARKS):
                history.append(LANDMARKS[current[0]])
                advance()
            elif event.key == "u" and history:
                name = history.pop()
                current[0] = LANDMARKS.index(name)
                pixels.pop(name, None)
            elif event.key == "enter":
                if len(pixels) < 2:
                    print(f"Frame {frame}: only {len(pixels)} point(s); skipping frame. Two or more recommended.")
                plt.close(fig)
                return
            elif event.key == "q":
                quit_[0] = True
                plt.close(fig)
                return
            refresh()

        fig.canvas.mpl_connect("button_press_event", click)
        fig.canvas.mpl_connect("key_press_event", key)
        refresh()
        plt.show(block=True)
        if quit_[0]:
            print("Cancelled; no file written.")
            return
        result["frames"].append({"frame": frame, "points": pixels})
        print(f"Frame {frame}: {len(pixels)} landmark(s) recorded")
    count = sum(len(row["points"]) for row in result["frames"])
    if count < 8:
        print(f"Only {count} landmarks; solver requires at least 8. Nothing written.")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {count} points in {len(images)} frames: {output}")
    print("Review coordinates manually before FK fitting; never guess a hidden link origin.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--frames", type=int, nargs="+", default=[0, 240, 480])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if len(args.frames) != len(set(args.frames)):
        p.error("Repeated frames")
    images = [(frame, args.images_dir / f"frame_{frame}.png") for frame in args.frames]
    for frame, path in images:
        if not path.is_file():
            p.error(f"Missing frame {frame}: {path}")
    annotate(images, args.output)


if __name__ == "__main__":
    main()
