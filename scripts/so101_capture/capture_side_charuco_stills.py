#!/usr/bin/env python3
"""Capture ChArUco stills from verified OBS Side source, no SO-101 motor I/O.

Default: manual Enter per frame. With --interval-seconds N: timed unattended
capture after --start-delay-seconds countdown. Uses OBS GetSourceScreenshot
at requested 640x480, same as the ArUco acquisition pathway.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
from pathlib import Path
import time

from multi_camera import OBSClient, load_config, verify_source, capture_image_shape


def capture_frame(client, source, width, height, quality, destination):
    from PIL import Image
    response = client.request("GetSourceScreenshot", {
        "sourceName": source, "imageFormat": "jpg",
        "imageWidth": width, "imageHeight": height,
        "imageCompressionQuality": quality})
    image_bytes = base64.b64decode(response["imageData"].split(",", 1)[1], validate=True)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if image.size != (width, height):
        raise RuntimeError(f"OBS returned unexpected image size {image.size}")
    image.save(destination)


def play_shutter_sound():
    """Play a short Mac system sound after successful save, fail open on audio issues."""
    if sys.platform != "darwin":
        print("Sound cue skipped: macOS only.", flush=True)
        return
    sound = Path("/System/Library/Sounds/Pop.aiff")
    if not sound.is_file():
        print("Sound cue unavailable: Pop.aiff not found.", flush=True)
        return
    try:
        subprocess.run(["/usr/bin/afplay", str(sound)],
                       check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=3)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"Sound cue unavailable ({type(exc).__name__}); recording continues.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--interval-seconds", type=float, default=None,
                        help="Capture every N seconds automatically, no keyboard input")
    parser.add_argument("--start-delay-seconds", type=float, default=8.0,
                        help="Delay before automatic capture to pick up the board")
    parser.add_argument("--no-sound", action="store_true",
                        help="Disable shutter sound after every successfully saved image")
    parser.add_argument("--execute", action="store_true",
                        help="OBS only; no SO-101 servo, camera move, or robot connection")
    args = parser.parse_args()
    if not args.camera_config.is_file():
        parser.error(f"Camera YAML not found: {args.camera_config}. "
                     "Use: find . -name 'cameras.aruco-640.local.yaml'")
    if args.frames < 1 or args.frames > 200:
        parser.error("frames must be in 1..200")
    if args.interval_seconds is not None and not (0.5 <= args.interval_seconds <= 60):
        parser.error("--interval-seconds must be between 0.5 and 60")
    if not 0 <= args.start_delay_seconds <= 120:
        parser.error("--start-delay-seconds must be between 0 and 120")
    cfg = load_config(args.camera_config, enabled="side")
    h, w, _ = capture_image_shape(cfg)
    if (w, h) != (640, 480):
        parser.error(f"Expected 640x480, found {w}x{h}")
    spec = cfg["active_cameras"]["side"]
    if spec.get("provider") != "obs_source":
        parser.error("Side must be an OBS source")
    if args.out_dir.exists():
        parser.error(f"Output already exists: {args.out_dir}; choose a new directory")
    print(f"Verified Side OBS source {spec['source_name']!r} at 640x480", flush=True)
    print("SO-101 robot is NEVER connected or commanded.", flush=True)
    print("Sound after each saved image:", "off" if args.no_sound else "on (Mac Pop)", flush=True)
    if not args.execute:
        print("Preflight only. Add --execute to connect to OBS.", flush=True)
        return
    client = OBSClient()
    try:
        verify_source(client, "side", spec)
        args.out_dir.mkdir(parents=True, exist_ok=False)
        saved = []
        try:
            if args.interval_seconds is not None:
                print(f"Automatic capture: {args.frames} frames at "
                      f"{args.interval_seconds:g}s intervals; starts after "
                      f"{args.start_delay_seconds:g}s.", flush=True)
                time.sleep(args.start_delay_seconds)
                start = time.monotonic()
            else:
                print("Manual capture: Enter to take photo, q to finish.", flush=True)
            for i in range(args.frames):
                if args.interval_seconds is None:
                    instruction = input(f"[{i+1}/{args.frames}] Enter / q: ").strip().lower()
                    if instruction == "q":
                        break
                else:
                    due = start + i * args.interval_seconds
                    time.sleep(max(0.0, due - time.monotonic()))
                dest = args.out_dir / f"side_charuco_{i:03d}.png"
                capture_frame(client, spec["source_name"], w, h,
                              spec.get("jpeg_quality", 90), dest)
                saved.append(dest.name)
                print(f"[{i+1}/{args.frames}] Saved {dest}", flush=True)
                if not args.no_sound:
                    play_shutter_sound()
        except KeyboardInterrupt:
            print("\nInterrupted by user; existing images retained.", flush=True)
        finally:
            meta = {
                "capture": "OBS source screenshot", "image_size": [w, h],
                "source_name": spec["source_name"],
                "image_format_from_obs": "jpeg",
                "mode": "timed" if args.interval_seconds is not None else "manual",
                "interval_seconds": args.interval_seconds,
                "sound_enabled": not args.no_sound,
                "sound_timing": "after successful file save",
                "saved_files": saved,
                "note": "No robot I/O. OBS source geometry must remain unchanged. "
                        "Camera UUID/password deliberately omitted."
            }
            (args.out_dir / "capture_summary.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            print(f"Captured {len(saved)} images into {args.out_dir}", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
