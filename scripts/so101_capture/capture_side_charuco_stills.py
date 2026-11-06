#!/usr/bin/env python3
"""Interactive Side ChArUco stills using the verified OBS source; NO robot I/O.

Press Enter once per distinct board pose. Uses the same OBS source screenshot
request (JPEG at requested 640x480) as the ArUco dataset capture adapter.
Never changes OBS settings, camera position, source UUID, or existing files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from multi_camera import OBSClient, load_config, verify_source, capture_image_shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--execute", action="store_true",
                        help="Connects only to OBS; never to SO-101 motor/controller")
    args = parser.parse_args()
    cfg = load_config(args.camera_config, enabled="side")
    h, w, _ = capture_image_shape(cfg)
    if (w, h) != (640, 480):
        parser.error(f"Expected exactly 640x480; got {w}x{h}")
    if args.frames < 1 or args.frames > 200:
        parser.error("frames must be between 1 and 200")
    spec = cfg["active_cameras"]["side"]
    if spec.get("provider") != "obs_source":
        parser.error("Side must be an OBS source")
    if args.out_dir.exists():
        parser.error(f"Output already exists; choose a new directory: {args.out_dir}")
    print(f"Verified config for Side OBS source {spec['source_name']!r}, 640x480")
    print("The SO-101 robot is never connected or commanded.")
    if not args.execute:
        print("Preflight only. Add --execute to connect to OBS and capture stills.")
        return
    import base64
    import io
    from PIL import Image
    import numpy as np
    client = OBSClient()
    try:
        identity = verify_source(client, "side", spec)
        args.out_dir.mkdir(parents=True, exist_ok=False)
        saved = []
        print("OBS connected. Position ChArUco board, then press Enter. 'q' ends.")
        for i in range(args.frames):
            instruction = input(f"[{i+1}/{args.frames}] Enter to capture / q to finish: ").strip().lower()
            if instruction == "q":
                break
            response = client.request("GetSourceScreenshot", {
                "sourceName": spec["source_name"], "imageFormat": "jpg",
                "imageWidth": w, "imageHeight": h,
                "imageCompressionQuality": spec.get("jpeg_quality", 90)})
            image_bytes = base64.b64decode(response["imageData"].split(",", 1)[1], validate=True)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            if image.size != (w, h):
                raise RuntimeError("OBS returned incorrect image size")
            out = args.out_dir / f"side_charuco_{i:03d}.png"
            image.save(out)
            saved.append(out.name)
            print("Saved", out, "pixels", image.size)
        meta = {
            "capture": "OBS source screenshot", "image_size": [w, h],
            "source_name": spec["source_name"],
            "image_format_from_obs": "jpeg",
            "saved_files": saved,
            "note": "No robot I/O. OBS source transform and optics must stay unchanged. Source UUID and credentials not copied to metadata."
        }
        (args.out_dir / "capture_summary.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"Captured {len(saved)} images to {args.out_dir}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
