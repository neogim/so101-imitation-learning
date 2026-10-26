#!/usr/bin/env python3
"""Independent, read-only side/wrist live viewer. NEVER opens motor ports.

Stop capture/ACT programs before launching: wrist AVFoundation is exclusively
owned by this process. Uses the same configured cameras as Dataset recording.
Displays latest frames without marking them consumed by recording. A viewer
failure does not send robot commands. Press q or Esc to close.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from multi_camera import CameraRig, load_config

def render(camera, label, target_size, stale_s):
    width, height = target_size
    with camera.lock:
        frame = None if camera.frame is None else camera.frame.copy()
        stamp = camera.timestamp
        err = camera.error
    now = time.perf_counter()
    age = None if stamp is None else max(0., now-stamp)
    fresh = err is None and age is not None and age <= stale_s
    if frame is None:
        img = Image.new("RGB", (width,height), (35,35,35))
    else:
        img = Image.fromarray(frame).resize((width,height), Image.Resampling.BILINEAR)
    strip=Image.new("RGB", (width,40), (16,16,16))
    draw=ImageDraw.Draw(strip)
    status=("ERROR: "+str(err)[:55] if err else
            "NO FRAME" if age is None else
            f"{age*1000:.0f} ms host age"+("" if fresh else " - STALE"))
    draw.text((8,5), f"{label.upper()} | {status}", fill=(100,230,150) if fresh else (255,160,90))
    out=Image.new("RGB", (width,height+40))
    out.paste(img,(0,40)); out.paste(strip,(0,0))
    return np.asarray(out)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-config",type=Path,default=Path("scripts/so101_capture/cameras.local.yaml"))
    parser.add_argument("--enable-cameras",default="side,wrist")
    parser.add_argument("--preview-fps",type=float,default=5)
    parser.add_argument("--stale-warning-ms",type=float,default=100)
    parser.add_argument("--tile-width",type=int,default=640)
    args=parser.parse_args()
    if not 0 < args.preview_fps <= 15 or not 160 <= args.tile_width <= 1280:
        parser.error("preview-fps must be in (0,15] and tile-width in [160,1280]")
    import cv2
    cfg=load_config(args.camera_config,enabled=args.enable_cameras)
    if len(cfg["active_cameras"]) != 2:
        parser.error("Select exactly two cameras")
    rig=CameraRig(cfg)
    print("Preview only: robot control is NOT connected. Close other wrist-camera users first.",flush=True)
    rig.connect()
    names=tuple(rig.cameras)
    width=args.tile_width
    height=round(width*rig.image_shape[0]/rig.image_shape[1])
    delay=1/args.preview_fps
    try:
        cv2.namedWindow("SO-101 side + wrist | q: quit",cv2.WINDOW_NORMAL)
        while True:
            start=time.perf_counter()
            frames=[render(rig.cameras[name],name,(width,height),args.stale_warning_ms/1000)
                    for name in names]
            rgb=np.concatenate(frames,axis=1)
            cv2.imshow("SO-101 side + wrist | q: quit",cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            key=cv2.waitKey(1)&0xff
            if key in (ord("q"),27): break
            # Keep UI rendering outside capture workers; never block their update loops.
            end=time.perf_counter()+max(0.,delay-(time.perf_counter()-start))
            while time.perf_counter()<end:
                if cv2.waitKey(10)&0xff in (ord("q"),27): return
    finally:
        rig.disconnect()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
