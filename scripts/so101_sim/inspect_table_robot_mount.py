"""Inspect table/robot relative geometry without modifying Real2Sim alignment.

Usage: python scripts/so101_sim/inspect_table_robot_mount.py
  --extrinsics outputs/.../result.json --intrinsics assets/...json
  --table-base-x-cm 11 --table-base-y-cm 6.25 --table-yaw-delta-deg -2.5
  --output outputs/real2sim/table-mount-inspection

No real clamp measurement is implied by the URDF base origin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

import compare_real_sim_motion as movie


def analyze(model, anchor_xy=(0.0, 0.0)):
    ids = movie.table_geom_ids(model)
    table = ids[0]
    center = model.geom_pos[table].copy()
    quat = model.geom_quat[table].copy()
    rot = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()[:2, :2]
    half = model.geom_size[table, :2].copy()
    anchor = np.asarray(anchor_xy, dtype=float)
    if anchor.shape != (2,) or not np.isfinite(anchor).all():
        raise ValueError("Anchor must have two finite base coordinates")
    local = rot.T @ (anchor - center[:2])
    edges = {
        "x+": float(half[0] - local[0]),
        "x-": float(half[0] + local[0]),
        "y+": float(half[1] - local[1]),
        "y-": float(half[1] + local[1]),
    }
    # A sign is meaningful along that particular edge normal only.
    # Projected segment boundaries matter separately for finite tabletop edges.
    edge_segments = {}
    for name, (a,b) in {
        "x+": ([half[0],-half[1]], [half[0],half[1]]),
        "x-": ([-half[0],-half[1]], [-half[0],half[1]]),
        "y+": ([-half[0],half[1]], [half[0],half[1]]),
        "y-": ([-half[0],-half[1]], [half[0],-half[1]]),
    }.items():
        edge_segments[name] = [(center[:2] + rot @ np.asarray(p)).tolist()
                               for p in (a,b)]
    inside = bool(np.all(np.abs(local) <= half + 1e-9))
    return {
        "status": "DIAGNOSTIC_ONLY_NOT_REAL_CLAMP_CALIBRATION",
        "table_half_extents_m": half.tolist(),
        "table_center_base_xyz_m": center.tolist(),
        "table_quat_wxyz": quat.tolist(),
        "robot_anchor_base_xy_m": anchor.tolist(),
        "robot_anchor_table_local_xy_m": local.tolist(),
        "robot_anchor_inside_table_footprint": inside,
        "edge_signed_inward_distance_m": edges,
        "table_edges_in_base_xy_m": edge_segments,
        "note": "Robot anchor defaults to URDF base origin, not actual measured clamp contact. Edge distance is measured perpendicular to the infinite edge line; a finite segment may be farther.",
    }


def draw_diagnostic(report, output):
    out = Image.new("RGB", (900, 760), (245, 247, 250))
    d = ImageDraw.Draw(out)
    center = np.asarray(report["table_center_base_xyz_m"][:2])
    anchor = np.asarray(report["robot_anchor_base_xy_m"])
    edgepts = np.asarray([p for edge in report["table_edges_in_base_xy_m"].values()
                          for p in edge])
    allpoints = np.vstack([edgepts, anchor])
    minxy = allpoints.min(axis=0) - .12
    maxxy = allpoints.max(axis=0) + .12
    scale = min(720 / max(maxxy[0] - minxy[0], 1e-5),
                590 / max(maxxy[1] - minxy[1], 1e-5))
    def pix(xy):
        x, y = xy
        return (int(90 + (x - minxy[0]) * scale),
                int(680 - (y - minxy[1]) * scale))
    edges = report["table_edges_in_base_xy_m"]
    corners = [edges["x-"][0], edges["x+"][0], edges["x+"][1], edges["x-"][1]]
    d.polygon([pix(v) for v in corners], fill=(207, 164, 117), outline=(90, 65, 40), width=4)
    for name, pts in edges.items():
        a, b = pix(pts[0]), pix(pts[1])
        d.line([a, b], fill=(50, 72, 105), width=5)
        midpoint = (np.asarray(pts[0]) + np.asarray(pts[1])) / 2
        d.text(pix(midpoint), name, fill=(0, 0, 0), stroke_width=2, stroke_fill=(255,255,255))
    ax, ay = pix(anchor)
    d.ellipse((ax-9, ay-9, ax+9, ay+9), fill=(215, 43, 42))
    d.text((ax+12, ay-22), "SO-101 BASE ORIGIN (NOT CLAMP)", fill=(150, 25, 25))
    cx, cy = pix(center)
    d.ellipse((cx-5, cy-5, cx+5, cy+5), fill=(35, 60, 130))
    d.text((18, 18), "TABLE / ROBOT BASE XY - DIAGNOSTIC ONLY", fill=(25, 25, 25))
    d.text((18, 42), "Real clamp contact position is not measured.", fill=(110, 35, 35))
    d.text((18, 67), "Table footprint = 800 x 510 mm (rounded corners not modeled).", fill=(60,60,60))
    d.line([pix(anchor), pix(anchor + [0.10, 0])], fill=(200, 50, 40), width=3)
    d.text(pix(anchor + [0.11, 0]), "+base X", fill=(200, 50, 40))
    d.line([pix(anchor), pix(anchor + [0, 0.10])], fill=(35, 120, 75), width=3)
    d.text(pix(anchor + [0, 0.11]), "+base Y", fill=(35, 120, 75))
    out.save(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extrinsics", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path, required=True)
    p.add_argument("--urdf", type=Path, default=movie.DEFAULT_URDF)
    p.add_argument("--table-extra-left-cm", type=float, default=4.)
    p.add_argument("--table-extra-front-cm", type=float, default=5.)
    p.add_argument("--table-base-x-cm", type=float, default=11.)
    p.add_argument("--table-base-y-cm", type=float, default=6.25)
    p.add_argument("--table-base-z-cm", type=float, default=0.)
    p.add_argument("--table-yaw-delta-deg", type=float, default=-2.5)
    p.add_argument("--robot-anchor-x-cm", type=float, default=0.)
    p.add_argument("--robot-anchor-y-cm", type=float, default=0.)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Output already exists; refusing overwrite")
    extr = json.loads(args.extrinsics.read_text())
    intr = json.loads(args.intrinsics.read_text())
    model, _ = movie.make_calibrated_model(
        extr, intr, args.urdf, table_extra_left_m=args.table_extra_left_cm/100.,
        table_extra_front_m=args.table_extra_front_cm/100.)
    ids = movie.table_geom_ids(model)
    movie.set_table_pose_delta(
        model, model.geom_pos[ids].copy(), model.geom_quat[ids].copy(),
        [args.table_base_x_cm/100., args.table_base_y_cm/100., args.table_base_z_cm/100.],
        args.table_yaw_delta_deg)
    report = analyze(model, (args.robot_anchor_x_cm/100., args.robot_anchor_y_cm/100.))
    report["scene_flags"] = {
        "table_extra_left_cm": args.table_extra_left_cm,
        "table_extra_front_cm": args.table_extra_front_cm,
        "table_base_xyz_cm": [args.table_base_x_cm, args.table_base_y_cm, args.table_base_z_cm],
        "table_yaw_delta_deg": args.table_yaw_delta_deg,
    }
    args.output.mkdir(parents=True)
    (args.output / "table_robot_geometry.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    draw_diagnostic(report, args.output / "table_robot_topview.png")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
