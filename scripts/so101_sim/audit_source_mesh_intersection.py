#!/usr/bin/env python3
"""Step A final check: actual watertight source-STL solids overlap in 3D.

World-aligned original STL meshes are boolean intersected with manifold3d.
Unlike convex hull overlap, a positive intersection volume of the *source*
STL solids establishes actual source-geometry interference at the zero pose.
This does NOT imply a real-world robot is misassembled; meshes/calibration may
differ from actual hardware. Never modify collision filters or run hardware.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh

from step_a_surface_overlay import inspect_faces, PAIRS


def volume_interference(mesh_a, mesh_b):
    """Return actual source-STL solid overlap, or propagate a boolean failure."""
    for name, mesh in (("a", mesh_a), ("b", mesh_b)):
        if not mesh.is_watertight or not mesh.is_volume:
            raise ValueError(f"Source mesh {name} is not a valid watertight volume")
    overlap = trimesh.boolean.intersection([mesh_a, mesh_b], engine="manifold")
    if overlap is None:
        raise RuntimeError("Boolean engine produced no result")
    if not isinstance(overlap, trimesh.Trimesh):
        raise RuntimeError(f"Unexpected boolean result: {type(overlap).__name__}")
    v = float(abs(overlap.volume))
    if not np.isfinite(v):
        raise RuntimeError("Non-finite intersection volume")
    return overlap, v


def audit(out: Path, *, threshold_mm3=1.0, export_mesh=False):
    if out.exists():
        raise ValueError(f"Output directory exists: {out}")
    if threshold_mm3 <= 0:
        raise ValueError("threshold_mm3 must be positive")
    meshes, hulls, contacts, align = inspect_faces()
    rows, intersections = [], {}
    for a, b in PAIRS:
        source_intersection, volume_m3 = volume_interference(meshes[a], meshes[b])
        hull_intersection, hull_volume_m3 = volume_interference(hulls[a], hulls[b])
        intersections[(a, b)] = source_intersection
        rows.append({
            "geom_a": a, "geom_b": b,
            "source_stl_overlap_volume_mm3": volume_m3 * 1e9,
            "convex_hull_overlap_volume_mm3": hull_volume_m3 * 1e9,
            "source_stl_intersection_faces": int(len(source_intersection.faces)),
            "hull_intersection_faces": int(len(hull_intersection.faces)),
            "source_stl_volume_overlap_above_threshold":
                bool(volume_m3 * 1e9 > threshold_mm3),
            "zero_pose_mujoco_contacts":
                sum(c["pair"] == [a, b] for c in contacts),
        })

    out.mkdir(parents=True)
    with (out / "source_mesh_intersection.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if export_mesh:
        for (a, b), inter in intersections.items():
            if len(inter.faces):
                inter.export(out / f"geom{a}_{b}_actual_overlap.stl")
    report = {
        "status": "SOURCE_STL_BOOLEAN_SOLID_INTERSECTION_AT_ZERO_JOINT_POSE",
        "method": "trimesh.boolean.intersection engine=manifold; world-aligned source solids",
        "threshold_mm3": threshold_mm3,
        "alignment_p95_m": {str(k): float(v) for k, v in align.items()},
        "pairs": rows,
        "interpretation": [
            "Source-volume overlap > threshold: actual overlap of the supplied STL solids at zero joint pose.",
            "Source-volume overlap <= threshold: no appreciable volumetric overlap at chosen threshold; surface-only contact may still exist.",
            "Convex hull overlap > source overlap suggests convexification adds false occupied volume, but does not alone prove specific contact points are artifacts.",
            "STL meshes and URDF joint calibration can differ from real hardware.",
            "Only initial all-zero joint pose is evaluated; dynamic clearance across joint workspace is not verified.",
            "A boolean engine failure must not be silently treated as zero overlap.",
        ],
    }
    (out / "source_mesh_intersection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threshold-mm3", type=float, default=1.0)
    p.add_argument("--export-intersection-mesh", action="store_true")
    a = p.parse_args()
    report = audit(a.out, threshold_mm3=a.threshold_mm3,
                   export_mesh=a.export_intersection_mesh)
    print("Saved", a.out)
    for r in report["pairs"]:
        print(f"geom{r['geom_a']}-{r['geom_b']}: "
              f"source_overlap={r['source_stl_overlap_volume_mm3']:.3f} mm3, "
              f"convex_overlap={r['convex_hull_overlap_volume_mm3']:.3f} mm3, "
              f"source_interference={r['source_stl_volume_overlap_above_threshold']}")
    print("Solid volume overlap at zero pose only; not dynamic clearance or hardware validation.")


if __name__ == "__main__":
    main()
