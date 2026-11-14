#!/usr/bin/env python3
"""Audit source STL nonconvexity vs MuJoCo convex mesh collision approximation.

This deliberately keeps the original STL coordinate system separate from
MuJoCo world-frame contacts: the STL cannot be overlaid on a world contact
without applying URDF origin/mesh transforms. Volume ratio is a diagnostic
of convexification, NOT proof that a specific contact is an artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

import real_to_sim as r2s


MESHES = {
    "geom0": "base_motor_holder_so101_v1.stl",
    "geom1": "base_so101_v2.stl",
    "geom2": "sts3215_03a_v1.stl",
    "geom6": "rotation_pitch_so101_v1.stl",
}


def inspect_stl(path):
    mesh = trimesh.load_mesh(path, process=True, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Not a triangle STL mesh: {path}")
    hull = mesh.convex_hull
    src_volume = float(abs(mesh.volume)) if mesh.is_volume else None
    hull_volume = float(hull.volume)
    bbox = np.asarray(mesh.bounds, dtype=float)
    report = {
        "filename": path.name,
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "bounds_in_stl_units": bbox.tolist(),
        "extents_in_stl_units": mesh.extents.tolist(),
        "watertight": bool(mesh.is_watertight),
        "is_valid_volume": bool(mesh.is_volume),
        "source_volume_in_stl_units_cubed": src_volume,
        "convex_hull_volume_in_stl_units_cubed": hull_volume,
        "hull_to_source_volume_ratio": (
            hull_volume / src_volume if src_volume is not None and src_volume > 0 else None),
        "euler_number": int(mesh.euler_number),
        "convex_hull_vertices": int(len(hull.vertices)),
        "convex_hull_triangles": int(len(hull.faces)),
    }
    return report, mesh, hull


def save_plot(mesh, hull, label, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(11, 5))
    ax1, ax2 = fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")
    for ax, obj, title in ((ax1, mesh, "Original STL vertices"),
                            (ax2, hull, "Convex hull envelope")):
        xyz = np.asarray(obj.vertices)
        stride = max(1, len(xyz) // 2500)
        ax.scatter(xyz[::stride, 0], xyz[::stride, 1], xyz[::stride, 2],
                   s=1.5, alpha=.40)
        if ax is ax2:
            tri = xyz[np.asarray(obj.faces, dtype=int)]
            edges = np.concatenate((tri[:, [0,1]], tri[:, [1,2]], tri[:, [2,0]]))
            # Plot only a subset of edges to bound rendering cost.
            edges = edges[::max(1,len(edges)//2500)]
            ax.add_collection3d(Line3DCollection(edges, colors="gray", linewidths=.35, alpha=.4))
        mid = mesh.bounds.mean(axis=0)
        radius = max(mesh.extents) * .65
        ax.set_xlim(mid[0]-radius, mid[0]+radius)
        ax.set_ylim(mid[1]-radius, mid[1]+radius)
        ax.set_zlim(mid[2]-radius, mid[2]+radius)
        ax.set_title(title)
        ax.set_xlabel("STL X")
        ax.set_ylabel("STL Y")
        ax.set_zlabel("STL Z")
        ax.view_init(elev=25, azim=42)
    fig.suptitle(f"{label}: raw STL geometry and convex hull (STL local frame)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def run(out, plot=True):
    if out.exists():
        raise ValueError(f"Output exists: {out}")
    root = r2s.DEFAULT_URDF.parent / "assets"
    results = {}
    # Compute before creating output folder to avoid incomplete runs.
    meshes = {}
    for geom, name in MESHES.items():
        results[geom], mesh, hull = inspect_stl(root / name)
        meshes[geom] = (mesh, hull)
    out.mkdir(parents=True)
    if plot:
        for geom, (mesh, hull) in meshes.items():
            save_plot(mesh, hull, geom, out / f"{geom}_stl_vs_hull.png")
    report = {
        "status": "SOURCE_STL_CONVEXITY_AUDIT_NOT_VERIFIED_COLLISION_CLEARANCE",
        "source_urdf": str(r2s.DEFAULT_URDF),
        "meshes": results,
        "notes": [
            "STL local coordinate and scale are not MuJoCo world coordinates.",
            "Plot pairs each original STL with its hull in local coordinates; do not overlay with world contact points.",
            "Volume ratios are only valid for watertight oriented volume meshes.",
            "MuJoCo contact distances remain separate evidence; concavity alone does not prove cause.",
            "No URDF or hardware is modified.",
        ],
    }
    (out / "stl_convexity.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    report = run(args.out, plot=not args.no_plots)
    print("Saved", args.out / "stl_convexity.json")
    for geom, item in report["meshes"].items():
        print(f"{geom} {item['filename']} watertight={item['watertight']} "
              f"volume_valid={item['is_valid_volume']} "
              f"hull/mesh_vol={item['hull_to_source_volume_ratio']} "
              f"triangles={item['triangles']}")


if __name__ == "__main__":
    main()
