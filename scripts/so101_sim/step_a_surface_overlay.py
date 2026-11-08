#!/usr/bin/env python3
"""Step A: Render actual STL TRIANGLE FACES vs convex-hull FACES and contacts.

All geometry is in the same MuJoCo world frame (zero joint pose). The convex
hull is computed from each world-aligned source STL. A convex hull generated
from vertices agrees with the convex-envelope geometry assumption but is not
a guarantee of identical MuJoCo internal collision faces/contact manifold.
This is a geometry diagnostic, NOT a certified STL-vs-STL intersection test.
No original files or physics settings are modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

import real_to_sim as r2s
from overlay_stl_mujoco_contacts import FILES, source_to_world
from so101_reach_env import SO101ReachEnv

PAIRS = ((0, 6), (1, 6), (2, 6))
IDS = (0, 1, 2, 6)
COLORS = {0: "#277da1", 1: "#f2a541", 2: "#4b9c7a", 6: "#9d62a7"}


def inspect_faces():
    env = SO101ReachEnv(seed=101)
    try:
        env.reset(seed=101)
        model, data = env.model, env.data
        source, hulls, alignment = {}, {}, {}
        for gid in IDS:
            stl = trimesh.load_mesh(r2s.DEFAULT_URDF.parent / "assets" / FILES[gid],
                                    process=True, force="mesh")
            mesh, _compiled_vertices, error = source_to_world(model, data, gid, stl)
            source[gid] = mesh
            hulls[gid] = mesh.convex_hull
            alignment[gid] = error
        contacts = []
        for i in range(data.ncon):
            c = data.contact[i]
            a, b = sorted((int(c.geom1), int(c.geom2)))
            if (a, b) in PAIRS:
                contacts.append({
                    "pair": [a, b],
                    "position_world_m": np.array(c.pos, dtype=float).tolist(),
                    "mujoco_signed_distance_mm": float(c.dist) * 1000,
                })
        if len(contacts) != 12:
            raise RuntimeError(f"Expected 12 initial base-shoulder contacts; got {len(contacts)}")
        return source, hulls, contacts, alignment
    finally:
        env.close()


def nearest_surface(mesh, points):
    # trimesh.nearest.on_surface needs rtree; naive is fine for 12 contact points.
    _closest, d, _triangles = trimesh.proximity.closest_point_naive(mesh, points)
    return np.asarray(d, dtype=float) * 1000


def compute_report(source, hulls, contacts, alignment):
    rows = []
    # ConvexHull equations classify contact points against the *closed*
    # envelope; unsigned distance alone cannot tell inside from outside.
    hull_planes = {
        gid: ConvexHull(np.asarray(source[gid].vertices)).equations
        for gid in IDS
    }
    for item in contacts:
        a, b = item["pair"]
        p = np.asarray(item["position_world_m"], dtype=float).reshape(1, 3)
        inside = {gid: bool(np.all(hull_planes[gid][:, :3] @ p[0]
                                        + hull_planes[gid][:, 3] <= 1e-8))
                  for gid in (a, b)}
        rows.append({
            "geom_a": a, "geom_b": b,
            "x_m": float(p[0, 0]), "y_m": float(p[0, 1]),
            "z_m": float(p[0, 2]),
            "mujoco_contact_distance_mm": item["mujoco_signed_distance_mm"],
            "raw_a_surface_mm": float(nearest_surface(source[a], p)[0]),
            "raw_b_surface_mm": float(nearest_surface(source[b], p)[0]),
            "hull_a_surface_mm": float(nearest_surface(hulls[a], p)[0]),
            "hull_b_surface_mm": float(nearest_surface(hulls[b], p)[0]),
            "inside_hull_a": inside[a],
            "inside_hull_b": inside[b],
        })
    by_pair = {}
    for a, b in PAIRS:
        subset = [r for r in rows if (r["geom_a"], r["geom_b"]) == (a, b)]
        by_pair[f"{a}-{b}"] = {
            name: float(np.mean([r[name] for r in subset]))
            for name in ("mujoco_contact_distance_mm", "raw_a_surface_mm",
                         "raw_b_surface_mm", "hull_a_surface_mm", "hull_b_surface_mm")
        }
    return {
        "status": "STEP_A_WORLD_SURFACE_OVERLAY_NOT_PHYSICAL_CLEARANCE_CERTIFICATION",
        "units": "world position m; surface distances mm",
        "alignment_p95_m": {str(k): float(v) for k, v in alignment.items()},
        "contact_rows": rows,
        "mean_by_pair": by_pair,
        "mesh_info": {
            str(g): {
                "raw_triangles": len(source[g].faces),
                "convex_hull_triangles": len(hulls[g].faces),
                "watertight": bool(source[g].is_watertight),
            } for g in IDS
        },
        "caveats": [
            "inside_hull flags test geometric convex envelope inclusion, not inclusion in source STL.",
            "Both surfaces are world-frame and depicted as TRIANGLE FACES; not just vertices.",
            "Convex hull surfaces come from source STL vertices; exact MuJoCo contact representation may differ.",
            "Distances to raw/hull surfaces are UNSIGNED; contact midpoint may lie away from both surfaces.",
            "No mesh-to-mesh intersection or physical running clearance is certified here.",
            "Never deploy proxy shapes or remove collision checks based only on this plot.",
        ],
    }


def _add_mesh(ax, mesh, color, alpha, max_triangles):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    faces = np.asarray(mesh.faces, dtype=int)
    # Deterministic subsampling for a quick visual; full mesh used for distances.
    stride = max(1, int(np.ceil(len(faces) / max_triangles)))
    tris = np.asarray(mesh.vertices)[faces[::stride]]
    poly = Poly3DCollection(tris, alpha=alpha, facecolor=color,
                            edgecolor="none", linewidth=0)
    ax.add_collection3d(poly)
    return len(tris)


def draw_panel(ax, source, hulls, contacts, gids, *, title, focus,
               max_triangles=3000):
    from matplotlib.lines import Line2D
    allpts = []
    for gid in gids:
        _add_mesh(ax, source[gid], COLORS[gid], .23, max_triangles)
        # Hull surface is a set of actual opaque-ish triangles; not a wireframe.
        _add_mesh(ax, hulls[gid], COLORS[gid], .07, max_triangles)
        allpts.append(np.asarray(source[gid].vertices))
    selected = [c for c in contacts if set(c["pair"]).issubset(gids)]
    if selected:
        cp = np.asarray([c["position_world_m"] for c in selected], dtype=float)
        ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2],
                   marker="x", c="red", s=75, linewidths=2, depthshade=False)
    if focus and selected:
        center = cp.mean(axis=0)
        radius = .038
    else:
        extent = np.concatenate(allpts, axis=0)
        center = (extent.min(axis=0) + extent.max(axis=0)) / 2
        radius = float(np.max(np.ptp(extent, axis=0)) * .59)
    ax.set(xlim=(center[0]-radius, center[0]+radius),
           ylim=(center[1]-radius, center[1]+radius),
           zlim=(center[2]-radius, center[2]+radius))
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("world X [m]"); ax.set_ylabel("world Y [m]")
    ax.set_zlabel("world Z [m]")
    ax.view_init(elev=23, azim=-68)
    ax.set_title(title, fontsize=11)
    legend = [Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS[g],
                     label=f"geom{g}: STL + transparent hull", markersize=10) for g in gids]
    legend.append(Line2D([0], [0], marker="x", linestyle="none",
                         color="red", label="MuJoCo contacts"))
    ax.legend(handles=legend, loc="upper left", fontsize=7)


def render(source, hulls, contacts, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for filename, gids, focus in (
        ("stl_vs_convex_surface_overlay.png", IDS, False),
        ("geom1_6_contact_closeup.png", (1, 6), True),
    ):
        fig = plt.figure(figsize=(12, 7))
        for n, azim in enumerate((-68, 25)):
            ax = fig.add_subplot(1, 2, n+1, projection="3d")
            draw_panel(ax, source, hulls, contacts, gids,
                       title=("Original STL faces + convex hull faces\n"
                              + ("focused on geom1-6" if focus else "all four parts")),
                       focus=focus)
            ax.view_init(elev=23 if n == 0 else 12, azim=azim)
        fig.suptitle("Step A: source STL faces (stronger) vs convex hull faces (faint) "
                     "and contacts\n"
                     "Both surfaces are in world coordinates; faces subsampled for drawing")
        fig.tight_layout()
        fig.savefig(out / filename, dpi=145)
        plt.close(fig)


def run(out):
    if out.exists():
        raise ValueError(f"Output directory already exists: {out}")
    source, hulls, contacts, alignment = inspect_faces()
    report = compute_report(source, hulls, contacts, alignment)
    out.mkdir(parents=True)
    with (out / "surface_gap_summary.csv").open("x", newline="") as f:
        cols = ["geom_a", "geom_b", "x_m", "y_m", "z_m",
                "mujoco_contact_distance_mm",
                "raw_a_surface_mm", "raw_b_surface_mm",
                "hull_a_surface_mm", "hull_b_surface_mm",
                "inside_hull_a", "inside_hull_b"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(report["contact_rows"])
    (out / "surface_overlay_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render(source, hulls, contacts, out)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = run(args.out)
    print("Saved", args.out)
    for pair, metrics in report["mean_by_pair"].items():
        print(f"pair {pair}: raw STL={metrics['raw_a_surface_mm']:.2f}/"
              f"{metrics['raw_b_surface_mm']:.2f}mm; "
              f"hull={metrics['hull_a_surface_mm']:.2f}/"
              f"{metrics['hull_b_surface_mm']:.2f}mm; "
              f"MuJoCo contact={metrics['mujoco_contact_distance_mm']:.2f}mm")
    print("NOTE: unsigned surface distances, not a proven physical clearance.")


if __name__ == "__main__":
    main()
