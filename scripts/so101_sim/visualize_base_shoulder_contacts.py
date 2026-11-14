#!/usr/bin/env python3
"""Visualize zero-pose SO-101 base/shoulder MuJoCo collision meshes.

The 3D projections are constructed from *compiled* MuJoCo mesh vertices,
world geom poses and MuJoCo contact points. These are physics collision
surfaces (often convexified), not necessarily exact original STL triangles.
This is offline diagnostics only; no hardware connection or URDF changes.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from so101_reach_env import SO101ReachEnv
from inspect_reach_collisions import geom_inventory


BASE_SHOULDER_IDS = (0, 1, 2, 6)


def collision_snapshot():
    env = SO101ReachEnv(seed=101)
    try:
        env.reset(seed=101)
        model, data = env.model, env.data
        geom_map = geom_inventory(model)
        vertices = {}
        for gid in BASE_SHOULDER_IDS:
            meshid = int(model.geom_dataid[gid])
            if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                raise ValueError(f"geom{gid} is not a compiled mesh")
            start = int(model.mesh_vertadr[meshid])
            end = start + int(model.mesh_vertnum[meshid])
            local_vertices = np.array(model.mesh_vert[start:end], dtype=float)
            rotation = np.array(data.geom_xmat[gid], dtype=float).reshape(3, 3)
            center = np.array(data.geom_xpos[gid], dtype=float)
            world_vertices = local_vertices @ rotation.T + center
            if not np.isfinite(world_vertices).all() or not len(world_vertices):
                raise ValueError(f"Invalid mesh vertices for geom{gid}")
            vertices[gid] = world_vertices

        contacts = []
        for i in range(data.ncon):
            contact = data.contact[i]
            a, b = sorted((int(contact.geom1), int(contact.geom2)))
            if a in BASE_SHOULDER_IDS and b in BASE_SHOULDER_IDS:
                contacts.append({
                    "pair": [a, b],
                    "position_m": np.asarray(contact.pos, dtype=float).tolist(),
                    "signed_distance_m": float(contact.dist),
                    "normal_mujoco_frame": np.asarray(contact.frame[:3], dtype=float).tolist(),
                })
        return geom_map, vertices, contacts
    finally:
        env.close()


def plot_snapshot(vertices, contacts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 6))
    axes = [fig.add_subplot(121, projection="3d"),
            fig.add_subplot(122, projection="3d")]
    all_points = np.concatenate([vertices[k] for k in BASE_SHOULDER_IDS])
    center = np.mean(all_points, axis=0)
    reach = float(np.max(np.ptp(all_points, axis=0))) * .6
    reach = max(reach, .04)

    for ax, elev, azim in zip(axes, (26, 15), (-65, 30)):
        for gid in BASE_SHOULDER_IDS:
            xyz = vertices[gid]
            step = max(1, len(xyz) // 1200)
            ax.scatter(xyz[::step, 0], xyz[::step, 1], xyz[::step, 2],
                       s=2, alpha=.30, label=f"geom{gid}")
        if contacts:
            positions = np.array([x["position_m"] for x in contacts])
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                       marker="x", s=65, c="red", linewidths=1.7,
                       label="MuJoCo contact points")
        ax.set(xlim=(center[0]-reach, center[0]+reach),
               ylim=(center[1]-reach, center[1]+reach),
               zlim=(center[2]-reach, center[2]+reach))
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.view_init(elev=elev, azim=azim)
        ax.legend(fontsize=7)
    fig.suptitle("SO-101 base / shoulder: compiled collision mesh vertices and contacts\n"
                 "0 rad joint pose, red X = MuJoCo contact location; not original STL visual mesh")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_snapshot(out):
    if out.exists():
        raise ValueError(f"Output directory already exists: {out}")
    geom_map, vertices, contacts = collision_snapshot()
    out.mkdir(parents=True)
    plot_snapshot(vertices, contacts, out / "base_shoulder_contacts_3d.png")
    with (out / "contacts.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["geom_a", "geom_b", "x_m", "y_m", "z_m", "signed_distance_m"])
        for x in contacts:
            writer.writerow([*x["pair"], *x["position_m"], x["signed_distance_m"]])
    report = {
        "status": "COLLISION_MESH_VERTEX_AND_CONTACT_SNAPSHOT_NOT_PRODUCTION_VALIDATION",
        "joint_pose": "all_five_at_zero_rad",
        "geom_info": [x for x in geom_map if x["geom_id"] in BASE_SHOULDER_IDS],
        "vertices_per_geom": {str(g): len(v) for g, v in vertices.items()},
        "mesh_world_aabb_m": {
            str(g): {"min": v.min(axis=0).tolist(), "max": v.max(axis=0).tolist()}
            for g, v in vertices.items()},
        "contact_points": contacts,
        "negative_distances_may_indicate_penetration": True,
        "notes": [
            "MuJoCo compiled collision meshes may use convex hulls of nonconvex STL.",
            "Points show vertex clouds, not exact triangle surfaces.",
            "Signed contact distance is from MuJoCo's collision detector, not STL surface-overlap volume.",
            "Do not exclude contacts or modify original URDF based on image alone.",
        ],
    }
    (out / "geometry_snapshot.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = save_snapshot(args.out)
    print(f"Saved {args.out / 'base_shoulder_contacts_3d.png'}")
    print(f"contacts={len(report['contact_points'])}")
    for item in report["contact_points"]:
        print("pair", item["pair"], "contact_distance_m",
              round(item["signed_distance_m"], 6))


if __name__ == "__main__":
    main()
