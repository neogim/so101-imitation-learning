#!/usr/bin/env python3
"""World-frame overlay of original SO-101 STL surfaces, hulls and MuJoCo contacts.

Uses MuJoCo's compiler mesh frame transform, then verifies alignment against
compiled mesh vertices. Contact-to-STL distances are distances to surfaces,
not signed penetrations; proximity at contact midpoint alone cannot prove
that a contact is false. No URDF/robot modifications.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree

import real_to_sim as r2s
from so101_reach_env import SO101ReachEnv
from visualize_base_shoulder_contacts import BASE_SHOULDER_IDS

FILES = {0: "base_motor_holder_so101_v1.stl",
         1: "base_so101_v2.stl",
         2: "sts3215_03a_v1.stl",
         6: "rotation_pitch_so101_v1.stl"}


def source_to_world(model, data, geom_id, mesh):
    """Map raw STL vertex positions through compiler recentering and geom pose."""
    mesh_id = int(model.geom_dataid[geom_id])
    q = np.asarray(model.mesh_quat[mesh_id], dtype=float)
    rotation = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(rotation, q)
    R = rotation.reshape(3, 3)
    scaled = mesh.vertices * np.asarray(model.mesh_scale[mesh_id], dtype=float)
    centered = (scaled - np.asarray(model.mesh_pos[mesh_id], dtype=float)) @ R
    G = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    world = centered @ G.T + np.asarray(data.geom_xpos[geom_id], dtype=float)

    start = int(model.mesh_vertadr[mesh_id])
    compiled = np.asarray(model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]])
    compiled_world = compiled @ G.T + np.asarray(data.geom_xpos[geom_id])
    # Match source vertices to MuJoCo geometry, not arbitrary STL-space axes.
    tree = cKDTree(compiled_world)
    sample = world[::max(1, len(world)//2500)]
    d, _ = tree.query(sample, k=1)
    p95 = float(np.percentile(d, 95))
    if p95 > .002:
        raise RuntimeError(f"geom{geom_id} STL-to-compiled alignment p95={p95:.6f}m >2mm; "
                           "cannot interpret the overlay")
    result = mesh.copy()
    result.vertices = world
    return result, compiled_world, p95


def evaluate(out, sample_points=2000):
    if out.exists():
        raise ValueError("Output already exists")
    env = SO101ReachEnv(seed=101)
    try:
        env.reset(seed=101)
        model, data = env.model, env.data
        raw = {}
        compiled = {}
        alignment = {}
        for gid in BASE_SHOULDER_IDS:
            path = r2s.DEFAULT_URDF.parent / "assets" / FILES[gid]
            src = trimesh.load_mesh(path, process=True, force="mesh")
            world, pts, error = source_to_world(model, data, gid, src)
            raw[gid], compiled[gid], alignment[gid] = world, pts, error
        contacts = []
        for i in range(data.ncon):
            c = data.contact[i]
            a,b = sorted((int(c.geom1), int(c.geom2)))
            if a not in raw or b not in raw:
                continue
            point = np.array(c.pos, dtype=float)
            metrics = {}
            for gid in (a,b):
                # AABB and nearest triangle surface are distinct; no R-tree dependency.
                near, dist, tid = trimesh.proximity.closest_point_naive(
                    raw[gid], point.reshape(1,3))
                metrics[str(gid)] = {
                    "distance_to_source_stl_surface_mm": float(dist[0]*1000),
                    "nearest_source_triangle_index": int(tid[0]),
                    "nearest_source_stl_surface_world_m": near[0].tolist(),
                }
            contacts.append({
                "pair": [a,b], "point_world_m": point.tolist(),
                "mujoco_signed_contact_distance_mm": float(c.dist*1000),
                "source_surface_distance": metrics,
            })
    finally:
        env.close()

    out.mkdir(parents=True)
    (out/"world_aligned_contacts.json").write_text(
        json.dumps({"status":"WORLD_ALIGNED_STL_AND_COMPILED_CONTACT_DIAGNOSTIC",
                    "alignment_p95_m": alignment,
                    "contact_points":contacts,
                    "warnings":[
                        "A MuJoCo contact position is not necessarily an STL surface point.",
                        "Nearest-surface distance is unsigned and does not measure real overlap.",
                        "Convex hull collision representation may differ from the raw STL surface.",
                        "No contact policy, URDF, or hardware was modified.",
                    ]},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (out/"contact_to_stl_surface.csv").open("x",newline="") as f:
        w=csv.writer(f)
        w.writerow(["geom_a","geom_b","signed_mujoco_dist_mm","x_m","y_m","z_m",
                    "stl_a_surface_dist_mm","stl_b_surface_dist_mm"])
        for c in contacts:
            a,b=c["pair"];w.writerow([a,b,c["mujoco_signed_contact_distance_mm"],
                *c["point_world_m"],
                c["source_surface_distance"][str(a)]["distance_to_source_stl_surface_mm"],
                c["source_surface_distance"][str(b)]["distance_to_source_stl_surface_mm"]])
    plot_overlay(raw, compiled, contacts, out/"world_aligned_stl_contacts.png", sample_points)
    return {"alignment_p95_m":alignment, "contact_points":contacts}


def plot_overlay(raw, compiled, contacts, path, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig=plt.figure(figsize=(14,7))
    axes=[fig.add_subplot(121,projection="3d"),fig.add_subplot(122,projection="3d")]
    colors={0:"#186f9a",1:"#eb9a2a",2:"#38a178",6:"#a34e92"}
    for ax,show_compiled,title in zip(axes,[False,True],
                                      ["Original STL surfaces (vertices)",
                                       "Compiled mesh vertices + contacts"]):
        all_xyz=[]
        for gid in BASE_SHOULDER_IDS:
            pts=compiled[gid] if show_compiled else np.asarray(raw[gid].vertices)
            stride=max(1,len(pts)//n)
            ax.scatter(pts[::stride,0],pts[::stride,1],pts[::stride,2],
                       c=colors[gid],s=1.4,alpha=.25,label=f"geom{gid}")
            all_xyz.append(pts)
        cp=np.array([c["point_world_m"] for c in contacts])
        if len(cp):
            ax.scatter(cp[:,0],cp[:,1],cp[:,2],c="red",marker="x",s=80,
                       label="MuJoCo contacts",depthshade=False)
        xyz=np.concatenate(all_xyz)
        mid=(xyz.max(axis=0)+xyz.min(axis=0))/2
        radius=max(xyz.max(axis=0)-xyz.min(axis=0))*.58
        ax.set(xlim=(mid[0]-radius,mid[0]+radius),
               ylim=(mid[1]-radius,mid[1]+radius),
               zlim=(mid[2]-radius,mid[2]+radius))
        ax.set_xlabel("world X [m]");ax.set_ylabel("world Y [m]")
        ax.set_zlabel("world Z [m]");ax.set_title(title)
        ax.view_init(elev=20,azim=-70)
        ax.legend(fontsize=8)
    fig.suptitle("SO-101 base / shoulder, zero joint pose; BOTH panels in world coordinates\n"
                 "Dots are sampled vertices, not filled triangle surfaces")
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=evaluate(a.out)
    print("Saved",a.out/"world_aligned_stl_contacts.png")
    print("STL-to-compiled alignment 95th percentile (m):",result["alignment_p95_m"])
    for c in result["contact_points"]:
        print(c["pair"],"MuJoCo contact(mm)=",round(c["mujoco_signed_contact_distance_mm"],2),
              "dist to STL surfaces(mm)=", [
                round(c["source_surface_distance"][str(g)]["distance_to_source_stl_surface_mm"],2)
                for g in c["pair"]])


if __name__=="__main__":
    main()
