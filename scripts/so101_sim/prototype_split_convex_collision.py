#!/usr/bin/env python3
"""Step B: ONE disposable split-convex collision prototype for SO-101.

Split each selected source collision STL into a 2x2x2 grid of solid
intersections, convexify *each* occupied cell, then replace ONLY that
collision element in a copied URDF. Visual meshes, inertia, joints and the
source URDF/STLs remain unchanged.

This prototype is a geometric hypothesis, not hardware/safety validation.
No SAC training or robot commands are performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh

import real_to_sim as r2s
from so101_reach_env import SO101ReachEnv

# Three geometry sources implicated in zero-pose false contacts.
TARGETS = {
    ("base_link", "base_motor_holder_so101_v1.stl"),
    ("base_link", "base_so101_v2.stl"),
    ("shoulder_link", "rotation_pitch_so101_v1.stl"),
}


def split_to_convex_parts(mesh: trimesh.Trimesh, *, min_volume_mm3=0.5):
    """Make convex chunks by actual SOLID boolean crop, not vertex clustering."""
    bounds = np.asarray(mesh.bounds, dtype=float)
    middle = bounds.mean(axis=0)
    parts = []
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                idx = (ix, iy, iz)
                low = np.array([bounds[0, i] if idx[i] == 0 else middle[i]
                                for i in range(3)])
                high = np.array([middle[i] if idx[i] == 0 else bounds[1, i]
                                 for i in range(3)])
                # Slightly overlapping cut boxes avoid losing border triangles;
                # resulting proximity/overlap between pieces is a limitation.
                epsilon = 1e-7
                center = (low + high) / 2
                box = trimesh.creation.box(extents=high-low+2*epsilon)
                box.apply_translation(center)
                solid = trimesh.boolean.intersection([mesh, box], engine="manifold")
                if solid is None or solid.is_empty:
                    continue
                if abs(float(solid.volume)) * 1e9 < min_volume_mm3:
                    continue
                hull = solid.convex_hull
                # Some grid intersections are degenerate/sliver components.
                # A non-solid hull cannot be used as a MuJoCo collision part.
                if (not hull.is_watertight or not np.isfinite(hull.volume)
                        or abs(float(hull.volume)) < 1e-12):
                    continue
                if hull.volume < 0:
                    hull.invert()
                parts.append(hull)
    if not parts:
        raise RuntimeError("No convex pieces created")
    return parts


def generate_model(out: Path):
    if out.exists():
        raise ValueError(f"Output exists: {out}")
    source_urdf = r2s.DEFAULT_URDF.resolve()
    source_assets = source_urdf.parent / "assets"
    tree = ET.parse(source_urdf)
    root = tree.getroot()

    # Resolve ALL unrelated visual/collision meshes to original absolute STL paths,
    # because the generated URDF is stored under a different directory.
    for node in root.findall(".//mesh"):
        path = node.attrib["filename"]
        node.attrib["filename"] = str((source_urdf.parent / path).resolve())

    generated = {}
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "")
        for collision in list(link.findall("collision")):
            original_mesh = collision.find("geometry/mesh")
            if original_mesh is None:
                continue
            filename = Path(original_mesh.attrib["filename"]).name
            target = (link_name, filename)
            if target not in TARGETS:
                continue
            raw = trimesh.load_mesh(source_assets / filename,
                                    process=True, force="mesh")
            if not raw.is_watertight or not raw.is_volume:
                raise ValueError(f"Source STL is not a valid solid: {filename}")
            pieces = split_to_convex_parts(raw)
            index = list(link).index(collision)
            link.remove(collision)
            files = []
            for i, piece in enumerate(pieces):
                name = f"{link_name}_{Path(filename).stem}_chunk{i:02d}.stl"
                path = out / "collision_parts" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                piece.export(path)
                # Each piece retains the SAME part-local frame and origin
                # as its source STL: no rotations/translations are guessed.
                new_collision = ET.Element("collision")
                origin = collision.find("origin")
                if origin is not None:
                    new_collision.append(ET.fromstring(ET.tostring(origin)))
                geometry = ET.SubElement(new_collision, "geometry")
                ET.SubElement(geometry, "mesh", filename=str(path.resolve()))
                link.insert(index + i, new_collision)
                files.append(str(path.resolve()))
            generated[f"{link_name}:{filename}"] = files

    out.mkdir(parents=True, exist_ok=True)
    output_urdf = out / "so101_split_convex_diagnostic.urdf"
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)

    # Smoke-compile the GENERATED URDF; no model parameter is changed in code.
    spec = mujoco.MjSpec.from_file(str(output_urdf.resolve()))
    spec.compiler.discardvisual = True
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial_contacts = int(data.ncon)
    return {
        "status": "UNVALIDATED_SPLIT_CONVEX_DIAGNOSTIC_NOT_HARDWARE_SAFE",
        "source_urdf": str(source_urdf),
        "generated_urdf": str(output_urdf.resolve()),
        "collision_geoms": int(model.ngeom),
        "initial_contact_points": initial_contacts,
        "pieces": generated,
        "notes": [
            "2x2x2 solid chunks are individually convexified; this is ONE prototype.",
            "Original visual STL and all link inertial/joint properties are retained.",
            "This is not a fitted or safety-certified collision model.",
            "Adjacent pieces may overlap; surface-only contact is not classified.",
            "The physical SO-101 model, workspace clearance and SAC readiness are not proven.",
        ],
    }


def oracle(urdf: Path, seed=101, episode=0, steps=150, frame_skip=10):
    env = SO101ReachEnv(urdf=urdf, seed=seed, max_steps=steps,
                        frame_skip=frame_skip)
    try:
        _, info0 = env.reset(seed=seed + episode)
        wanted = env.goal_joint_angles.copy()
        env.targets = wanted.copy()
        initial_contacts = int(env.data.ncon)
        contact_steps = 0
        saturation = np.zeros(5)
        min_dist = float("inf")
        n = 0

        def record(row):
            nonlocal n, contact_steps, min_dist
            n += 1
            contact_steps += bool(row["contact_pairs"])
            saturation[:] += np.asarray(row["torque_saturated"], dtype=float)
            min_dist = min(min_dist, float(row["tcp_distance_m"]))

        for i in range(steps):
            _, _, terminated, truncated, info = env.step(
                np.zeros(5), trace_hook=record)
            if terminated or truncated:
                break
        return {
            "initial_contact_points": initial_contacts,
            "goal_xyz_m": info0["goal_base_m"],
            "goal_joint_rad": wanted.tolist(),
            "final_tcp_mm": float(info["distance_m"]*1000),
            "min_tcp_mm": min_dist*1000,
            "joint_error_l2_rad": float(np.linalg.norm(
                env.data.qpos[env.qids]-wanted)),
            "contact_step_fraction": contact_steps/n,
            "mean_torque_saturation_fraction": float(np.mean(saturation/n)),
            "reach_success": bool(info["is_success"]),
        }
    finally:
        env.close()


def compare(out: Path, *, seed=101, episode=0, steps=150, frame_skip=10):
    descriptor = generate_model(out)
    baseline = oracle(r2s.DEFAULT_URDF, seed, episode, steps, frame_skip)
    proxy = oracle(Path(descriptor["generated_urdf"]), seed, episode, steps, frame_skip)
    np.testing.assert_allclose(baseline["goal_xyz_m"], proxy["goal_xyz_m"], atol=1e-7)
    np.testing.assert_allclose(baseline["goal_joint_rad"], proxy["goal_joint_rad"], atol=1e-7)
    rows = [{**{"model": "original"}, **baseline},
            {**{"model": "split_convex_prototype"}, **proxy}]
    with (out/"pd_comparison.csv").open("x", newline="") as f:
        fields = ["model", "initial_contact_points", "final_tcp_mm", "min_tcp_mm",
                  "joint_error_l2_rad", "contact_step_fraction",
                  "mean_torque_saturation_fraction", "reach_success"]
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:row[key] for key in fields})
    descriptor["evaluation"] = rows
    (out/"summary.json").write_text(
        json.dumps(descriptor, indent=2, ensure_ascii=False)+"\n")
    return descriptor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--steps", type=int, default=150)
    args=p.parse_args()
    report=compare(args.out, seed=args.seed, episode=args.episode,
                   steps=args.steps)
    print("Saved", args.out/"pd_comparison.csv")
    print("split convex chunks:", {k: len(v) for k,v in report["pieces"].items()})
    for row in report["evaluation"]:
        print(row["model"], "TCP(mm)=",round(row["final_tcp_mm"],2),
              "initial contacts=",row["initial_contact_points"],
              "saturation=",round(row["mean_torque_saturation_fraction"],3),
              "success=",row["reach_success"])


if __name__ == "__main__":
    main()
