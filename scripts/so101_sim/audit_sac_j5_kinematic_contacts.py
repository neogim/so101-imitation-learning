#!/usr/bin/env python3
"""Audit physical geom contacts along a SAVED kinematic SAC J5-ablation rollout.

This is a contact *diagnostic*, not a physics-enabled replay. No SAC weights,
source URDF/STL, real episodes, or robot hardware are modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

import demo_real2sim_muhi_cup_reach as scene


def contact_records(model, data):
    records = []
    for i in range(data.ncon):
        contact = data.contact[i]
        a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        records.append({
            "geom_a": a or f"geom{contact.geom1}",
            "geom_b": b or f"geom{contact.geom2}",
            "distance_mm": 1000.0 * float(contact.dist),
            "constraint_active": int(contact.efc_address) >= 0,
        })
    return records


def audit(model, data, qids, rows):
    output = []
    for row in rows:
        q = np.array([float(row[f"joint_{i}"]) for i in range(1, 6)])
        if not np.isfinite(q).all():
            raise ValueError("Non-finite joint angles in rollout")
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)
        contacts = contact_records(model, data)
        output.append({
            "policy": row["policy"], "frame": int(row["frame"]),
            "stage": row["stage"], "contact_count": len(contacts),
            "penetrating_count": sum(x["distance_mm"] < 0 for x in contacts),
            "minimum_contact_distance_mm": min(
                (x["distance_mm"] for x in contacts), default=None),
            "contacts": contacts,
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=scene.DATASET)
    parser.add_argument("--episode", type=int, default=4)
    parser.add_argument("--object-layout-json", type=Path, default=scene.OBJECTS)
    parser.add_argument("--cup-pose-json", type=Path, default=scene.CUP_POSE)
    parser.add_argument("--extrinsics", type=Path, default=scene.EXTRINSICS)
    parser.add_argument("--intrinsics", type=Path, default=scene.INTRINSICS)
    parser.add_argument("--urdf", type=Path, default=scene.r2s.DEFAULT_URDF)
    parser.add_argument("--robot-joint-offsets-json", type=Path)
    parser.add_argument("--muhi-clearance-mm", type=float, default=75.)
    parser.add_argument("--cup-clearance-mm", type=float, default=45.)
    args = parser.parse_args()
    with args.rollout.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(r.get("policy") not in ("normal", "j5_fixed") for r in rows):
        parser.error("Expected nonempty J5 ablation rollout.csv")
    if args.out.exists():
        parser.error("Output already exists")
    (model, data, qids, *_rest) = scene.create_scene(args)
    audited = audit(model, data, qids, rows)
    pairs = Counter()
    for frame in audited:
        for contact in frame["contacts"]:
            pairs[tuple(sorted((contact["geom_a"], contact["geom_b"])))] += 1
    report = {
        "mode": "STATIC_KINEMATIC_CONTACT_AUDIT_NOT_DYNAMICS",
        "frames": len(audited),
        "frames_with_contacts": sum(f["contact_count"] > 0 for f in audited),
        "frames_with_penetration": sum(f["penetrating_count"] > 0 for f in audited),
        "top_contact_pairs": [
            {"geoms": list(pair), "contact_records": count}
            for pair, count in pairs.most_common(30)
        ],
        "limitations": [
            "mj_forward detects contacts, but does not integrate time or enforce contact response.",
            "Any preview-only visual geoms excluded by collision masks cannot appear as contacts.",
            "A negative contact distance indicates geom penetration, not necessarily physical STL penetration.",
            "The saved joint trajectory may exceed joint limits and is not hardware-safe.",
        ],
        "frames_detail": audited,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "contact_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.out / "contact_frames.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "policy", "frame", "stage", "contact_count", "penetrating_count",
            "minimum_contact_distance_mm"])
        writer.writeheader()
        for frame in audited:
            writer.writerow({k: frame[k] for k in writer.fieldnames})
    print(f"Audited {len(audited)} frames; contact frames: {report['frames_with_contacts']}; "
          f"penetration frames: {report['frames_with_penetration']}")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
