#!/usr/bin/env python3
"""Read-only audit of SO-101 J5 parent/child links and ArUco observations.

Does not solve extrinsics, connect hardware, or alter the existing capture.
ID-to-link mapping is a hypothesis based on observed J5-only motion.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET

CANDIDATE_LINKS = {0: "gripper_link", 1: "gripper_link", 2: "wrist_link"}


def inspect(urdf: Path, manifest: Path):
    root = ET.parse(urdf).getroot()
    links = {tag.attrib["name"] for tag in root.findall("link")}
    joint = next((x for x in root.findall("joint")
                  if x.attrib.get("name") == "wrist_roll"), None)
    if joint is None:
        raise ValueError("URDF has no wrist_roll joint")
    parent = joint.find("parent")
    child = joint.find("child")
    axis = joint.find("axis")
    if parent is None or child is None:
        raise ValueError("Invalid wrist_roll joint")
    parent_link = parent.attrib["link"]
    child_link = child.attrib["link"]
    if parent_link not in links or child_link not in links:
        raise ValueError("Unknown wrist_roll parent/child links")
    if parent_link != "wrist_link" or child_link != "gripper_link":
        raise ValueError("Unexpected wrist_roll structure: verify URDF and mapping")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if "samples" not in data or not isinstance(data["samples"], list):
        raise ValueError("Manifest has no samples")
    edges = defaultdict(list)
    positions = defaultdict(list)
    for sample in data["samples"]:
        for marker in sample["markers"]:
            mid = int(marker["id"])
            edges[mid].append(float(marker["min_edge_px"]))
            positions[mid].append((sample["episode_index"], sample["frame_index"]))
    result = {
        "urdf": str(urdf),
        "manifest": str(manifest),
        "wrist_roll": {
            "parent": parent_link, "child": child_link,
            "axis": None if axis is None else axis.attrib.get("xyz")
        },
        "candidates": [
            {
                "id": mid,
                "candidate_link": CANDIDATE_LINKS.get(mid),
                "count": len(edges[mid]),
                "min_edge_px": round(min(edges[mid]), 3) if edges[mid] else None,
                "max_edge_px": round(max(edges[mid]), 3) if edges[mid] else None,
                "frame_refs": positions[mid],
            } for mid in range(6)
        ],
        "unverified": [
            "ID 0/1 may move with gripper_link; ID 2 may be fixed to wrist_link.",
            "Confirm J5-only motion and that ID 2 is not attached to a flexible cable.",
            "URDF has no measured marker-to-link transforms.",
            "Camera intrinsics for the OBS 640x480 image and FK convention remain unverified.",
        ],
    }
    return result


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path,
                        default=root / "data/urdf/so101/so101_new_calib.urdf")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.urdf, args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
