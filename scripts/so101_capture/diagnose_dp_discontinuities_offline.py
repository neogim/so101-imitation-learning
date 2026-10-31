#!/usr/bin/env python3
"""Read-only diagnosis of discontinuities in a saved offline DP selected_paths.json."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
LIMITS = {"jump": 5.0, "speed": 100.0, "accel": 2000.0}


def diagnose(rows: list[dict], mapping: str, method: str, fps: float = 20.0) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["mapping"] == mapping and row["method"] == method:
            groups[row["direction"]].append(row)
    if not groups:
        raise ValueError("No matching selected paths")
    result = {}
    for direction, frames in sorted(groups.items()):
        frames.sort(key=lambda x: int(x["frame"]))
        detections = []
        for left, right in zip(frames, frames[1:]):
            if right["frame"] != left["frame"] + 1 or not left["candidate"] or not right["candidate"]:
                continue
            q0 = np.array(left["candidate"]["motor"], dtype=float)
            q1 = np.array(right["candidate"]["motor"], dtype=float)
            if not (np.isfinite(q0).all() and np.isfinite(q1).all()):
                raise ValueError("Nonfinite candidate")
            delta = np.abs(q1 - q0)
            for joint_index in np.flatnonzero(delta > LIMITS["jump"]):
                detections.append({"kind": "jump", "frame_from": int(left["frame"]),
                  "frame_to": int(right["frame"]), "joint": JOINTS[joint_index],
                  "value": float(delta[joint_index]), "threshold": LIMITS["jump"],
                  "seed_before": left["candidate"]["seed"], "seed_after": right["candidate"]["seed"],
                  "active_bounds_before": left["candidate"]["active_bound_joints"],
                  "active_bounds_after": right["candidate"]["active_bound_joints"]})
        # In this fixed 20 Hz recording, delta positions also define velocity.
        for left, right in zip(frames, frames[1:]):
            if right["frame"] != left["frame"] + 1 or not left["candidate"] or not right["candidate"]:
                continue
            delta = np.abs(np.array(right["candidate"]["motor"]) - np.array(left["candidate"]["motor"]))
            for j in np.flatnonzero(delta * fps > LIMITS["speed"]):
                detections.append({"kind":"speed", "frame_from": int(left["frame"]), "frame_to": int(right["frame"]),
                  "joint": JOINTS[j], "value": float(delta[j] * fps), "threshold": LIMITS["speed"]})
        for a,b,c in zip(frames,frames[1:],frames[2:]):
            if c["frame"] != b["frame"]+1 or b["frame"] != a["frame"]+1 or any(x["candidate"] is None for x in (a,b,c)):
                continue
            x,y,z=(np.asarray(t["candidate"]["motor"],dtype=float) for t in (a,b,c))
            acc=np.abs(z-2*y+x)*fps*fps
            for j in np.flatnonzero(acc > LIMITS["accel"]):
                detections.append({"kind":"accel","frame":int(c["frame"]),"joint":JOINTS[j],
                  "value":float(acc[j]),"threshold":LIMITS["accel"]})
        detections.sort(key=lambda x:(x.get("frame_to",x.get("frame")),x["kind"],x["joint"]))
        result[direction]={"violations": len(detections),"first_events":detections[:12],
          "max_ratio":max((x["value"]/x["threshold"] for x in detections),default=0.0)}
    return result


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths",type=Path,required=True)
    p.add_argument("--mapping",default="shoulder+_elbow+")
    p.add_argument("--method",default="dp_second_order")
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    result=diagnose(json.loads(a.paths.read_text()),a.mapping,a.method)
    brief={"mapping":a.mapping,"method":a.method,"direction_diagnostics":result,
      "notes":"Offline finite candidate paths only; physical mapping, collision and safety unverified."}
    print(json.dumps(brief,indent=2,ensure_ascii=False))
    if a.output:
        if a.output.exists():p.error("Output already exists")
        a.output.write_text(json.dumps(brief,indent=2,ensure_ascii=False)+"\n")

if __name__=="__main__":
    main()
