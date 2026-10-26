#!/usr/bin/env python3
"""Audit gripper supervision and find candidate grasp-centric windows. Read only.

Window marks first sustained closure rise (per episode), includes lead-in and
lift/settle context. Not a new Dataset; not verified task-event labels.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--threshold",type=float,default=25.0)
    p.add_argument("--pre-seconds",type=float,default=3.0)
    p.add_argument("--post-seconds",type=float,default=3.0)
    a=p.parse_args()
    root=a.dataset.resolve(); out=a.output.resolve()
    if out.exists(): raise FileExistsError(out)
    if root in out.parents: raise ValueError("Output must be outside Dataset")
    split=json.loads((root/"episode_split.json").read_text())
    data={}
    for f in sorted((root/"data").rglob("*.parquet")):
        t=pq.read_table(f,columns=["episode_index","frame_index","action"])
        for e,k,v in zip(t["episode_index"].to_pylist(),t["frame_index"].to_pylist(),t["action"].to_pylist()):
            if len(v)!=7 or not np.isfinite(v).all(): raise ValueError("Invalid action")
            data.setdefault(int(e),[]).append((int(k),float(v[6])))
    info=json.loads((root/"meta/info.json").read_text())
    fps=float(info["fps"]);pre=int(round(a.pre_seconds*fps));post=int(round(a.post_seconds*fps))
    train=set(map(int,split["train_episode_indices"]))
    val=set(map(int,split["validation_episode_indices"]))
    if len(train)!=45 or len(val)!=8 or train&val or train|val!=set(data):
        raise ValueError("Dataset episodes do not match 45/8 split")
    rows=[]
    for e,pts in sorted(data.items()):
        pts.sort()
        frames=np.array([q[0] for q in pts]);g=np.array([q[1] for q in pts])
        if not np.array_equal(frames,np.arange(len(g))):
            raise ValueError(f"Non-contiguous frame_index: {e}")
        above=np.flatnonzero(g>a.threshold)
        if len(above):
            onset=int(above[0])
            start=max(0,onset-pre);stop=min(len(g),onset+post+1)
            status="heuristic_first_gt_over_threshold"
        else:
            onset=-1;start=-1;stop=-1;status="no_gt_over_threshold"
        rows.append({"episode_index":e,"split":"train" if e in train else "validation",
          "frames":len(g),"gt_over_threshold":int(len(above)),
          "gt_over_threshold_fraction":float(len(above)/len(g)),
          "gripper_min":float(g.min()),"gripper_max":float(g.max()),
          "first_over_threshold_frame":onset,"proposed_window_start":start,
          "proposed_window_stop_exclusive":stop,"window_status":status})
    train_rows=[r for r in rows if r["split"]=="train"]
    gt_total=sum(r["gt_over_threshold"] for r in train_rows)
    frames_total=sum(r["frames"] for r in train_rows)
    report={"fps":fps,"threshold":a.threshold,"pre_seconds":a.pre_seconds,"post_seconds":a.post_seconds,
      "train_episodes":45,"validation_episodes":8,
      "train_gt_over_threshold_frames":gt_total,
      "train_gt_over_threshold_fraction":gt_total/frames_total,
      "train_episodes_with_threshold":sum(r["first_over_threshold_frame"]>=0 for r in train_rows),
      "warning":"Threshold crossing is NOT confirmed grasp timing; visually review each proposed window. No new image-action pairs generated."}
    out.mkdir(parents=True)
    with (out/"gripper_episode_audit.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
