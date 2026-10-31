#!/usr/bin/env python3
"""Verify offline Cartesian LeRobot dataset copy with independent structural checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

def run(source,dest):
    source,dest=source.resolve(),dest.resolve()
    info=json.loads((dest/"meta/info.json").read_text())
    manifest=json.loads((dest/"cartesian_experimental_manifest.json").read_text())
    if info["features"]["action"]["shape"]!=[7]:
        raise ValueError("Destination action metadata not seven-dimensional")
    if info["features"]["observation.state"]["shape"]!=[6]:
        raise ValueError("Destination state metadata changed")
    src_files=sorted((source/"data").rglob("*.parquet"))
    count=0
    for src in src_files:
        dst=dest/src.relative_to(source)
        if not dst.is_file():raise ValueError(f"Missing output partition: {dst}")
        a=pq.read_table(src)
        b=pq.read_table(dst)
        if len(a)!=len(b):raise ValueError(f"Changed row count: {dst}")
        if a.column_names!=b.column_names:raise ValueError(f"Changed columns: {dst}")
        for col in a.column_names:
            if col=="action":
                for v in b[col].to_pylist():
                    if len(v)!=7 or not np.isfinite(np.asarray(v,float)).all():
                        raise ValueError(f"Invalid output Cartesian action: {dst}")
            elif not a[col].equals(b[col]):
                raise ValueError(f"Unexpected modified source column {col}: {dst}")
        count+=len(a)
    if count!=info["total_frames"] or count!=manifest["source_frames"]:
        raise ValueError("Dataset frame count mismatch")
    src_videos={p.relative_to(source) for p in (source/"videos").rglob("*") if p.is_file()} if (source/"videos").exists() else set()
    dst_videos={p.relative_to(dest) for p in (dest/"videos").rglob("*") if p.is_file()} if (dest/"videos").exists() else set()
    if src_videos!=dst_videos:raise ValueError("Video file set differs")
    if (dest/"meta/stats.json").exists():
        stats=json.loads((dest/"meta/stats.json").read_text())
        for k in ("mean","std","min","max"):
            if np.asarray(stats["action"][k]).size!=7:
                raise ValueError(f"Action stats {k} is not seven-dimensional")
    print(json.dumps({"structural_check":"PASS","frames":count,
                      "unchanged_observation_columns":True,"video_files":len(src_videos),
                      "action_dimensions":7,"act_training_verified":False},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--converted",type=Path,required=True)
    a=p.parse_args()
    run(a.dataset,a.converted)

if __name__=="__main__":main()
