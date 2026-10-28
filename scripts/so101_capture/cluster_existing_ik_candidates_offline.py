#!/usr/bin/env python3
"""Read-only candidate-cluster audit on existing per_seed.json, no IK/hardware.

Clusters full five-joint motor angle vectors per frame and direction using an
explicit Euclidean-degree threshold. These are candidate groups, not proven
topologically separate IK branches.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

def groups(points, radius_deg):
    """Connected components of the threshold-distance graph."""
    a=np.asarray(points,dtype=float)
    n=len(a)
    parent=list(range(n))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for i in range(n):
        for j in range(i):
            if np.linalg.norm(a[i]-a[j])<=radius_deg:
                x,y=root(i),root(j)
                parent[x]=y
    labels=[root(i) for i in range(n)]
    order={v:k for k,v in enumerate(dict.fromkeys(labels))}
    return [order[v] for v in labels]

def analyze(candidates, paths, mapping, radius, start, end):
    selected={(r["direction"],int(r["frame"])):r["candidate"] for r in paths
              if r["mapping"]==mapping and r["method"]=="dp_second_order"
              and start<=int(r["frame"])<=end}
    grouped=defaultdict(list)
    for c in candidates:
        if c["mapping"]==mapping and start<=int(c["frame"])<=end and c["feasible_witness"] and c["position_residual_mm"]<=2:
            grouped[(c["direction"],int(c["frame"]))].append(c)
    result=[]
    for key, items in sorted(grouped.items()):
        points=np.asarray([x["candidate_motor_deg"] for x in items],dtype=float)
        labels=groups(points,radius)
        count=len(set(labels))
        chosen=selected.get(key)
        nearest=None
        if chosen is not None:
            ds=np.linalg.norm(points-np.asarray(chosen["motor"],dtype=float),axis=1)
            nearest={"cluster":labels[int(np.argmin(ds))],"distance_deg":float(ds.min())}
        if count>1:
            result.append({"direction":key[0],"frame":key[1],"clusters":count,
                           "sizes":[labels.count(i) for i in range(count)],
                           "candidate_count":len(items),"chosen":nearest})
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates",required=True,type=Path)
    p.add_argument("--paths",required=True,type=Path)
    p.add_argument("--mapping",default="shoulder+_elbow+")
    p.add_argument("--start",type=int,default=0);p.add_argument("--end",type=int,default=119)
    p.add_argument("--radius-deg",type=float,default=5)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if a.radius_deg<=0 or a.start>a.end or a.output.exists():p.error("Invalid range/radius or output exists")
    records=analyze(json.loads(a.candidates.read_text()),json.loads(a.paths.read_text()),
                    a.mapping,a.radius_deg,a.start,a.end)
    report={"mapping":a.mapping,"radius_deg":a.radius_deg,"multi_cluster_frame_directions":len(records),
            "examples":records[:30],"records":records,
            "limitations":["Cluster ID is local to each frame; not a persistent branch ID.",
                           "5D Euclidean motor angle distance and radius are arbitrary analysis choices.",
                           "Selected DP candidates might have no exact seed-array match because of deduplication.",
                           "This does not prove physical IK branches, singularities, or safety."]}
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print("mapping:",a.mapping,"radius:",a.radius_deg,"multi-cluster frame-directions:",len(records))
    for r in records[:15]:
        print(f"  {r['direction']} f{r['frame']}: {r['clusters']} groups sizes={r['sizes']}")
    print("Full details:",a.output)
if __name__=="__main__":main()
