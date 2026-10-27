#!/usr/bin/env python3
"""Read-only full-frame audit of recorded SO-101 joint goals against URDF limits.

Optional --ik-replay accepts JSON array with episode, frame, ik_arm_deg; it is
checked against recorded frames and compared independently. This does NOT
generate IK solutions or send motor commands. Physical calibration unknown.
"""
from __future__ import annotations
import argparse,csv,json,xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import numpy as np
NAMES=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll")
def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds=LeRobotDataset("local/muhi_ik_full_audit",root=args.dataset,video_backend="pyav")
    if len(ds)!=30846:raise ValueError("Dataset frame count changed")
    raw=ds.hf_dataset
    ep=np.asarray(raw["episode_index"],dtype=int)
    fr=np.asarray(raw["frame_index"],dtype=int)
    goal=np.asarray(raw["action"],dtype=float)
    if goal.shape!=(len(ep),6) or not np.isfinite(goal).all():raise ValueError("Invalid 6D action")
    split=json.loads((args.split_dataset/"episode_split.json").read_text())
    train=set(map(int,split["train_episode_indices"]))
    valid=set(map(int,split["validation_episode_indices"]))
    if len(train)!=45 or len(valid)!=8 or train&valid or train|valid!=set(range(53)):
        raise ValueError("Invalid fixed split")
    tree=ET.parse(args.urdf).getroot()
    joints={j.attrib["name"]:j for j in tree.findall("joint")}
    bounds=[]
    for n in NAMES:
        lim=joints[n].find("limit")
        if lim is None:raise ValueError(f"Missing URDF limit {n}")
        bounds.append([np.rad2deg(float(lim.attrib[x])) for x in ("lower","upper")])
    lower=np.array(bounds)[:,0];upper=np.array(bounds)[:,1]
    bad=(goal[:,:5]<lower)|(goal[:,:5]>upper)
    allrows=[];violations=Counter();rates=[]
    for eid in sorted(set(ep)):
        idx=np.flatnonzero(ep==eid)
        idx=idx[np.argsort(fr[idx])]
        if not np.array_equal(fr[idx],np.arange(len(idx))):
            raise ValueError(f"Episode {eid} frames not contiguous")
        q=goal[idx,:5]
        dq=np.diff(q,axis=0)
        if dq.size:
            rates.extend(np.abs(dq*ds.fps).reshape(-1).tolist())
        for j,n in enumerate(NAMES):
            violations[n]+=int(bad[idx,j].sum())
        for pos,k in enumerate(idx):
            row={"episode":int(eid),"frame":int(fr[k]),"split":"train" if eid in train else "heldout",
                 "any_urdf_violation":int(bad[k].any())}
            for j,n in enumerate(NAMES):
                row[f"{n}_goal"]=float(goal[k,j])
                row[f"{n}_urdf_violation"]=int(bad[k,j])
                row[f"{n}_delta_prev"]=float(q[pos,j]-q[pos-1,j]) if pos else ""
            allrows.append(row)
    report={"source":"recorded joint Action (NOT generated IK)","frames":len(ep),
       "episodes":len(set(ep)),"train_episodes":len(train),"heldout_episodes":len(valid),
       "urdf_bounds_degrees":dict(zip(NAMES,bounds)),
       "urdf_violating_frames":int(bad.any(axis=1).sum()),
       "violations_by_joint":dict(violations),
       "max_abs_recorded_goal_change_per_20fps_frame":float(max((abs(float(r[f"{n}_delta_prev"])) for r in allrows for n in NAMES if r[f"{n}_delta_prev"]!=""),default=0)),
       "max_abs_interval_average_goal_velocity_deg_per_s":float(max(rates,default=0)),
       "limitations":["Recorded values may not be physical URDF joint angles; zero offset/sign/calibration must be verified",
          "Recorded action step does not prove realized servo velocity or collision clearance",
          "No IK solve, motor action, or safety certification performed"]}
    if args.ik_replay:
        rows=json.loads(args.ik_replay.read_text())
        mapping={(int(r["episode"]),int(r["frame"])):r for r in rows}
        if len(mapping)!=len(rows):raise ValueError("Duplicate IK replay keys")
        if len(mapping)!=len(ep):raise ValueError(f"IK replay must cover ALL {len(ep)} frames")
        seen=set()
        ik=np.empty((len(ep),5),dtype=float)
        for i,(e,f) in enumerate(zip(ep,fr)):
            k=(int(e),int(f))
            if k not in mapping:raise ValueError(f"Missing IK {k}")
            ik[i]=mapping[k]["ik_arm_deg"]
            seen.add(k)
        if set(mapping)!=seen or not np.isfinite(ik).all():raise ValueError("Invalid IK replay")
        ikbad=(ik<lower)|(ik>upper)
        report["ik_replay"]={"frames":len(ik),"urdf_violating_frames":int(ikbad.any(axis=1).sum()),
              "mean_abs_diff_from_recorded":np.abs(ik-goal[:,:5]).mean(axis=0).tolist(),
              "max_abs_diff_from_recorded":np.abs(ik-goal[:,:5]).max(axis=0).tolist()}
        # IK frame-to-frame continuity measured only within episodes.
        per_ep=[]
        for e in sorted(set(ep)):
            ii=np.flatnonzero(ep==e)
            ii=ii[np.argsort(fr[ii])]
            if len(ii)>1:per_ep.append(np.abs(np.diff(ik[ii],axis=0)))
        report["ik_replay"]["max_abs_frame_delta_deg"]=float(max((x.max() for x in per_ep),default=0))
    args.output.mkdir(parents=True)
    with (args.output/"recorded_goal_audit.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]))
        w.writeheader();w.writerows(allrows)
    (args.output/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--split-dataset",type=Path,required=True)
    p.add_argument("--urdf",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--ik-replay",type=Path)
    run(p.parse_args())
if __name__=="__main__":main()
