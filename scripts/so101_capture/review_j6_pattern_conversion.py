#!/usr/bin/env python3
"""Read-only episode review for J6 early-opening conversion, no synthetic labels.

Outputs J6/J1-J5 movement windows and paired camera contact sheet for ONE
train episode. Does not classify with certainty or alter Dataset.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

def load(root):
    grouped={}
    for f in sorted((root/"data").rglob("*.parquet")):
        t=pq.read_table(f,columns=["episode_index","frame_index","observation.state","action"])
        for e,k,s,a in zip(*[t[n].to_pylist() for n in ("episode_index","frame_index","observation.state","action")]):
            if len(s)!=6 or len(a)!=7: raise ValueError("Feature dimension mismatch")
            grouped.setdefault(int(e),[]).append((int(k),np.array(s,dtype=float),float(a[6])))
    for e,rows in grouped.items():
        rows.sort(key=lambda q:q[0])
        if [r[0] for r in rows]!=list(range(len(rows))): raise ValueError(f"Bad frame order {e}")
    return grouped

def analyze(rows, arm_tolerance, min_j6_delta):
    state=np.stack([r[1] for r in rows]);j6=state[:,5];arm=state[:,:5]
    da=np.max(np.abs(np.diff(arm,axis=0)),axis=1)
    dg=np.diff(j6)
    moving=np.abs(dg)>=min_j6_delta
    windows=[]
    active=np.flatnonzero(moving)
    if len(active):
        groups=np.split(active,np.where(np.diff(active)>5)[0]+1)
        for g in groups:
            i=int(g[0]);j=min(len(rows)-1,int(g[-1]+1))
            windows.append({"start":i,"end":j,"j6_change":float(j6[j]-j6[i]),
                "j6_abs_change":float(np.abs(dg[i:j]).sum()),
                "arm_peak_step":float(da[i:j].max()) if j>i else 0.0,
                "arm_static_by_step":bool(np.all(da[i:j]<=arm_tolerance))})
    return state,windows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--episode",type=int,default=None,help="Optional train episode contact sheet")
    p.add_argument("--arm-step-tolerance",type=float,default=0.5)
    p.add_argument("--j6-step-threshold",type=float,default=0.4)
    a=p.parse_args()
    root=a.dataset.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError(f"Will not overwrite {out}")
    if out==root or root in out.parents:raise ValueError("Output must be outside Dataset")
    split=json.loads((root/"episode_split.json").read_text())
    train=set(map(int,split["train_episode_indices"]))
    val=set(map(int,split["validation_episode_indices"]))
    data=load(root)
    if set(data)!=train|val or train&val:raise ValueError("Split inconsistent")
    if a.episode is not None and a.episode not in train:
        raise ValueError("Preview episode must be TRAIN (avoid validation leakage)")
    overview=[];segments=[]
    for e,rows in sorted(data.items()):
        state,wins=analyze(rows,a.arm_step_tolerance,a.j6_step_threshold)
        j6=state[:,5]
        overview.append({"episode":e,"split":"train" if e in train else "validation",
            "frames":len(rows),"j6_initial":float(j6[0]),"j6_min":float(j6.min()),
            "j6_max":float(j6.max()),"movement_windows":len(wins),
            "arm_stationary_j6_windows":sum(w["arm_static_by_step"] for w in wins)})
        for w in wins:segments.append({"episode":e,**w})
    out.mkdir(parents=True)
    for name,rows in (("episodes.csv",overview),("j6_segments.csv",segments)):
        if rows:
            with (out/name).open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report={"train_episodes":len(train),"validation_episodes":len(val),
      "preview_episode":a.episode,"arm_step_tolerance":a.arm_step_tolerance,
      "j6_step_threshold":a.j6_step_threshold,
      "warning":"Candidate windows reflect signal changes only; J6 open/close polarity is not assumed; check camera frames before labeling pattern 1/2."}
    if a.episode is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        e=a.episode;state,wins=analyze(data[e],a.arm_step_tolerance,a.j6_step_threshold)
        j6=state[:,5];fig,ax=plt.subplots(figsize=(13,3))
        ax.plot(np.arange(len(j6))/20,j6,label="J6 observation.state")
        for w in wins:
            ax.axvspan(w["start"]/20,w["end"]/20,alpha=.16,
                       color="green" if w["arm_static_by_step"] else "orange")
        ax.set_xlabel("Seconds");ax.set_ylabel("J6 state units");ax.legend();fig.tight_layout()
        fig.savefig(out/f"episode_{e:03d}_j6.png",dpi=130);plt.close(fig)
        import os
        os.environ["HF_HUB_OFFLINE"]="1";os.environ["HF_DATASETS_OFFLINE"]="1"
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds=LeRobotDataset("local/muhi_cartesian_pattern_review",root=root,video_backend="pyav")
        episodes=np.array(ds.hf_dataset["episode_index"],dtype=int)
        frames=np.array(ds.hf_dataset["frame_index"],dtype=int)
        lookup={int(frames[i]):int(i) for i in np.flatnonzero(episodes==e)}
        picks={0,len(data[e])-1}
        # Two frames before/after each J6 movement start/end, limited to first 3 windows.
        for w in wins[:3]:
            for idx in (w["start"]-15,w["start"],w["end"],w["end"]+15):
                picks.add(int(np.clip(idx,0,len(data[e])-1)))
        picks=sorted(picks)
        fig,axs=plt.subplots(len(picks),2,figsize=(7,2.6*len(picks)),squeeze=False)
        for ri,k in enumerate(picks):
            item=ds[lookup[k]]
            for ci,key in enumerate(("observation.images.side","observation.images.wrist")):
                im=item[key].cpu().numpy().transpose(1,2,0)
                axs[ri,ci].imshow(np.clip(im,0,1))
                axs[ri,ci].set_title(f"ep {e} frame {k} | {key.split('.')[-1]} | J6={j6[k]:.2f}",fontsize=9)
                axs[ri,ci].axis("off")
        fig.tight_layout();fig.savefig(out/f"episode_{e:03d}_cameras.png",dpi=120);plt.close(fig)
        report["preview_frames"]=picks
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
