#!/usr/bin/env python3
"""Find J6 action transitions in TRAIN episodes and render side/wrist review sheets.

Candidate transitions are NOT automatically grasp/lift/release labels.
Read-only; preserves the original fixed 45/8 split. LeRobot 0.6.2 local dataset.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def img(t):
    a=t.detach().cpu().numpy()
    if a.ndim==3 and a.shape[0] in (1,3,4):a=np.moveaxis(a,0,-1)
    if np.issubdtype(a.dtype,np.floating):
        a=np.clip(a,0,1)*255
    a=a.astype("uint8")
    if a.ndim==3 and a.shape[2]==1:a=a[:,:,0]
    return Image.fromarray(a).convert("RGB")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--split-dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--delta",type=float,default=2.0,help="Absolute J6 action difference per frame, calibrated units")
    p.add_argument("--smooth",type=int,default=5,help="Odd-number moving-average window")
    p.add_argument("--min-gap",type=int,default=20,help="Minimum separation between transitions, frames")
    p.add_argument("--max-per-episode",type=int,default=12)
    p.add_argument("--offsets",type=int,nargs="+",default=[-20,-10,0,10,20],help="Frames from a transition")
    a=p.parse_args()
    if a.output.exists():p.error("Output directory already exists")
    if a.smooth<1 or a.smooth%2!=1 or a.min_gap<1 or a.max_per_episode<1:p.error("Invalid parameters")
    if a.delta<=0:p.error("--delta must be positive")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    split=json.loads((a.split_dataset/"episode_split.json").read_text())
    train=sorted(map(int,split["train_episode_indices"]))
    val=sorted(map(int,split["validation_episode_indices"]))
    if len(train)!=45 or len(val)!=8 or set(train)&set(val) or set(train)|set(val)!=set(range(53)):
        raise ValueError("Unexpected split")
    ds=LeRobotDataset("local/muhi_joint6_stage2a_audit",root=a.dataset,video_backend="pyav")
    if len(ds)!=30846:raise ValueError("Unexpected total frames")
    eps=np.asarray(ds.hf_dataset["episode_index"],dtype=int)
    fis=np.asarray(ds.hf_dataset["frame_index"],dtype=int)
    if set(eps)!=set(range(53)):raise ValueError("Episode count mismatch")
    # parquet-backed Arrow avoids expensive video decode during transition search.
    actions=np.asarray(ds.hf_dataset["action"],dtype=np.float64)
    if actions.shape!=(len(ds),6):raise ValueError("Unexpected 6D actions")
    if not np.isfinite(actions).all():raise ValueError("Nonfinite actions")
    a.output.mkdir(parents=True)
    candidates=[];samples=[]
    for ep in train:
        ii=np.where(eps==ep)[0]
        ii=ii[np.argsort(fis[ii])]
        if not len(ii):raise ValueError(f"Empty episode {ep}")
        if not np.array_equal(fis[ii],np.arange(len(ii))):
            raise ValueError(f"Noncontiguous frame indices episode {ep}")
        v=actions[ii,5]
        pad=np.pad(v,(a.smooth//2,a.smooth//2),mode="edge")
        smooth=np.convolve(pad,np.ones(a.smooth)/a.smooth,mode="valid")
        change=np.diff(smooth,prepend=smooth[0])
        # Candidate point = large local rate of change, not a semantics class.
        ranked=np.argsort(-np.abs(change))
        picked=[]
        for f in ranked:
            if abs(change[f])<a.delta:break
            if f==0 or any(abs(int(f)-x)<a.min_gap for x in picked):continue
            picked.append(int(f))
            if len(picked)>=a.max_per_episode:break
        for frame in sorted(picked):
            entry={"episode":ep,"frame":frame,"time_s":round(frame/float(ds.fps),3),
                "j6_before":float(v[max(0,frame-1)]),"j6_at":float(v[frame]),
                "smoothed_delta":float(change[frame]),"direction":"rising" if change[frame]>0 else "falling",
                "manual_phase":"UNREVIEWED","manual_notes":""}
            candidates.append(entry)
            # Frames, side view and wrist view in a single contact sheet.
            tilew,tileh=260,228
            canvas=Image.new("RGB",(tilew*len(a.offsets),tileh*2+48),"white")
            draw=ImageDraw.Draw(canvas)
            for c,off in enumerate(a.offsets):
                local=min(max(frame+off,0),len(ii)-1)
                datum=ds[int(ii[local])]
                draw.text((c*tilew+5,5),f"ep{ep:02d} frame{local:04d} J6={v[local]:.2f}",fill="black")
                for j,cam in enumerate(("side","wrist")):
                    key="observation.images."+cam
                    photo=img(datum[key])
                    photo.thumbnail((tilew-10,tileh-35))
                    canvas.paste(photo,(c*tilew+5,25+j*tileh))
                    draw.text((c*tilew+5,26+j*tileh),cam,fill="blue")
            file=f"ep_{ep:02d}_frame_{frame:04d}.jpg"
            canvas.save(a.output/file,quality=88)
            entry["contact_sheet"]=file
            # Propose *candidate* center frame, not training windows crossing episode boundary.
            samples.append({"episode":ep,"candidate_frame":frame,"start_frame":max(0,frame-50),
                "end_frame_exclusive":min(len(ii),frame+51),"manual_phase":"UNREVIEWED"})
    fields=["episode","frame","time_s","j6_before","j6_at","smoothed_delta","direction","manual_phase","manual_notes","contact_sheet"]
    with (a.output/"candidates.csv").open("w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=fields);w.writeheader();w.writerows(candidates)
    with (a.output/"proposed_windows.csv").open("w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=list(samples[0]) if samples else ["episode","candidate_frame","start_frame","end_frame_exclusive","manual_phase"])
        w.writeheader();w.writerows(samples)
    summary={"train_episode_count":len(train),"validation_episode_indices":val,
      "candidate_count":len(candidates),"threshold_delta":a.delta,"smooth_window":a.smooth,
      "min_gap":a.min_gap,"fps":float(ds.fps),
      "notes":["Only train45 are inspected. Heldout8 must NEVER enter fine-tuning.",
      "The candidate transitions are unsigned action changes, NOT physical gripper open/close labels.",
      "Review contact sheets and label grasp/release by watching paired video and J6 calibration.",
      "proposed_windows.csv is a candidate list, NOT a LeRobot training dataset."]}
    (a.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
