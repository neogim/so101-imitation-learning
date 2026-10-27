#!/usr/bin/env python3
"""Prepare human phase-label review CSV, then validate reviewed Stage 2A windows.

This script does NOT edit LeRobot datasets or train a model. It never includes
the fixed eight validation episodes in training manifests.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

PHASES={"UNREVIEWED","approach","grasp_onset","lift","transport","release","other","uncertain"}
FOCUS={"grasp_onset","lift","release"}
def load_csv(p):
    with p.open(newline="") as f:return list(csv.DictReader(f))
def write_csv(p,rows,keys):
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates",type=Path,required=True)
    p.add_argument("--split-dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--reviewed",type=Path,help="Human-annotated review CSV, for validation")
    p.add_argument("--before",type=int,default=50)
    p.add_argument("--after",type=int,default=50)
    a=p.parse_args()
    if a.output.exists():p.error("Output already exists")
    split=json.loads((a.split_dataset/"episode_split.json").read_text())
    train=set(map(int,split["train_episode_indices"]))
    valid=set(map(int,split["validation_episode_indices"]))
    if len(train)!=45 or len(valid)!=8 or train&valid or train|valid!=set(range(53)):
        raise ValueError("Wrong 45/8 split")
    if a.before<0 or a.after<0:raise ValueError("Negative window")
    rows=load_csv(a.reviewed or a.candidates)
    result=[];used=set()
    for r in rows:
        ep=int(r["episode"]);frame=int(r["frame"])
        if ep not in train:raise ValueError(f"Non-train episode: {ep}")
        if frame<0:raise ValueError("Negative frame")
        k=(ep,frame)
        if k in used:raise ValueError(f"Duplicated sample {k}")
        used.add(k)
        phase=r.get("manual_phase","UNREVIEWED").strip() or "UNREVIEWED"
        if phase not in PHASES:raise ValueError(f"Unknown phase {phase}; choices: {sorted(PHASES)}")
        result.append({"episode":ep,"frame":frame,
            "j6_at":r.get("j6_at",""),"direction":r.get("direction",""),
            "contact_sheet":r.get("contact_sheet",""),"manual_phase":phase,
            "manual_notes":r.get("manual_notes",""),
            "start_frame":max(0,frame-a.before),
            "end_frame_exclusive_uncapped":frame+a.after+1})
    a.output.mkdir(parents=True)
    review_fields=list(result[0]) if result else ["episode","frame","manual_phase"]
    write_csv(a.output/"review.csv",result,review_fields)
    # Only genuinely human-reviewed, confident grasp phase can become a focus window.
    approved=[r for r in result if r["manual_phase"] in FOCUS]
    write_csv(a.output/"approved_focus_windows.csv",approved,review_fields)
    summary={"candidate_count":len(result),"approved_focus_count":len(approved),
        "validation_episodes_excluded":sorted(valid),
        "phase_counts":{ph:sum(r["manual_phase"]==ph for r in result) for ph in sorted(PHASES)},
        "warnings":["If unreviewed, approved_focus_count must remain zero",
            "Frame end is not bounded to actual episode length: clamp when building sampler",
            "Approved windows are NOT a trainable LeRobot dataset or safe robot commands",
            "ACT future 50-step actions must preserve episode boundaries; no frame reshuffling."]}
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
