#!/usr/bin/env python3
"""Evaluate original 6D ACT checkpoints on eight heldout episodes. No motors.

At each selected dataset frame predict one fresh 50-action chunk and evaluate
the FIRST predicted action. Optional multi-horizon evaluation is separate.
Read-only and no collision or joint-safety certification.
"""
from __future__ import annotations
import argparse,csv,json,os
from pathlib import Path
import numpy as np
import torch

JOINTS=["J1","J2","J3","J4","J5","J6"]
OBS=("observation.state","observation.images.side","observation.images.wrist")
def evaluate(dataset:Path,split_dataset:Path,checkpoint:Path,output:Path,stride:int,per_episode:int):
    dataset,split_dataset,checkpoint,output=[x.resolve() for x in (dataset,split_dataset,checkpoint,output)]
    if output.exists():raise FileExistsError(output)
    if stride<1 or per_episode<1:raise ValueError("stride and per_episode must be positive")
    for key in ("HF_HUB_OFFLINE","HF_DATASETS_OFFLINE","HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key]="1"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    split=json.loads((split_dataset/"episode_split.json").read_text())
    train=set(map(int,split["train_episode_indices"]))
    valid=sorted(map(int,split["validation_episode_indices"]))
    if len(train)!=45 or len(valid)!=8 or train & set(valid) or train | set(valid)!=set(range(53)):
        raise ValueError("Invalid 45/8 train-validation split")
    assert valid==[5,8,15,18,23,31,46,51],valid
    ds=LeRobotDataset("local/muhi_joint6_validation_053",root=dataset,video_backend="pyav")
    if len(ds)!=30846:raise ValueError("Expected 30,846 total frames")
    eps=np.asarray(ds.hf_dataset["episode_index"],dtype=int)
    fis=np.asarray(ds.hf_dataset["frame_index"],dtype=int)
    chosen=[]
    for ep in valid:
        candidates=np.where((eps==ep)&(fis%stride==0))[0]
        chosen.extend(candidates[:per_episode].tolist())
    if not chosen:raise ValueError("No validation frames")
    if any(eps[i] in train for i in chosen):raise ValueError("Train leakage")
    device="mps" if torch.backends.mps.is_available() else "cpu"
    policy=ACTPolicy.from_pretrained(str(checkpoint),local_files_only=True).to(device).eval()
    pre,post=make_pre_post_processors(policy_cfg=policy.config,pretrained_path=str(checkpoint),
                  preprocessor_overrides={"device_processor":{"device":device}})
    rows=[]
    with torch.inference_mode():
        for idx in chosen:
            item=ds[idx]
            obs={k:item[k].unsqueeze(0) for k in OBS}
            pred_chunk=policy.predict_action_chunk(pre(obs))
            if pred_chunk.ndim!=3 or pred_chunk.shape[-1]!=6:
                raise ValueError(f"Unexpected shape {tuple(pred_chunk.shape)}")
            p=post(pred_chunk[:,0,:]).detach().cpu().numpy().reshape(-1)
            y=item["action"].detach().cpu().numpy().reshape(-1)
            if p.shape!=(6,) or y.shape!=(6,) or not np.isfinite(p).all():
                raise ValueError("Bad action output")
            r={"episode":int(eps[idx]),"frame":int(fis[idx])}
            for k,n in enumerate(JOINTS):
                r["gt_"+n]=float(y[k]);r["pred_"+n]=float(p[k]);r["error_"+n]=float(p[k]-y[k])
            rows.append(r)
    errors=np.asarray([[r["error_"+j] for j in JOINTS] for r in rows])
    summary={"checkpoint":str(checkpoint),"mode":"teacher_forced_fresh_chunk_first_action",
      "validation_episode_indices":valid,"samples":len(rows),"stride":stride,
      "units":"LeRobot joint calibration units; not validated as physical degrees",
      "mae":{j:float(np.abs(errors[:,k]).mean()) for k,j in enumerate(JOINTS)},
      "rmse":{j:float(np.sqrt(np.mean(errors[:,k]**2))) for k,j in enumerate(JOINTS)},
      "arm_joint_mae_mean":float(np.abs(errors[:,:5]).mean()),
      "gripper_gt_over25":int(sum(r["gt_J6"]>25 for r in rows)),
      "gripper_pred_over25":int(sum(r["pred_J6"]>25 for r in rows)),
      "notes":["No FK/TCP mm computation: existing URDF-vs-calibration mapping unresolved",
               "No actual gripping or closed-loop success can be inferred",
               "A J6>25 indicator is exploratory; actual close/open direction must be checked",
               "First action only, not 50-step rollout"]}
    output.mkdir(parents=True)
    with (output/"predictions.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/"metrics.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True,type=Path)
    p.add_argument("--split-dataset",required=True,type=Path)
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--stride",type=int,default=20)
    p.add_argument("--per-episode",type=int,default=20)
    a=p.parse_args()
    evaluate(a.dataset,a.split_dataset,a.checkpoint,a.output,a.stride,a.per_episode)
if __name__=="__main__":main()
