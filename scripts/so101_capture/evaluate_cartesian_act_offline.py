#!/usr/bin/env python3
"""Offline ACT evaluation, FIRST action of a fresh chunk per held-out frame.

This is teacher-forced, one-step imitation evaluation, NOT a closed-loop roll-out.
No hardware or IK. Do not use select_action's stateful queue here.
"""
from __future__ import annotations
import argparse, csv, json, os
from pathlib import Path
import numpy as np
import torch

NAMES=["dx","dy","dz","drx","dry","drz","gripper"]
OBS=("observation.state","observation.images.side","observation.images.wrist")

def evaluate(root:Path, checkpoint:Path, output:Path, stride:int, limit:int):
    root,checkpoint,output=[p.resolve() for p in (root,checkpoint,output)]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if stride < 1 or limit < 1:
        raise ValueError("stride and limit must be positive")
    for var in ("HF_HUB_OFFLINE","HF_DATASETS_OFFLINE","HF_HUB_DISABLE_TELEMETRY"):
        os.environ[var]="1"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    split=json.loads((root/"episode_split.json").read_text())
    train=set(map(int,split["train_episode_indices"]))
    valid=set(map(int,split["validation_episode_indices"]))
    if len(train)!=45 or len(valid)!=8 or train & valid or train | valid!=set(range(53)):
        raise ValueError("Invalid split")
    ds=LeRobotDataset("local/muhi_cartesian_eval_053",root=root,video_backend="pyav")
    if len(ds)!=30846:
        raise ValueError("Unexpected frame count")
    # NOTE: hf_dataset is raw complete Dataset, not a filtered subset
    episodes=np.asarray(ds.hf_dataset["episode_index"],dtype=int)
    frames=np.asarray(ds.hf_dataset["frame_index"],dtype=int)
    indices=[i for i,e in enumerate(episodes) if e in valid and frames[i]%stride==0]
    if limit:
        # Equal-frequency cap by episode to ensure each held-out trajectory is represented.
        per_episode=max(1,limit//len(valid))
        selected=[]
        for e in sorted(valid):
            selected.extend([i for i in indices if episodes[i]==e][:per_episode])
        indices=sorted(selected)
    if not indices or any(episodes[i] in train for i in indices):
        raise ValueError("Validation selection empty or leaked into train")

    device="mps" if torch.backends.mps.is_available() else "cpu"
    policy=ACTPolicy.from_pretrained(str(checkpoint),local_files_only=True)
    policy.to(device).eval()
    pre,post=make_pre_post_processors(policy_cfg=policy.config,
                                      pretrained_path=str(checkpoint),
                                      preprocessor_overrides={"device_processor":{"device":device}})
    output.mkdir(parents=True)
    rows=[]
    with torch.inference_mode():
        for count,i in enumerate(indices):
            item=ds[int(i)]
            obs={k:item[k].unsqueeze(0) for k in OBS}
            normalized=pre(obs)
            predicted=policy.predict_action_chunk(normalized)
            if predicted.ndim!=3 or predicted.shape[-1]!=7:
                raise ValueError(f"Unexpected prediction shape {tuple(predicted.shape)}")
            # The processor applies the same inverse normalization as ACT inference.
            yhat=post(predicted[:,0,:]).detach().cpu().numpy().reshape(-1)
            y=item["action"].detach().cpu().numpy().reshape(-1)
            if yhat.shape!=(7,) or y.shape!=(7,) or not np.isfinite(yhat).all():
                raise ValueError(f"Bad prediction at frame {i}")
            row={"episode_index":int(episodes[i]),"frame_index":int(frames[i])}
            for j,n in enumerate(NAMES):
                row[f"gt_{n}"]=float(y[j])
                row[f"pred_{n}"]=float(yhat[j])
                row[f"error_{n}"]=float(yhat[j]-y[j])
            rows.append(row)
            if (count+1)%50==0:
                print(f"Evaluated {count+1}/{len(indices)} frames",flush=True)
    fields=list(rows[0])
    with (output/"predictions.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    errors=np.array([[r[f"error_{n}"] for n in NAMES] for r in rows])
    report={"checkpoint":str(checkpoint),"episodes":sorted(valid),"n_samples":len(rows),
            "stride":stride,"mode":"teacher_forced_fresh_chunk_first_action_only",
            "units":["m","m","m","rad","rad","rad","gripper_0_100"],
            "mae":{n:float(v) for n,v in zip(NAMES,np.abs(errors).mean(axis=0))},
            "rmse":{n:float(v) for n,v in zip(NAMES,np.sqrt(np.square(errors).mean(axis=0)))},
            "gripper_gt_over25":int(sum(r["gt_gripper"]>25 for r in rows)),
            "gripper_pred_over25":int(sum(r["pred_gripper"]>25 for r in rows))}
    (output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for n in ("gripper","dx","dy","dz"):
            fig,ax=plt.subplots(figsize=(12,3))
            xs=np.arange(len(rows))
            ax.plot(xs,[r[f"gt_{n}"] for r in rows],label="Ground truth")
            ax.plot(xs,[r[f"pred_{n}"] for r in rows],label="ACT prediction",alpha=.8)
            ax.set_title(n+" | validation only | fresh-chunk t=0")
            ax.set_xlabel("Sample index (8 episodes concatenated)")
            ax.legend();fig.tight_layout()
            fig.savefig(output/f"{n}.png",dpi=140);plt.close(fig)
    except ImportError:
        print("matplotlib missing: CSV and JSON were saved, graphs skipped")
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True,type=Path)
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--stride",type=int,default=20)
    p.add_argument("--limit",type=int,default=160)
    a=p.parse_args()
    print(json.dumps(evaluate(a.dataset,a.checkpoint,a.output,a.stride,a.limit),indent=2))
if __name__=="__main__":main()
