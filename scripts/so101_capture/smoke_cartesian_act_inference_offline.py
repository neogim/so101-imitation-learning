#!/usr/bin/env python3
"""Offline ACT fresh-chunk t=0 evaluation for the 53-episode 7D dataset; NO hardware."""
import argparse,csv,json,os
from pathlib import Path
import numpy as np
import torch

NAMES=("dx_m","dy_m","dz_m","drx_rad","dry_rad","drz_rad","gripper")
OBS=("observation.state","observation.images.side","observation.images.wrist")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,default=Path("outputs/so101-cartesian-act-dataset-v2"))
    p.add_argument("--checkpoint",type=Path,default=Path("outputs/so101-cartesian-act-smoke-v1/checkpoints/000005/pretrained_model"))
    p.add_argument("--output",type=Path,default=Path("outputs/so101-cartesian-act-offline-pred-v1"))
    p.add_argument("--episode",type=int,default=0)
    p.add_argument("--stride",type=int,default=100)
    p.add_argument("--max-frames",type=int,default=8)
    a=p.parse_args()
    for k in ("HF_HUB_OFFLINE","HF_DATASETS_OFFLINE","HF_HUB_DISABLE_TELEMETRY"):os.environ[k]="1"
    if a.output.exists():raise FileExistsError(f"Output already exists: {a.output}")
    if a.stride<1 or not 1<=a.max_frames<=100:raise ValueError("Invalid stride/max-frames")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    ds=LeRobotDataset("local/so101_cartesian_act_experimental",root=a.dataset.resolve(),video_backend="pyav")
    if tuple(ds.features["action"]["shape"])!=(7,):raise ValueError("Not a 7D Cartesian dataset")
    eps=np.asarray(ds.hf_dataset["episode_index"],dtype=int)
    frames=np.asarray(ds.hf_dataset["frame_index"],dtype=int)
    indices=[int(i) for i in np.flatnonzero((eps==a.episode)&(frames%a.stride==0))[:a.max_frames]]
    if not indices:raise ValueError(f"No frames for episode {a.episode}")
    ck=a.checkpoint.resolve()
    if not ck.is_dir():raise FileNotFoundError(ck)
    device="mps" if torch.backends.mps.is_available() else "cpu"
    policy=ACTPolicy.from_pretrained(str(ck),local_files_only=True)
    policy.to(device).eval()
    if set(policy.config.input_features)!=set(OBS) or tuple(policy.config.output_features["action"].shape)!=(7,):
        raise ValueError("Checkpoint feature mismatch")
    pre,post=make_pre_post_processors(policy_cfg=policy.config,pretrained_path=str(ck),
      preprocessor_overrides={"device_processor":{"device":device}})
    rows=[]
    with torch.inference_mode():
        for idx in indices:
            item=ds[idx]
            obs={k:item[k].unsqueeze(0) for k in OBS}
            pred=policy.predict_action_chunk(pre(obs))
            if pred.ndim!=3 or pred.shape[-1]!=7:raise ValueError(f"Bad chunk shape: {tuple(pred.shape)}")
            yhat=post(pred[:,0,:]).detach().cpu().numpy().reshape(-1)
            goal=item["action"].cpu().numpy().reshape(-1)
            state=item["observation.state"].cpu().numpy().reshape(-1)
            if any(x.shape!=shape or not np.isfinite(x).all() for x,shape in ((yhat,(7,)),(goal,(7,)),(state,(6,)))):
                raise ValueError("Non-finite/malformed data")
            row=dict(episode_index=int(eps[idx]),frame_index=int(frames[idx]))
            row.update({f"state_{j}":float(x) for j,x in enumerate(state)})
            row.update({f"pred_{n}":float(v) for n,v in zip(NAMES,yhat)})
            row.update({f"gt_{n}":float(v) for n,v in zip(NAMES,goal)})
            rows.append(row)
    a.output.mkdir(parents=True)
    with (a.output/"predictions.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    errors=np.array([[r[f"pred_{n}"]-r[f"gt_{n}"] for n in NAMES] for r in rows])
    summary=dict(status="OFFLINE_ONLY_NOT_HARDWARE_APPROVED",dataset=str(a.dataset),checkpoint=str(ck),
       device=device,episode=a.episode,frames=len(rows),first_action_only=True,
       prediction_shape=[7],mae={n:float(v) for n,v in zip(NAMES,np.abs(errors).mean(axis=0))},
       note="5-step checkpoint is untrained for meaningful accuracy; results are pipeline smoke only.")
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
