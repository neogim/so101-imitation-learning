#!/usr/bin/env python3
"""Evaluate SO-101 ACT Joint6/Cartesian7 baselines and DR on identical 160 frames.

Offline only. No motor or camera access. Evaluates clean images, not an
independent holdout unless training provenance proves episode exclusion.
"""
from __future__ import annotations
import argparse,csv,json,os
from pathlib import Path

EPISODES=(5,8,15,18,23,31,46,51)
FRAMES=20
LABELS={
 "joint6":("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"),
 "cartesian7":("dx_m","dy_m","dz_m","drx_rad","dry_rad","drz_rad","gripper"),
}
OBS=("observation.state","observation.images.side","observation.images.wrist")
DEFAULT_CHECKPOINTS={
 "joint6_baseline":"outputs/so101-act-comparison-20k-joint6/checkpoints/020000/pretrained_model",
 "cartesian7_baseline":"outputs/so101-act-comparison-20k-cartesian7/checkpoints/020000/pretrained_model",
}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--joint-dataset",type=Path,default=Path("outputs/so101-act-comparison-staged-v1/joint6"))
 p.add_argument("--cart-dataset",type=Path,default=Path("outputs/so101-act-comparison-staged-v1/cartesian7"))
 p.add_argument("--joint-baseline",type=Path,default=Path(DEFAULT_CHECKPOINTS["joint6_baseline"]))
 p.add_argument("--cart-baseline",type=Path,default=Path(DEFAULT_CHECKPOINTS["cartesian7_baseline"]))
 p.add_argument("--joint-dr",type=Path,required=True)
 p.add_argument("--cart-dr",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 p.add_argument("--device",choices=("mps","cpu"),default="mps")
 a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 for k in ("HF_HUB_OFFLINE","HF_DATASETS_OFFLINE","HF_HUB_DISABLE_TELEMETRY"):os.environ[k]="1"
 import numpy as np
 import torch
 from lerobot.datasets.lerobot_dataset import LeRobotDataset
 from lerobot.policies.act.modeling_act import ACTPolicy
 from lerobot.policies.factory import make_pre_post_processors
 if a.device=="mps" and not torch.backends.mps.is_available():raise RuntimeError("MPS unavailable")
 models=[
  ("joint6_baseline","joint6",a.joint_dataset,a.joint_baseline),
  ("joint6_dr","joint6",a.joint_dataset,a.joint_dr),
  ("cartesian7_baseline","cartesian7",a.cart_dataset,a.cart_baseline),
  ("cartesian7_dr","cartesian7",a.cart_dataset,a.cart_dr)]
 output_rows=[];summary={"status":"OFFLINE_TEACHER_FORCED_CLEAN_INPUTS","episodes":list(EPISODES),
   "frames_per_episode":FRAMES,"held_out_status":"UNVERIFIED: training split/leakage must be audited",
   "models":{}}
 selection_reference=None
 for model,kind,dataset_path,checkpoint in models:
  if not dataset_path.joinpath("meta/info.json").is_file():raise FileNotFoundError(dataset_path)
  if not checkpoint.is_dir():raise FileNotFoundError(checkpoint)
  ds=LeRobotDataset("local/so101_dr_eval_"+kind,root=dataset_path.resolve(),video_backend="pyav")
  dims=len(LABELS[kind])
  if tuple(ds.features["action"]["shape"])!=(dims,):raise ValueError(f"{model} action dims")
  eps=np.asarray(ds.hf_dataset["episode_index"],dtype=int)
  frames=np.asarray(ds.hf_dataset["frame_index"],dtype=int)
  indices=[]
  for ep in EPISODES:
   chosen=np.flatnonzero((eps==ep)&(frames>=0)&(frames<FRAMES))
   pairs=sorted((int(frames[i]),int(i)) for i in chosen)
   if [f for f,i in pairs]!=list(range(FRAMES)):raise ValueError(f"{model}: missing or duplicate frames in episode {ep}")
   indices.extend((ep,frame,idx) for frame,idx in pairs)
  keys=[(ep,f) for ep,f,i in indices]
  if selection_reference is not None and selection_reference!=keys:raise ValueError("Frame selection differs across datasets")
  selection_reference=keys
  policy=ACTPolicy.from_pretrained(str(checkpoint.resolve()),local_files_only=True)
  policy.to(a.device).eval()
  if set(policy.config.input_features)!=set(OBS) or tuple(policy.config.output_features["action"].shape)!=(dims,):
   raise ValueError(f"{model}: incompatible model features")
  pre,post=make_pre_post_processors(policy_cfg=policy.config,pretrained_path=str(checkpoint.resolve()),
                  preprocessor_overrides={"device_processor":{"device":a.device}})
  errors=[]
  with torch.inference_mode():
   for ep,fr,idx in indices:
    item=ds[idx]
    obs={k:item[k].unsqueeze(0) for k in OBS}
    # reset prevents cached trajectory chunks from leaking between unrelated observations
    policy.reset()
    chunk=policy.predict_action_chunk(pre(obs))
    if chunk.ndim!=3 or chunk.shape[-1]!=dims:raise ValueError(f"{model}: unexpected action chunk shape")
    pred=post(chunk[:,0,:]).detach().cpu().numpy().reshape(-1)
    gt=item["action"].cpu().numpy().reshape(-1)
    if pred.shape!=(dims,) or gt.shape!=(dims,) or not np.isfinite(pred).all() or not np.isfinite(gt).all():
     raise ValueError(f"{model}: nonfinite predictions")
    err=pred-gt;errors.append(err)
    output_rows.append({"model":model,"representation":kind,"episode":ep,"frame":fr,
     **{f"pred_{label}":float(x) for label,x in zip(LABELS[kind],pred)},
     **{f"gt_{label}":float(x) for label,x in zip(LABELS[kind],gt)}})
  err=np.asarray(errors)
  stat={"checkpoint":str(checkpoint),"dataset":str(dataset_path),"samples":len(indices),
        "MAE":{lab:float(val) for lab,val in zip(LABELS[kind],np.mean(np.abs(err),axis=0))}}
  if kind=="cartesian7":
   xyz=np.linalg.norm(err[:,:3],axis=1)*1000
   stat["xyz_error_mm"]={"mean":float(np.mean(xyz)),"median":float(np.median(xyz)),
                    "p95":float(np.percentile(xyz,95)),"max":float(np.max(xyz))}
  summary["models"][model]=stat
  print(f"{model}: {len(indices)} frames evaluated",flush=True)
 # Do not compute one global MAE over different target representations.
 a.output.mkdir(parents=True)
 keys=sorted(set().union(*(r.keys() for r in output_rows)))
 with (a.output/"predictions.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(output_rows)
 (a.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
