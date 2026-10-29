#!/usr/bin/env python3
"""Compute matched train-only 6D/7D ACT stats from an audited comparison plan.

Only reads datasets, samples common training image frames; does not train or
modify metadata. Outputs one JSON file for downstream preparation.
"""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

FEATURES=("action","observation.state")
IMAGES=("observation.images.side","observation.images.wrist")

def stats(x,images=False):
    a=np.asarray(x,dtype=np.float64)
    if a.ndim!=2 or a.shape[0]<2 or not np.isfinite(a).all():raise ValueError("Bad stats input")
    if np.any(a.std(axis=0)<=0):raise ValueError("Zero-variance feature")
    def pack(v):return v.reshape(-1,1,1).tolist() if images else v.tolist()
    result={k:pack(v) for k,v in (
        ("mean",a.mean(axis=0)),("std",a.std(axis=0)),
        ("min",a.min(axis=0)),("max",a.max(axis=0)))}
    result["count"]=[len(a)]
    return result

def tabular(root,train,dim):
    result={f:[] for f in FEATURES}
    n=0
    for file in sorted((root/"data").rglob("*.parquet")):
        t=pq.read_table(file,columns=["episode_index",*FEATURES])
        e=np.asarray(t["episode_index"].to_pylist(),dtype=int)
        keep=np.isin(e,list(train))
        n+=int(keep.sum())
        for k in FEATURES:
            x=np.asarray(t[k].to_pylist(),dtype=float)
            expected=dim if k=="action" else 6
            if x.ndim!=2 or x.shape[1]!=expected:raise ValueError(f"Invalid {k}: {file}")
            result[k].append(x[keep])
    return {k:stats(np.concatenate(v)) for k,v in result.items()},n

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=Path("outputs/so101-act-comparison-plan-v1/comparison_plan.json"))
    p.add_argument("--output",type=Path,default=Path("outputs/so101-act-comparison-trainonly-stats-v1.json"))
    p.add_argument("--image-samples",type=int,default=128)
    p.add_argument("--seed",type=int,default=1000)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if not 16<=a.image_samples<=1024:raise ValueError("image-samples must be 16..1024")
    plan=json.loads(a.plan.read_text())
    if plan["status"]!="COMPARISON_PLAN_ONLY_NOT_TRAINED" or not plan["same_recorded_observations"]:
        raise ValueError("Comparison audit not passed")
    train=set(map(int,plan["train_episode_indices"]))
    val=set(map(int,plan["validation_episode_indices"]))
    if len(train)!=45 or len(val)!=8 or train & val or train|val!=set(range(53)):
        raise ValueError("Unexpected split")
    root6=Path(plan["joint_dataset"]);root7=Path(plan["cartesian_dataset"])
    s6,n6=tabular(root6,train,6)
    s7,n7=tabular(root7,train,7)
    if n6!=n7 or n6!=plan["train_frames"]:raise ValueError("Frame-count mismatch")
    for field in ("mean","std","min","max"):
        if not np.allclose(s6["observation.state"][field],s7["observation.state"][field],atol=0,rtol=0):
            raise ValueError("Recorded state stats differ")
    os.environ["HF_HUB_OFFLINE"]="1";os.environ["HF_DATASETS_OFFLINE"]="1"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds=LeRobotDataset("local/act_comparison_stats",root=root6,video_backend="pyav")
    eps=np.asarray(ds.hf_dataset["episode_index"],dtype=int)
    selected=np.flatnonzero(np.isin(eps,list(train)))
    if len(selected)!=n6:raise ValueError("LeRobot frame order mismatch")
    rng=np.random.default_rng(a.seed)
    ix=rng.choice(selected,size=min(a.image_samples,len(selected)),replace=False)
    # Read camera resolution from dataset metadata, not a fixed 168x224 recipe.
    # Accumulate pixel moments in float64; do not retain hundreds of decoded images.
    accum={key:{"sum":np.zeros(3),"sum_sq":np.zeros(3),
                 "min":np.full(3,np.inf),"max":np.full(3,-np.inf),
                 "pixels":0,"shape":None} for key in IMAGES}
    for i in ix:
        sample=ds[int(i)]
        for key in IMAGES:
            ar=sample[key].detach().cpu().numpy()
            declared=tuple(ds.features[key]["shape"])
            # LeRobot metadata declares HWC, but decoded tensors are CHW.
            expected=(declared[2],declared[0],declared[1]) if len(declared)==3 and declared[2]==3 else declared
            if ar.ndim!=3 or ar.shape[0]!=3 or tuple(ar.shape)!=expected:
                raise ValueError(f"Image shape mismatch {key}: got CHW {ar.shape}, declared {declared}, expected {expected}")
            if not np.isfinite(ar).all():raise ValueError(f"Nonfinite image {key}")
            a=accum[key]
            if a["shape"] is None:a["shape"]=list(ar.shape)
            if a["shape"]!=list(ar.shape):raise ValueError(f"Inconsistent camera shape {key}")
            x=ar.astype(np.float64,copy=False).reshape(3,-1)
            a["sum"]+=x.sum(axis=1)
            a["sum_sq"]+=np.square(x).sum(axis=1)
            a["min"]=np.minimum(a["min"],x.min(axis=1))
            a["max"]=np.maximum(a["max"],x.max(axis=1))
            a["pixels"]+=x.shape[1]
    shared={}
    image_shapes={}
    for key, image_accum in accum.items():
        mean=image_accum["sum"]/image_accum["pixels"]
        std=np.sqrt(np.maximum(0,image_accum["sum_sq"]/image_accum["pixels"]-mean*mean))
        if np.any(std<=0):raise ValueError(f"Zero variance in {key}")
        shared[key]={
            "mean":mean.reshape(3,1,1).tolist(),
            "std":std.reshape(3,1,1).tolist(),
            "min":image_accum["min"].reshape(3,1,1).tolist(),
            "max":image_accum["max"].reshape(3,1,1).tolist(),
            "count":[len(ix)]}
        image_shapes[key]=image_accum["shape"]
    s6.update(shared);s7.update(shared)
    out=dict(status="TRAIN_ONLY_STATS_AUDITED_NOT_APPLIED",train_episode_indices=sorted(train),
      validation_episode_indices=sorted(val),train_frames=n6,image_sample_frames=len(ix),
      image_seed=a.seed,image_stats_shared=True,image_shapes=image_shapes,
      joint_stats=s6,cartesian_stats=s7,
      note="Read-only. Must explicitly validate training loader episode IDs before training.")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:out[k] for k in ("status","train_frames","image_sample_frames","image_stats_shared")},indent=2))
    print("Saved:",a.output)
if __name__=="__main__":main()
