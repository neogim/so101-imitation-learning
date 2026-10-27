#!/usr/bin/env python3
"""Compare J6 predictions between ACT checkpoints on identically sampled heldout frames.

Outputs per-episode line charts, metrics by checkpoint, and a grasp event diagnostic.
No robot action is performed. Threshold crossings are only proxies, NOT grasp success.
"""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

def read(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--eval-dir",action="append",required=True,help="Evaluation directory; pass for 10k/16k/20k")
    p.add_argument("--label",action="append",required=True,help="Same order as --eval-dir")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--threshold",type=float,default=25)
    a=p.parse_args()
    if len(a.eval_dir)!=len(a.label) or len(set(a.label))!=len(a.label):
        p.error("Each evaluation directory needs one distinct label")
    if a.output.exists():p.error("Output already exists: "+str(a.output))
    datasets={}
    for label,d in zip(a.label,a.eval_dir):
        path=Path(d)
        if not (path/"metrics.json").is_file() or not (path/"predictions.csv").is_file():
            p.error("Missing evaluation output in "+str(path))
        metrics=json.loads((path/"metrics.json").read_text())
        rows=read(path/"predictions.csv")
        data={}
        for r in rows:
            k=(int(r["episode"]),int(r["frame"]))
            if k in data:raise ValueError("duplicate sample "+str(k))
            data[k]=r
        datasets[label]=(metrics,data)
    labels=list(datasets)
    keys=set(datasets[labels[0]][1])
    if any(set(x[1])!=keys for x in datasets.values()):
        raise ValueError("Different sample frames! Re-evaluate all checkpoints with same stride and limit.")
    for k in keys:
        ground=[float(data[k]["gt_J6"]) for _,data in datasets.values()]
        if max(ground)-min(ground)>1e-4:raise ValueError("GT mismatch "+str(k))
    a.output.mkdir(parents=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:raise SystemExit("pip install matplotlib") from e
    summary={}
    for label,(_,data) in datasets.items():
        gt=np.array([float(data[k]["gt_J6"]) for k in sorted(keys)])
        pred=np.array([float(data[k]["pred_J6"]) for k in sorted(keys)])
        gt_pos=gt>a.threshold;pred_pos=pred>a.threshold
        tp=int((gt_pos & pred_pos).sum());fp=int((~gt_pos & pred_pos).sum());fn=int((gt_pos & ~pred_pos).sum())
        summary[label]={"samples":len(keys),"J6_MAE":float(np.abs(gt-pred).mean()),
            "gt_above_threshold":int(gt_pos.sum()),"pred_above_threshold":int(pred_pos.sum()),
            "threshold_TP":tp,"threshold_FP":fp,"threshold_FN":fn,
            "threshold_precision":tp/(tp+fp) if tp+fp else None,
            "threshold_recall":tp/(tp+fn) if tp+fn else None,
            "J6_bias":float(np.mean(pred-gt))}
    episodes=sorted(set(k[0] for k in keys))
    for ep in episodes:
        ek=sorted((k for k in keys if k[0]==ep),key=lambda x:x[1])
        fig,ax=plt.subplots(figsize=(12,4))
        base=datasets[labels[0]][1]
        ax.plot([k[1] for k in ek],[float(base[k]["gt_J6"]) for k in ek],label="Teacher J6",lw=2,color="black")
        for label,(_,data) in datasets.items():
            ax.plot([k[1] for k in ek],[float(data[k]["pred_J6"]) for k in ek],label=label,alpha=.8)
        ax.axhline(a.threshold,color="gray",ls=":",label="Diagnostic threshold")
        ax.set(xlabel="Frame index (20 fps)",ylabel="J6 calibrated action",title=f"Heldout episode {ep} | teacher-forced fresh chunk t=0")
        ax.grid(alpha=.25);ax.legend();fig.tight_layout()
        fig.savefig(a.output/f"episode_{ep:02d}_J6.png",dpi=160)
        plt.close(fig)
    (a.output/"summary.json").write_text(json.dumps({"threshold_note":"J6 > threshold is NOT verified close/open or actual grasp", "checkpoint_metrics":summary},indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
