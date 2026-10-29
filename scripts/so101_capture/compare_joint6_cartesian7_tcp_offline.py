#!/usr/bin/env python3
"""Compare 20k Joint6 vs Cartesian7 predictions in a shared, hypothetical URDF TCP metric.
No robot I/O. URDF angle mapping is NOT physically calibrated.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import placo

JOINTS=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll")

def read_csv(path,ep,frame):
    with path.open(newline="") as f:
        rows=list(csv.DictReader(f))
    result={}
    for r in rows:
        key=(int(r[ep]),int(r[frame]))
        if key in result: raise ValueError(f"Duplicate key {key}")
        result[key]=r
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint-csv",type=Path,required=True)
    p.add_argument("--cart-csv",type=Path,required=True)
    p.add_argument("--urdf",type=Path,default=Path("data/urdf/so101/so101_new_calib.urdf"))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    j=read_csv(a.joint_csv,"episode","frame")
    c=read_csv(a.cart_csv,"episode_index","frame_index")
    if set(j)!=set(c):raise ValueError(f"Frame key mismatch: joint={len(j)}, cart={len(c)}, common={len(set(j)&set(c))}")
    if len(j)!=160 or {e for e,f in j}!={5,8,15,18,23,31,46,51}:
        raise ValueError("Unexpected validation subset")
    robot=placo.RobotWrapper(str(a.urdf.resolve()))
    def fk(q_deg):
        for name,value in zip(JOINTS,np.deg2rad(q_deg)):
            robot.set_joint(name,float(value))
        robot.update_kinematics()
        return np.asarray(robot.get_T_world_frame("gripper_frame_link"),float)[:3,3].copy()

    results=[]
    for key in sorted(j):
        jr,cr=j[key],c[key]
        gt=np.array([float(jr[f"gt_J{k}"]) for k in range(1,6)])
        pred=np.array([float(jr[f"pred_J{k}"]) for k in range(1,6)])
        cp=np.array([float(cr[f"pred_d{axis}"]) for axis in "xyz"])
        cg=np.array([float(cr[f"gt_d{axis}"]) for axis in "xyz"])
        if not all(np.isfinite(x).all() for x in (gt,pred,cp,cg)):
            raise ValueError(f"Nonfinite {key}")
        target=fk(gt)
        pred_joint=fk(pred)
        joint_err=float(np.linalg.norm(pred_joint-target)*1000)
        # Since Cartesian7 is a delta action, its common-target error is
        # norm(pred_delta - gt_delta), assuming the saved Cartesian GT delta
        # is FK-consistent with the same GT joint action.
        cart_err=float(np.linalg.norm(cp-cg)*1000)
        results.append({"episode":key[0],"frame":key[1],
                        "joint_tcp_error_mm":joint_err,"cart_tcp_error_mm":cart_err,
                        "cart_gt_delta_mm":(cg*1000).tolist()})
    def stats(field):
        v=np.array([r[field] for r in results])
        return {"mean_mm":float(v.mean()),"median_mm":float(np.median(v)),
                "rmse_mm":float(np.sqrt(np.mean(v**2))),"p95_mm":float(np.percentile(v,95)),
                "max_mm":float(v.max())}
    report={
        "status":"OFFLINE_ONLY_UNCALIBRATED_MAPPING",
        "mapping":"q_urdf_deg = q_lerobot_deg (5 arm joints); UNVERIFIED",
        "frame_count":len(results),
        "evaluation":"fresh ACT chunk action 0, teacher-forced held-out frames",
        "joint6":stats("joint_tcp_error_mm"),
        "cartesian7":stats("cart_tcp_error_mm"),
        "warning":"Joint6 is FK(predicted absolute joints) vs FK(ground-truth absolute joints). Cartesian7 is norm(predicted delta minus GT delta). The latter is comparable only if the recorded GT delta is the same FK-derived target under this URDF mapping. No physical calibration, IK safety or closed-loop rollout evaluated."
    }
    a.output.mkdir(parents=True)
    with (a.output/"per_frame.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["episode","frame","joint_tcp_error_mm","cart_tcp_error_mm","cart_gt_delta_mm"])
        w.writeheader()
        for r in results:
            r=dict(r);r["cart_gt_delta_mm"]=json.dumps(r["cart_gt_delta_mm"])
            w.writerow(r)
    (a.output/"comparison.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
