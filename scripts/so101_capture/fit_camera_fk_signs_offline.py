#!/usr/bin/env python3
"""Offline fixed-camera 2D/3D registration for SO-101 sign hypotheses.

Manual corresponding pixels are required. Uses one shared camera pose for all
frames, fixed approximate intrinsics, no joint offsets. It is not calibration
until annotations and plausible camera parameters are verified.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from compare_recorded_pose_fk_signs_offline import read_observations, fk_points
from evaluate_mapping_hypotheses_offline import HYPOTHESES
import placo

# Endpoints as link origins. These may be behind the casing: label uncertainty explicitly.
ALLOWED=("base_link","shoulder_link","upper_arm_link","lower_arm_link","wrist_link","gripper_link","gripper_frame_link")

def project(points, params, width, height, f):
    rot=Rotation.from_rotvec(params[:3]).as_matrix()
    cam=(rot @ points.T).T+params[3:6]
    z=cam[:,2]
    uv=np.column_stack((f*cam[:,0]/np.maximum(z,1e-9)+width/2,
                        f*cam[:,1]/np.maximum(z,1e-9)+height/2))
    return uv,z

def solve(label, annotated, observations, robot, width, height, f):
    xyz=[]; pix=[]
    for row in annotated:
        pts=fk_points(robot,observations[row["frame"]]["observation.state"],HYPOTHESES[label])
        for joint,uv in row["points"].items():
            if joint not in ALLOWED:raise ValueError(f"Unknown joint {joint}")
            if len(uv)!=2 or not np.isfinite(uv).all():raise ValueError("Bad pixel")
            xyz.append(pts[joint]);pix.append(uv)
    xyz=np.asarray(xyz);pix=np.asarray(pix)
    if len(xyz)<8:raise ValueError("At least 8 annotated 2D/3D correspondences are required")
    # Same camera extrinsics shared across all frames; non-physical solutions penalized.
    def residual(v):
        uv,z=project(xyz,v,width,height,f)
        return np.r_[(uv-pix).ravel(), np.maximum(0,.05-z)*1000]
    rng=np.random.default_rng(7)
    starts=[np.r_[np.zeros(3),[0,0,.7]]]
    starts += [np.r_[Rotation.random(random_state=rng).as_rotvec(),[0,0,d]]
               for d in (.5,1.,1.5,2.) for _ in range(12)]
    results=[]
    for seed in starts:
        fit=least_squares(residual,seed,max_nfev=150)
        uv,z=project(xyz,fit.x,width,height,f)
        if np.min(z)>.05:
            rmse=float(np.sqrt(np.mean(np.sum((uv-pix)**2,axis=1))))
            results.append((rmse,fit.x,uv,z))
    if not results:return {"mapping":label,"status":"NO_POSITIVE_DEPTH_FIT"}
    rmse,pose,uv,z=min(results,key=lambda x:x[0])
    return {"mapping":label,"status":"FIT_NOT_PHYSICAL_VALIDATION",
            "reprojection_rmse_px":rmse,"n_points":len(xyz),
            "camera_rotation_rotvec_rad":pose[:3].tolist(),
            "camera_translation_m":pose[3:].tolist(),
            "min_depth_m":float(min(z)),
            "reprojected_pixels":uv.tolist()}

def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    doc=json.loads(args.annotations.read_text())
    rows=doc.get("frames",[])
    if not rows:raise ValueError("Annotations are empty; label images before solving")
    width,height=int(doc["width"]),int(doc["height"])
    if width<=0 or height<=0:raise ValueError("Image size invalid")
    focal=float(args.focal_px) if args.focal_px else max(width,height)*1.2
    ids=set()
    for row in rows:
        index=int(row["frame"])
        if index in ids:raise ValueError("Repeated frame")
        ids.add(index)
        if not row.get("points"):raise ValueError(f"No points for frame {index}")
    observations=read_observations(args.dataset,args.episode,ids)
    robot=placo.RobotWrapper(str(args.urdf.resolve()))
    results=[solve(label,rows,observations,robot,width,height,focal) for label in HYPOTHESES]
    output={"classification":"EXPLORATORY_FIXED_CAMERA_FK_REPROJECTION",
            "focal_px":focal,"image_size":[width,height],
            "results":results,
            "warnings":["Manual landmarks and link origins must correspond physically",
                        "Assumed focal length/principal point and zero joint offsets",
                        "Same camera pose constrained across frames, but fit alone is not proof",
                        "Compare against held-out frames before accepting any sign",
                        "No motors actuated, no data or URDF modified"]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+"\n")
    for x in results:print(x["mapping"],x["status"],"RMSE px",x.get("reprojection_rmse_px"))
    print("Output:",args.output)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--urdf",type=Path,required=True)
    p.add_argument("--annotations",type=Path,required=True)
    p.add_argument("--episode",type=int,default=0)
    p.add_argument("--focal-px",type=float)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
if __name__=="__main__":main()
