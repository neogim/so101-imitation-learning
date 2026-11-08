#!/usr/bin/env python3
"""Compare camera intrinsic models on stored Side-OBS checkerboard stills.

Read only. Does not access robot, cameras, OBS, or the LeRobot dataset.
A low reprojection RMS is not sufficient evidence for a physically correct lens model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

MODELS = {
    "pinhole": cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3,
    "radial_k1": cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3,
    "radial_k1k2": cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
    "full_5coeff": 0,
}


def collect(folder, columns, rows, square_mm):
    if columns < 3 or rows < 3 or square_mm <= 0:
        raise ValueError("Invalid board geometry")
    obj = np.zeros((columns*rows, 3), np.float32)
    obj[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * (square_mm / 1000)
    imgs, detected, rejected = [], [], []
    size = None
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"} or not path.is_file():
            continue
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            rejected.append({"image":path.name, "reason":"unreadable"})
            continue
        h, w = gray.shape
        if size is None:
            size = (w, h)
        if size != (w, h):
            raise ValueError(f"Image sizes differ: {path.name}")
        ok, corners = cv2.findChessboardCornersSB(gray, (columns, rows),
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            imgs.append(obj.copy())
            detected.append({"image": path.name, "corners":corners.astype(np.float32)})
        else:
            rejected.append({"image":path.name, "reason":"corners_not_detected"})
    if size is None or len(detected) < 12:
        raise ValueError(f"Need >=12 detected views; got {len(detected)}")
    return imgs, detected, rejected, size


def _fit(objects, corners, size, flags):
    err, K, D, _, _ = cv2.calibrateCamera(
        objects, corners, size, None, None, flags=flags)
    if not np.isfinite([err, *K.flatten(), *D.flatten()]).all():
        raise ValueError("Nonfinite calibration output")
    return float(err), K, D


def _held_out_error(obj, pixel, K, D):
    # Solve an independent pose for the held-out image, then measure corner fit.
    # This is NOT an independent validation of camera pose / world alignment.
    ok, rvec, tvec = cv2.solvePnP(obj, pixel, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    predicted = cv2.projectPoints(obj, rvec, tvec, K, D)[0].reshape(-1, 2)
    measured = pixel.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((predicted - measured)**2, axis=1))))


def compare(objects, detected, size, *, do_leave_one_out=True):
    corners = [x["corners"] for x in detected]
    result = {}
    for name, flags in MODELS.items():
        train_rms, K, D = _fit(objects, corners, size, flags)
        leave_one_out = []
        if do_leave_one_out:
            for i in range(len(objects)):
                oo = objects[:i] + objects[i+1:]
                pp = corners[:i] + corners[i+1:]
                try:
                    _, K_i, D_i = _fit(oo, pp, size, flags)
                    error = _held_out_error(objects[i], corners[i], K_i, D_i)
                except (ValueError, cv2.error):
                    error = None
                leave_one_out.append({"image":detected[i]["image"], "rms_px": error})
        scores = [a["rms_px"] for a in leave_one_out if a["rms_px"] is not None]
        result[name] = {
            "training_rms_px": train_rms,
            "loo_rms_mean_px": float(np.mean(scores)) if scores else None,
            "loo_rms_median_px": float(np.median(scores)) if scores else None,
            "loo_failures": len(leave_one_out) - len(scores),
            "per_image_loo": leave_one_out,
            "intrinsics": {
                "width":size[0], "height":size[1],
                "fx":float(K[0,0]), "fy":float(K[1,1]),
                "cx":float(K[0,2]), "cy":float(K[1,2]),
                "distortion":D.ravel().astype(float).tolist()
            }
        }
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images",type=Path,required=True)
    p.add_argument("--columns",type=int,default=7)
    p.add_argument("--rows",type=int,default=5)
    p.add_argument("--square-mm",type=float,default=20)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--skip-loo",action="store_true",help="fast smoke test; no holdout validation")
    args=p.parse_args()
    try:
        imgs=args.images.expanduser().resolve(strict=True)
        if not imgs.is_dir():
            raise ValueError("Images path is not a directory")
        target=args.output.expanduser().resolve()
        if target.exists():
            raise ValueError(f"Output already exists: {target}")
        objects, detected, rejected, size=collect(imgs,args.columns,args.rows,args.square_mm)
        models=compare(objects,detected,size,do_leave_one_out=not args.skip_loo)
        report={
            "status":"EXPLORATORY_NOT_CALIBRATED",
            "source":str(imgs),
            "image_width_height":list(size),
            "board_inner_corners":[args.columns,args.rows],
            "board_square_mm":args.square_mm,
            "detected_images":[r["image"] for r in detected],
            "rejected_images":rejected,
            "models":models,
            "warnings":[
                "LOO estimates a board pose for each held-out frame; not a fully independent generalization test.",
                "Low RMS alone cannot resolve degenerate distortion/focal estimates.",
                "These are 640x480 screenshot crops; determine OBS and LeRobot crop/scale/rotation before mapping to 224x168 ACT images.",
                "Paper board may bend. Confirm its flatness and physical square size.",
                "Do not use these intrinsics for Side camera PnP until cross-checked against image pipeline and held-out geometry."
            ],
        }
        target.mkdir(parents=True)
        (target/"model_comparison.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        for key,m in models.items():
            (target/f"intrinsics_{key}_candidate.json").write_text(
                json.dumps(m["intrinsics"],indent=2)+"\n")
        print("Candidate intrinsics ONLY; none automatically selected for deployment.")
        for key,m in models.items():
            print(f"{key:15} train={m['training_rms_px']:.3f}px"
                  f"  LOO-mean={m['loo_rms_mean_px'] if m['loo_rms_mean_px'] is not None else 'not run'}"
                  f"  fx={m['intrinsics']['fx']:.2f} fy={m['intrinsics']['fy']:.2f}"
                  f"  distortion={m['intrinsics']['distortion']}")
        print(f"Report: {target/'model_comparison.json'}")
    except (OSError,ValueError,cv2.error) as e:
        p.error(str(e))


if __name__=="__main__":
    main()
