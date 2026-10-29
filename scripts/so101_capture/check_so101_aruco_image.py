#!/usr/bin/env python3
"""Offline ArUco detection from one image. Never opens camera or servo."""
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/so101-aruco-still-check"))
    parser.add_argument("--expected-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dictionary", choices=["DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_6X6_250"], default="DICT_4X4_50")
    args = parser.parse_args()

    import cv2
    import numpy as np

    if not hasattr(cv2, "aruco"):
        parser.error("cv2.aruco unavailable: use OpenCV with ArUco support")
    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img is None:
        parser.error(f"Unable to read image: {args.image}")

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, rejected = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(img)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(img, dictionary, parameters=params)

    found = []
    annotated = img.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        for pts, id_row in zip(corners, ids):
            pts = pts.reshape(4, 2)
            sides = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
            found.append({
                "id": int(id_row[0]),
                "corners_px": pts.round(2).tolist(),
                "side_lengths_px": [round(v, 2) for v in sides],
                "min_side_px": round(min(sides), 2)
            })

    detected = sorted({m["id"] for m in found})
    result = {
        "source": str(args.image),
        "image_width_px": int(img.shape[1]),
        "image_height_px": int(img.shape[0]),
        "dictionary": args.dictionary,
        "detected_ids": detected,
        "expected_ids": args.expected_ids,
        "missing_expected_ids": sorted(set(args.expected_ids) - set(detected)),
        "rejected_candidates": len(rejected),
        "markers": found,
        "note": "Detection only: no 3D pose, link verification, J2/J3 zero offsets or servo commands."
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = args.out_dir / "annotated.png"
    report_path = args.out_dir / "report.json"
    if not cv2.imwrite(str(annotated_path), annotated):
        raise RuntimeError(f"Could not write {annotated_path}")
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Annotated: {annotated_path}\nReport: {report_path}")
    return 0 if not result["missing_expected_ids"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
