#!/usr/bin/env python3
"""Read-only SO-101 ArUco video analysis. Never imports or operates robot hardware.

Marker IDs 0,1,2 should be on consecutive rigid links surrounding J2/J3.
A two-dimensional image can show approximate changes; not calibrated 3D angles.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def angle_deg(points):
    import numpy as np
    p = np.asarray(points, dtype=float).reshape(4, 2)
    v = p[1] - p[0]
    return math.degrees(math.atan2(float(v[1]), float(v[0])))


def signed_delta(a, b):
    return (b - a + 180.0) % 360.0 - 180.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True, help="Existing video file; no live capture")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/so101-aruco-video-check"))
    ap.add_argument("--sample-every", type=int, default=1, help="Analyze every Nth frame")
    ap.add_argument("--marker-ids", type=int, nargs=3, default=[0, 1, 2], metavar=("BASE", "UPPER", "FOREARM"))
    ap.add_argument("--annotated-video", action="store_true", help="Write diagnostic MP4; extra CPU/disk cost")
    args = ap.parse_args()
    if args.sample_every < 1 or len(set(args.marker_ids)) != 3:
        ap.error("--sample-every must be >=1 and IDs must be distinct")

    import cv2
    import numpy as np
    if not hasattr(cv2, "aruco"):
        ap.error("OpenCV ArUco unavailable (install opencv-contrib-python or a compatible build)")
    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        ap.error(f"Cannot open: {args.video}")

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(d, params) if hasattr(cv2.aruco, "ArucoDetector") else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, total, sampled, all_three = [], 0, 0, 0
    counts = {i: 0 for i in args.marker_ids}
    fps = float(video.get(cv2.CAP_PROP_FPS))
    writer = None
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            frame_no = total
            total += 1
            if frame_no % args.sample_every:
                continue
            sampled += 1
            corners, ids, _ = (detector.detectMarkers(frame) if detector is not None
                               else cv2.aruco.detectMarkers(frame, d, parameters=params))
            seen = {}
            if ids is not None:
                for pts, id_row in zip(corners, ids):
                    marker_id = int(id_row[0])
                    if marker_id not in args.marker_ids:
                        continue
                    xy = pts.reshape(4, 2)
                    lengths = [float(np.linalg.norm(xy[(j + 1) % 4] - xy[j])) for j in range(4)]
                    seen[marker_id] = {"angle": angle_deg(xy), "min_side": min(lengths)}
                    counts[marker_id] += 1
            base, upper, forearm = args.marker_ids
            if len(seen) == 3:
                all_three += 1
            row = {"frame": frame_no, "seconds": round(frame_no / fps, 4) if fps > 0 else "",
                   "all_three": int(len(seen) == 3)}
            for marker_id in args.marker_ids:
                row[f"id{marker_id}_seen"] = int(marker_id in seen)
                row[f"id{marker_id}_edge_deg"] = round(seen[marker_id]["angle"], 3) if marker_id in seen else ""
                row[f"id{marker_id}_min_side_px"] = round(seen[marker_id]["min_side"], 2) if marker_id in seen else ""
            row["j2_image_relative_deg"] = round(signed_delta(seen[base]["angle"], seen[upper]["angle"]), 3) if base in seen and upper in seen else ""
            row["j3_image_relative_deg"] = round(signed_delta(seen[upper]["angle"], seen[forearm]["angle"]), 3) if upper in seen and forearm in seen else ""
            rows.append(row)
            if args.annotated_video:
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(args.out_dir / "annotated.mp4"), fourcc, fps / args.sample_every if fps > 0 else 10, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("Cannot create annotated MP4")
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.putText(frame, f"frame {frame_no} visible {sorted(seen)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                writer.write(frame)
    finally:
        video.release()
        if writer is not None:
            writer.release()

    if not rows:
        raise RuntimeError("No readable frames")
    csv_path = args.out_dir / "markers_by_frame.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = {
        "video": str(args.video), "dictionary": "DICT_4X4_50",
        "marker_ids_base_upper_forearm": args.marker_ids,
        "fps_reported": fps, "frames_total": total,
        "frames_sampled": sampled,
        "detection_rate": {str(i): round(counts[i] / sampled, 5) for i in args.marker_ids},
        "all_three_rate": round(all_three / sampled, 5),
        "output_csv": str(csv_path),
        "cautions": [
            "Image relative angles are not joint angles; perspective, out-of-plane motion and camera pose distort them.",
            "To validate physical joint signs/magnitudes, collect synchronized follower joint telemetry and calibrated 3D poses.",
            "This script never accesses cameras, servos, ROS or ACT checkpoints."
        ],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
