"""Read-only inventory of SO-101 datasets, labels, Parquet, and videos."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import av
import pyarrow.parquet as pq


NORMAL_ENDS = {"manual", "time_limit"}
IMAGE_PREFIX = "observation.images."
PRODUCTION_PREFIX = "muhi-side-wrist-production-30-"
EPISODE_FIELDS = (
    "session", "production_candidate", "episode_index", "frames", "end_reason",
    "normal_end", "interrupted", "pick", "place", "overall", "adopt",
    "eligible", "eligibility_reason", "label_source",
)


def video_frame_count(path):
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width, height = stream.width, stream.height
        average_rate = float(stream.average_rate) if stream.average_rate else None
        for _ in container.decode(video=0):
            count += 1
    return count, width, height, average_rate


def labels_for(session, labels_dir):
    offline = labels_dir / f"{session.name}.json"
    if offline.is_file():
        return json.loads(offline.read_text()), "offline_labels"
    outcomes = session / "episode_outcomes.json"
    if outcomes.is_file():
        return json.loads(outcomes.read_text()), "episode_outcomes"
    return [], "missing"


def classify_scope(name):
    if name.startswith(PRODUCTION_PREFIX):
        return "production"
    if "test" in name or name.startswith(("j6-", "joint-slew")):
        return "test"
    return "historical"


def inspect_session(session, labels_dir):
    info_path = session / "meta/info.json"
    if not info_path.is_file():
        return None, [], {"session": session.name, "reason": "missing_meta_info"}
    info = json.loads(info_path.read_text())
    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    fps = info.get("fps")
    features = info.get("features", {})
    image_keys = sorted(k for k in features if k.startswith(IMAGE_PREFIX))
    image_shapes = {k: features[k].get("shape") for k in image_keys}
    data_files = sorted((session / "data").rglob("*.parquet"))
    parquet_counts = Counter()
    parquet_rows = 0
    parquet_error = ""
    try:
        for path in data_files:
            table = pq.read_table(path, columns=["episode_index"])
            values = table.column("episode_index").to_pylist()
            parquet_rows += len(values)
            parquet_counts.update(int(v) for v in values)
    except Exception as exc:
        parquet_error = f"{type(exc).__name__}: {exc}"

    metadata_lengths = {}
    metadata_error = ""
    try:
        for path in sorted((session / "meta/episodes").rglob("*.parquet")):
            for row in pq.read_table(path, columns=["episode_index", "length"]).to_pylist():
                metadata_lengths[int(row["episode_index"])] = int(row["length"])
    except Exception as exc:
        metadata_error = f"{type(exc).__name__}: {exc}"

    labels, label_source = labels_for(session, labels_dir)
    labels_by_episode = {int(row["episode_index"]): row for row in labels}
    episode_rows = []
    for episode_index in range(total_episodes):
        label = labels_by_episode.get(episode_index, {})
        frames = parquet_counts.get(episode_index, 0)
        end_reason = label.get("end_reason", "unknown")
        pick = label.get("pick", "unknown")
        place = label.get("place", "unknown")
        overall = label.get("overall", "unknown")
        adopt = label.get("adopt") is True
        normal = end_reason in NORMAL_ENDS
        frame_ok = frames == metadata_lengths.get(episode_index) == label.get("frames", frames)
        eligible = normal and pick == place == overall == "success" and adopt and frame_ok
        reasons = []
        if not normal:
            reasons.append(f"end={end_reason}")
        if (pick, place, overall) != ("success", "success", "success"):
            reasons.append("outcome_not_all_success")
        if not adopt:
            reasons.append("adopt_false")
        if not frame_ok:
            reasons.append("episode_frame_mismatch")
        episode_rows.append({
            "session": session.name,
            "production_candidate": session.name.startswith(PRODUCTION_PREFIX),
            "episode_index": episode_index,
            "frames": frames,
            "end_reason": end_reason,
            "normal_end": normal,
            "interrupted": end_reason == "interrupted",
            "pick": pick,
            "place": place,
            "overall": overall,
            "adopt": adopt,
            "eligible": eligible,
            "eligibility_reason": "eligible" if eligible else ";".join(reasons),
            "label_source": label_source,
        })

    videos = {}
    video_errors = []
    for key in image_keys:
        paths = sorted((session / "videos" / key).rglob("*.mp4"))
        decoded = 0
        properties = []
        try:
            for path in paths:
                count, width, height, rate = video_frame_count(path)
                decoded += count
                properties.append({"path": str(path), "frames": count, "width": width,
                                   "height": height, "fps": rate})
        except Exception as exc:
            video_errors.append(f"{key}:{type(exc).__name__}:{exc}")
        videos[key] = {"files": len(paths), "frames": decoded, "details": properties,
                       "matches_parquet": bool(paths) and decoded == parquet_rows}

    side_ok = videos.get("observation.images.side", {}).get("matches_parquet", False)
    wrist_ok = videos.get("observation.images.wrist", {}).get("matches_parquet", False)
    frame_consistent = (
        not parquet_error and not metadata_error and parquet_rows == total_frames
        and sum(metadata_lengths.values()) == total_frames
        and all(item["matches_parquet"] for item in videos.values())
    )
    scope = classify_scope(session.name)
    session_row = {
        "session": session.name,
        "scope": scope,
        "production_candidate": scope == "production",
        "dataset_version": info.get("codebase_version"),
        "fps": fps,
        "total_episodes": total_episodes,
        "normal_ends": sum(row["normal_end"] for row in episode_rows),
        "interrupted": sum(row["interrupted"] for row in episode_rows),
        "unknown_end": sum(row["end_reason"] == "unknown" for row in episode_rows),
        "pick_success": sum(row["pick"] == "success" for row in episode_rows),
        "place_success": sum(row["place"] == "success" for row in episode_rows),
        "overall_success": sum(row["overall"] == "success" for row in episode_rows),
        "adopt_true": sum(row["adopt"] for row in episode_rows),
        "eligible": sum(row["eligible"] for row in episode_rows),
        "eligible_frames": sum(row["frames"] for row in episode_rows if row["eligible"]),
        "total_frames": total_frames,
        "parquet_frames": parquet_rows,
        "image_keys": ",".join(image_keys),
        "image_shapes": json.dumps(image_shapes, sort_keys=True),
        "side_video_ok": side_ok,
        "wrist_video_ok": wrist_ok,
        "frame_consistent": frame_consistent,
        "label_source": label_source,
        "errors": ";".join(filter(None, [parquet_error, metadata_error, *video_errors])),
    }
    detail = {"session": session.name, "videos": videos, "features": features}
    return session_row, episode_rows, detail


def aggregate(rows):
    return {
        "sessions": len(rows),
        "episodes": sum(r["total_episodes"] for r in rows),
        "normal_ends": sum(r["normal_ends"] for r in rows),
        "interrupted": sum(r["interrupted"] for r in rows),
        "unknown_end": sum(r["unknown_end"] for r in rows),
        "pick_success": sum(r["pick_success"] for r in rows),
        "place_success": sum(r["place_success"] for r in rows),
        "overall_success": sum(r["overall_success"] for r in rows),
        "adopt_true": sum(r["adopt_true"] for r in rows),
        "eligible": sum(r["eligible"] for r in rows),
        "eligible_frames": sum(r["eligible_frames"] for r in rows),
    }


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "workspace/so101-data")
    parser.add_argument("--labels-dir", type=Path,
                        default=Path(__file__).resolve().parent / "labels")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    sessions, episodes, details, skipped = [], [], [], []
    for session in sorted(path for path in args.root.iterdir() if path.is_dir()):
        row, episode_rows, extra = inspect_session(session, args.labels_dir)
        if row is None:
            skipped.append(extra)
            continue
        sessions.append(row)
        episodes.extend(episode_rows)
        details.append(extra)

    write_csv(args.output / "sessions.csv", sessions, sessions[0].keys() if sessions else [])
    write_csv(args.output / "episodes.csv", episodes, EPISODE_FIELDS)
    write_csv(args.output / "skipped.csv", skipped, ("session", "reason"))
    production = [row for row in sessions if row["production_candidate"]]
    summary = {
        "root": str(args.root),
        "all_datasets": aggregate(sessions),
        "production_candidates": aggregate(production),
        "production_policy": f"session name starts with {PRODUCTION_PREFIX}",
        "eligible_policy": "normal end + pick/place/overall success + adopt=true + frame counts agree",
        "skipped_directories": len(skipped),
        "episodes": episodes,
        "session_details": details,
    }
    summary_rows = []
    for scope, values in (("all_datasets", summary["all_datasets"]),
                          ("production_candidates", summary["production_candidates"])):
        summary_rows.append({"scope": scope, **values})
    write_csv(args.output / "summary.csv", summary_rows, summary_rows[0].keys())
    (args.output / "inventory.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["production_candidates"], sort_keys=True))


if __name__ == "__main__":
    main()
