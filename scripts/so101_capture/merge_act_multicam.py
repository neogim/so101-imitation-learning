"""Merge explicitly adopted two-camera SO-101 episodes into a new v3 Dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import av
import numpy as np
import pyarrow.parquet as pq


FPS = 20
TASK = "Pick up the Muhi bottle and place it into the target container."
NORMAL_ENDS = {"manual", "time_limit"}
PRODUCTION_PREFIX = "muhi-side-wrist-production-30-"
IMAGE_KEYS = ("observation.images.side", "observation.images.wrist")
KEEP = ("observation.state", "action", *IMAGE_KEYS)
JOINTS = [f"{name}.pos" for name in (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)]
CONDITION_KEYS = (
    "fps", "seconds", "slew", "rotary_slew", "gripper_slew",
    "j6_max_closing_lead", "image_shape", "cameras", "use_degrees",
    "gripper_unit", "action", "action_requested", "task",
)


class MergeError(RuntimeError):
    pass


def json_load(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(session: Path) -> list[Path]:
    paths = []
    for relative in ("meta/info.json", "episode_outcomes.json", "capture_config.json"):
        path = session / relative
        if path.is_file():
            paths.append(path)
    for relative in ("data", "meta/episodes", *(f"videos/{key}" for key in IMAGE_KEYS)):
        root = session / relative
        if root.is_dir():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(set(paths))


def fingerprints(sessions: list[Path]) -> dict[str, dict[str, object]]:
    result = {}
    for session in sessions:
        for path in source_files(session):
            result[str(path)] = {"size": path.stat().st_size, "sha256": sha256(path)}
    return result


def video_frames(paths: list[Path], expected_shape: tuple[int, int, int]):
    for path in paths:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if (stream.height, stream.width, 3) != expected_shape:
                raise MergeError(f"Video shape mismatch: {path}")
            if stream.average_rate and abs(float(stream.average_rate) - FPS) > 0.01:
                raise MergeError(f"Video FPS mismatch: {path}")
            yield from container.decode(video=0)


def determine_sessions(root: Path, explicit: list[Path] | None) -> list[Path]:
    sessions = explicit or sorted(root.glob(f"{PRODUCTION_PREFIX}*"))
    resolved = [path.resolve() for path in sessions]
    if not resolved or len(resolved) != len(set(resolved)):
        raise MergeError("No production sessions or duplicate session arguments")
    for path in resolved:
        if path.parent != root.resolve() or not path.name.startswith(PRODUCTION_PREFIX):
            raise MergeError(f"Not an allowed production session: {path}")
        if not (path / "meta/info.json").is_file():
            raise MergeError(f"Missing Dataset metadata: {path}")
    return sorted(resolved)


def validate_features(info: dict, session: Path) -> dict:
    if info.get("codebase_version") != "v3.0" or info.get("fps") != FPS:
        raise MergeError(f"Dataset version/FPS mismatch: {session}")
    features = info.get("features", {})
    if set(KEEP) - features.keys():
        raise MergeError(f"Required features missing: {session}")
    for key in ("observation.state", "action"):
        if features[key].get("shape") != [6] or features[key].get("names") != JOINTS:
            raise MergeError(f"Joint schema mismatch for {key}: {session}")
    for key in IMAGE_KEYS:
        if features[key].get("shape") != [240, 320, 3]:
            raise MergeError(f"Camera schema mismatch for {key}: {session}")
    # LeRobot adds codec ``info`` only after the first video is encoded.  Empty
    # sessions therefore lack it; the actual feature contract is dtype/shape/names.
    return {key: {field: features[key].get(field) for field in ("dtype", "shape", "names")}
            for key in KEEP}


def episode_lengths(session: Path) -> dict[int, int]:
    result = {}
    for path in sorted((session / "meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(path, columns=["episode_index", "length"]).to_pylist():
            episode = int(row["episode_index"])
            if episode in result:
                raise MergeError(f"Duplicate episode metadata: {session} episode={episode}")
            result[episode] = int(row["length"])
    return result


def label_selection(session: Path, count: int, lengths: dict[int, int]) -> tuple[list[int], list[dict]]:
    path = session / "episode_outcomes.json"
    if not path.is_file():
        raise MergeError(f"Missing episode outcomes: {session}")
    labels = json_load(path)
    by_episode = {int(row["episode_index"]): row for row in labels}
    if len(by_episode) != len(labels) or set(by_episode) != set(range(count)):
        raise MergeError(f"Outcome coverage/uniqueness mismatch: {session}")
    selected, decisions = [], []
    for episode in range(count):
        row = by_episode[episode]
        eligible = (
            row.get("end_reason") in NORMAL_ENDS
            and all(row.get(key) == "success" for key in ("pick", "place", "overall"))
            and row.get("adopt") is True
            and int(row.get("frames", -1)) == lengths.get(episode)
        )
        decisions.append({
            "source": str(session), "source_episode_index": episode,
            "frames": lengths.get(episode), "eligible": eligible,
            "end_reason": row.get("end_reason", "unknown"),
            "pick": row.get("pick", "unknown"), "place": row.get("place", "unknown"),
            "overall": row.get("overall", "unknown"), "adopt": row.get("adopt") is True,
        })
        if eligible:
            selected.append(episode)
    return selected, decisions


def validate_session(session: Path) -> dict:
    info = json_load(session / "meta/info.json")
    features = validate_features(info, session)
    count, total = int(info.get("total_episodes", -1)), int(info.get("total_frames", -1))
    lengths = episode_lengths(session)
    if set(lengths) != set(range(count)) or sum(lengths.values()) != total:
        raise MergeError(f"Episode metadata totals mismatch: {session}")
    selected, decisions = label_selection(session, count, lengths)
    data_files = sorted((session / "data").rglob("*.parquet"))
    videos = {key: sorted((session / "videos" / key).rglob("*.mp4")) for key in IMAGE_KEYS}
    if not data_files or any(not paths for paths in videos.values()):
        if count == 0 and total == 0:
            return {"session": session, "features": features, "rows": [], "videos": videos,
                    "selected": selected, "decisions": decisions, "lengths": lengths,
                    "capture": {}, "total_frames": 0}
        raise MergeError(f"Missing Parquet or camera video: {session}")
    rows = []
    for path in data_files:
        rows.extend(pq.read_table(path, columns=[
            "episode_index", "frame_index", "timestamp", "observation.state", "action",
        ]).to_pylist())
    if len(rows) != total:
        raise MergeError(f"Parquet total mismatch: {session}")
    counts = Counter()
    streams = [video_frames(videos[key], tuple(features[key]["shape"])) for key in IMAGE_KEYS]
    decoded = 0
    try:
        for items in zip(rows, *streams, strict=True):
            row, side, wrist = items
            episode = int(row["episode_index"])
            frame = counts[episode]
            if episode not in lengths or int(row["frame_index"]) != frame:
                raise MergeError(f"Frame ordering mismatch: {session} episode={episode} frame={frame}")
            if abs(float(row["timestamp"]) - frame / FPS) > 0.002:
                raise MergeError(f"Timestamp mismatch: {session} episode={episode} frame={frame}")
            if side.width != 320 or wrist.width != 320:
                raise MergeError(f"Decoded camera mismatch: {session}")
            for key in ("observation.state", "action"):
                values = np.asarray(row[key], dtype=np.float32)
                if values.shape != (6,) or not np.isfinite(values).all():
                    raise MergeError(f"Invalid {key}: {session} episode={episode} frame={frame}")
            counts[episode] += 1
            decoded += 1
    except ValueError as exc:
        raise MergeError(f"Parquet/video frame mismatch: {session}: {exc}") from exc
    if decoded != total or any(counts[episode] != length for episode, length in lengths.items()):
        raise MergeError(f"Per-episode frame mismatch: {session}")
    capture_path = session / "capture_config.json"
    capture = json_load(capture_path) if capture_path.is_file() else {}
    capture = {key: capture.get(key) for key in CONDITION_KEYS if key in capture}
    return {"session": session, "features": features, "rows": rows, "videos": videos,
            "selected": selected, "decisions": decisions, "lengths": lengths,
            "capture": capture, "total_frames": total}


def make_plan(root: Path, sessions_arg: list[Path] | None, expected_episodes: int | None,
              expected_frames: int | None) -> dict:
    sessions = determine_sessions(root, sessions_arg)
    sources = [validate_session(session) for session in sessions]
    selected = [
        {"source": str(source["session"]), "source_episode_index": episode,
         "frames": source["lengths"][episode]}
        for source in sources for episode in source["selected"]
    ]
    excluded = [decision for source in sources for decision in source["decisions"]
                if not decision["eligible"]]
    frames = sum(item["frames"] for item in selected)
    if expected_episodes is not None and len(selected) != expected_episodes:
        raise MergeError(f"Expected {expected_episodes} selected episodes, found {len(selected)}")
    if expected_frames is not None and frames != expected_frames:
        raise MergeError(f"Expected {expected_frames} selected frames, found {frames}")
    reference = next((source["features"] for source in sources if source["rows"]), None)
    if reference is None or any(source["features"] != reference for source in sources):
        raise MergeError("Source feature schemas differ or all sources are empty")
    return {"sessions": sessions, "sources": sources, "selected": selected,
            "excluded": excluded, "selected_frames": frames, "features": reference}


def make_split(lineage: list[dict], seed: int, validation_fraction: float) -> dict:
    indices = list(range(len(lineage)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, round(len(indices) * validation_fraction))
    validation = sorted(indices[:validation_count])
    validation_set = set(validation)
    return {
        "seed": seed, "validation_fraction": validation_fraction,
        "train_episode_indices": [index for index in range(len(lineage)) if index not in validation_set],
        "validation_episode_indices": validation,
        "rule": "deterministic episode-level shuffle; no frame crosses splits",
    }


def merge(plan: dict, output: Path, repo_id: str, split_seed: int,
          validation_fraction: float, before: dict) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "so101-act-merge-hf"))
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id, fps=FPS, root=output, robot_type="so101_follower", features=plan["features"],
        use_videos=True, video_backend="pyav", rgb_encoder=RGBEncoderConfig(vcodec="h264"),
        image_writer_threads=2, image_writer_processes=0,
    )
    incomplete = output / "MERGE_INCOMPLETE"
    incomplete.write_text("Merge or verification has not completed.\n")
    lineage = []
    try:
        for source in plan["sources"]:
            desired = set(source["selected"])
            if not desired:
                continue
            counts = Counter()
            streams = [video_frames(source["videos"][key], tuple(plan["features"][key]["shape"]))
                       for key in IMAGE_KEYS]
            try:
                for items in zip(source["rows"], *streams, strict=True):
                    row, side, wrist = items
                    episode = int(row["episode_index"])
                    if episode not in desired:
                        continue
                    state = np.asarray(row["observation.state"], dtype=np.float32)
                    action = np.asarray(row["action"], dtype=np.float32)
                    dataset.add_frame({
                        "observation.state": state, "action": action,
                        "observation.images.side": side.to_ndarray(format="rgb24"),
                        "observation.images.wrist": wrist.to_ndarray(format="rgb24"),
                        "task": TASK,
                    })
                    counts[episode] += 1
                    if counts[episode] == source["lengths"][episode]:
                        output_episode = len(lineage)
                        dataset.save_episode(parallel_encoding=False)
                        lineage.append({
                            "episode_index": output_episode, "source": str(source["session"]),
                            "source_episode_index": episode, "frames": counts[episode],
                            "capture_conditions": source["capture"],
                        })
                        print(f"Saved {len(lineage)}/{len(plan['selected'])}: "
                              f"{source['session'].name} episode={episode}", flush=True)
            except ValueError as exc:
                raise MergeError(f"Source changed during merge: {source['session']}: {exc}") from exc
            if any(counts[episode] != source["lengths"][episode] for episode in desired):
                raise MergeError(f"Selected episode incomplete during merge: {source['session']}")
    finally:
        dataset.finalize()
    if len(lineage) != len(plan["selected"]) or sum(x["frames"] for x in lineage) != plan["selected_frames"]:
        raise MergeError("Merged output totals differ from validated plan")
    after = fingerprints(plan["sessions"])
    if after != before:
        raise MergeError("An input Dataset changed while merging")
    manifest = {
        "repo_id": repo_id, "episodes": len(lineage), "frames": plan["selected_frames"],
        "fps": FPS, "image_keys": list(IMAGE_KEYS), "action_requested": "excluded_from_ACT_dataset",
        "selection_rule": "normal end + Pick/Place/Overall success + adopt=true",
        "lineage": lineage, "excluded": plan["excluded"],
    }
    (output / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "source_fingerprints.json").write_text(json.dumps(before, indent=2) + "\n")
    (output / "episode_split.json").write_text(json.dumps(
        make_split(lineage, split_seed, validation_fraction), indent=2) + "\n")
    incomplete.unlink()
    (output / "MERGE_COMPLETE.json").write_text(json.dumps({
        "episodes": len(lineage), "frames": plan["selected_frames"], "verified": False,
    }, indent=2) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "workspace/so101-data")
    parser.add_argument("--sessions", type=Path, nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/muhi_side_wrist_success_053_20261025")
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--split-seed", type=int, default=20261102)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between 0 and 1")
    plan = make_plan(args.root.resolve(), args.sessions, args.expected_episodes, args.expected_frames)
    report = {
        "dry_run": args.dry_run, "sessions": len(plan["sessions"]),
        "source_episodes": sum(len(source["decisions"]) for source in plan["sources"]),
        "selected_episodes": len(plan["selected"]), "excluded_episodes": len(plan["excluded"]),
        "selected_frames": plan["selected_frames"], "output": str(output),
        "image_keys": list(IMAGE_KEYS), "fps": FPS,
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    before = fingerprints(plan["sessions"])
    merge(plan, output, args.repo_id, args.split_seed, args.validation_fraction, before)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
