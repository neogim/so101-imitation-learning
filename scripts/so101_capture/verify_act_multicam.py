"""Verify a merged side+wrist LeRobot v3 Dataset and mark it verified."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile

import av
import numpy as np
import pyarrow.parquet as pq

from merge_act_multicam import PRODUCTION_PREFIX, fingerprints


FPS = 20
IMAGE_KEYS = ("observation.images.side", "observation.images.wrist")


def verify(root: Path, expected_episodes: int, expected_frames: int,
           height: int = 240, width: int = 320) -> dict:
    if (root / "MERGE_INCOMPLETE").exists():
        raise ValueError("Dataset is marked incomplete")
    info = json.loads((root / "meta/info.json").read_text())
    if (info.get("codebase_version") != "v3.0" or info.get("fps") != FPS
            or info.get("total_episodes") != expected_episodes
            or info.get("total_frames") != expected_frames):
        raise ValueError("Dataset version/FPS/totals mismatch")
    features = info["features"]
    if set(IMAGE_KEYS) - features.keys() or "action_requested" in features:
        raise ValueError("Camera keys missing or action_requested leaked into ACT Dataset")
    if any(features[key]["shape"] != [height, width, 3] for key in IMAGE_KEYS):
        raise ValueError("Camera shape mismatch")
    if any(features[key]["shape"] != [6] for key in ("observation.state", "action")):
        raise ValueError("State/Action shape mismatch")
    tables = [pq.read_table(path) for path in sorted((root / "data").rglob("*.parquet"))]
    rows = [row for table in tables for row in table.to_pylist()]
    if len(rows) != expected_frames:
        raise ValueError("Parquet total mismatch")
    counts = Counter()
    for global_index, row in enumerate(rows):
        episode = int(row["episode_index"])
        frame = counts[episode]
        if (int(row["index"]) != global_index or int(row["frame_index"]) != frame
                or abs(float(row["timestamp"]) - frame / FPS) > 0.002
                or int(row["task_index"]) != 0):
            raise ValueError(f"Index/timestamp/task mismatch at {global_index}")
        for key in ("observation.state", "action"):
            values = np.asarray(row[key])
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {key} at {global_index}")
        counts[episode] += 1
    if set(counts) != set(range(expected_episodes)):
        raise ValueError("Episode indices are not contiguous")
    episode_rows = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        episode_rows.extend(pq.read_table(path).to_pylist())
    if len(episode_rows) != expected_episodes:
        raise ValueError("Episode metadata count mismatch")
    for row in episode_rows:
        if counts[int(row["episode_index"])] != int(row["length"]):
            raise ValueError("Episode metadata length mismatch")

    decoded = {}
    per_episode_video_match = {}
    for key in IMAGE_KEYS:
        total = 0
        video_counts = Counter()
        for path in sorted((root / "videos" / key).rglob("*.mp4")):
            file_index = int(path.stem.split("-")[-1])
            chunk_index = int(path.parent.name.split("-")[-1])
            local = [row for row in episode_rows
                     if int(row[f"videos/{key}/file_index"]) == file_index
                     and int(row[f"videos/{key}/chunk_index"]) == chunk_index]
            local.sort(key=lambda row: float(row[f"videos/{key}/from_timestamp"]))
            if not local:
                raise ValueError(f"{key} video has no episode metadata: {path}")
            starts = np.asarray([float(row[f"videos/{key}/from_timestamp"]) for row in local])
            stops = np.asarray([float(row[f"videos/{key}/to_timestamp"]) for row in local])
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                if (stream.height, stream.width) != (height, width):
                    raise ValueError(f"{key} video shape mismatch")
                for frame in container.decode(video=0):
                    timestamp = float(frame.pts * frame.time_base)
                    position = int(np.searchsorted(starts, timestamp + 1e-5, side="right") - 1)
                    if (position < 0 or position >= len(local)
                            or not starts[position] - 1e-5 <= timestamp < stops[position] - 1e-5):
                        raise ValueError(f"{key} frame outside episode interval: {path}")
                    video_counts[int(local[position]["episode_index"])] += 1
                    total += 1
        if total != expected_frames:
            raise ValueError(f"{key} video frame count mismatch: {total}")
        if any(video_counts[episode] != counts[episode] for episode in range(expected_episodes)):
            raise ValueError(f"{key} per-episode video/Parquet mismatch")
        decoded[key] = total
        per_episode_video_match[key] = True

    merge_manifest = root / "merge_manifest.json"
    source_fingerprints_unchanged = None
    train = validation = set()
    if merge_manifest.is_file():
        manifest = json.loads(merge_manifest.read_text())
        if len(manifest["lineage"]) != expected_episodes:
            raise ValueError("Lineage count mismatch")
        saved_fingerprints = json.loads((root / "source_fingerprints.json").read_text())
        sessions = sorted({parent for name in saved_fingerprints for parent in Path(name).parents
                           if parent.name.startswith(PRODUCTION_PREFIX)})
        if fingerprints(sessions) != saved_fingerprints:
            raise ValueError("Source Dataset fingerprint changed after merge")
        source_fingerprints_unchanged = True
        split = json.loads((root / "episode_split.json").read_text())
        train, validation = set(split["train_episode_indices"]), set(split["validation_episode_indices"])
        if train & validation or train | validation != set(range(expected_episodes)):
            raise ValueError("Episode split overlaps or is incomplete")
    else:
        lineage = json.loads((root / "source_lineage.json").read_text())
        if (lineage.get("episodes") != expected_episodes or lineage.get("frames") != expected_frames
                or set(lineage.get("image_keys", [])) != set(IMAGE_KEYS)):
            raise ValueError("Derived resize lineage mismatch")

    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "so101-act-merge-hf"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset("local/muhi_side_wrist_verify", root=root, video_backend="pyav")
    if len(dataset) != expected_frames or dataset.num_episodes != expected_episodes:
        raise ValueError("LeRobotDataset API totals mismatch")
    samples = {}
    for index in (0, expected_frames // 2, expected_frames - 1):
        item = dataset[index]
        samples[index] = {}
        for key in IMAGE_KEYS:
            shape = list(item[key].shape)
            if shape != [3, height, width]:
                raise ValueError(f"LeRobotDataset image readback mismatch: {key} {shape}")
            samples[index][key] = shape
    return {
        "episodes": expected_episodes, "frames": expected_frames, "fps": FPS,
        "image_keys": list(IMAGE_KEYS), "video_frames": decoded,
        "per_episode_video_match": per_episode_video_match,
        "source_fingerprints_unchanged": source_fingerprints_unchanged,
        "state_shape": [6], "action_shape": [6], "samples": samples,
        "train_episodes": len(train) if train else None,
        "validation_episodes": len(validation) if validation else None,
        "lerobot_dataset_readback": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    result = verify(args.root.resolve(), args.expected_episodes, args.expected_frames,
                    args.height, args.width)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    complete = args.root / "MERGE_COMPLETE.json"
    if complete.is_file():
        state = json.loads(complete.read_text())
        state["verified"] = True
        state["verification_report"] = str(args.report.resolve())
        complete.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
