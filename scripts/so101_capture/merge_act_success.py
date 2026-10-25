"""Merge explicitly successful SO-101 episodes into a new local ACT dataset.

Reads raw sessions only. The output contains state, sent action, and RGB video;
the Leader's action_requested is intentionally excluded.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from offline_labels import validate_labels

NORMAL = {"manual", "time_limit"}
KEEP = ("observation.state", "action", "observation.images.iphone")


def plan(source_root: Path, labels_dir: Path):
    selected = []
    reference_features = None
    for source in sorted(source_root.iterdir()):
        info_file = source / "meta/info.json"
        if not info_file.is_file():
            continue
        info = json.loads(info_file.read_text())
        if not info.get("total_episodes"):
            continue
        label_file = labels_dir / f"{source.name}.json"
        if label_file.exists():
            rows = validate_labels(json.loads(label_file.read_text()), source)
        else:
            outcomes = source / "episode_outcomes.json"
            rows = json.loads(outcomes.read_text()) if outcomes.exists() else []
        eligible = [row for row in rows if row.get("end_reason") in NORMAL
                    and all(row.get(key) == "success" for key in ("pick", "place", "overall"))
                    and row.get("adopt") is True]
        if not eligible:
            continue
        if info["fps"] != 20 or info["codebase_version"] != "v3.0":
            raise ValueError(f"Incompatible source: {source}")
        features = {key: info["features"][key] for key in KEEP}
        if reference_features is None:
            reference_features = features
        elif features != reference_features:
            raise ValueError(f"Feature mismatch: {source}")
        episodes = []
        for file in sorted((source / "meta/episodes").rglob("*.parquet")):
            episodes.extend(pq.read_table(file).to_pylist())
        lengths = {int(row["episode_index"]): int(row["length"]) for row in episodes}
        for row in eligible:
            ep = int(row["episode_index"])
            if ep not in lengths or row.get("frames") != lengths[ep]:
                raise ValueError(f"Episode length mismatch: {source} ep={ep}")
            selected.append((source, ep, lengths[ep], label_file if label_file.exists() else None))
    return selected, reference_features


def merge(source_root: Path, labels_dir: Path, output_root: Path, expected: int):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    selected, features = plan(source_root, labels_dir)
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} eligible episodes, found {len(selected)}")
    if output_root.exists():
        raise FileExistsError(output_root)
    if not output_root.parent.is_dir():
        raise FileNotFoundError(output_root.parent)
    dataset = LeRobotDataset.create(f"local/muhi_act_success_{expected:03d}", fps=20,
        root=output_root, robot_type="so101_follower", features=features,
        use_videos=True, video_backend="pyav", rgb_encoder=RGBEncoderConfig(vcodec="h264"),
        image_writer_threads=2)
    lineage = []
    task = "Pick up the Muhi bottle and place it into the target container."
    try:
        for source in dict.fromkeys(item[0] for item in selected):
            desired = {ep: (length, label) for path, ep, length, label in selected if path == source}
            data_files = sorted((source / "data").rglob("*.parquet"))
            video_files = sorted((source / "videos/observation.images.iphone").rglob("*.mp4"))
            if len(data_files) != 1 or len(video_files) != 1:
                raise ValueError(f"Expected one Parquet and MP4: {source}")
            columns = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
            table = pq.read_table(data_files[0], columns=columns)
            rows = table.to_pylist()
            actual = {ep: 0 for ep in desired}
            with av.open(str(video_files[0])) as container:
                stream = container.streams.video[0]
                if (stream.width, stream.height) != (640, 480):
                    raise ValueError(f"Video size mismatch: {source}")
                count = 0
                for row, video_frame in zip(rows, container.decode(video=0), strict=True):
                    count += 1
                    ep = int(row["episode_index"])
                    if ep not in desired:
                        continue
                    index = actual[ep]
                    if int(row["frame_index"]) != index or abs(float(row["timestamp"]) - index / 20) > 0.002:
                        raise ValueError(f"Frame/timestamp mismatch: {source} ep={ep} frame={index}")
                    frame = {"observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                             "action": np.asarray(row["action"], dtype=np.float32),
                             "observation.images.iphone": video_frame.to_ndarray(format="rgb24"),
                             "task": task}
                    if frame["observation.state"].shape != (6,) or frame["action"].shape != (6,):
                        raise ValueError(f"Joint shape mismatch: {source}")
                    if not np.isfinite(frame["observation.state"]).all() or not np.isfinite(frame["action"]).all():
                        raise ValueError(f"Nonfinite joints: {source}")
                    dataset.add_frame(frame)
                    actual[ep] += 1
                    if actual[ep] == desired[ep][0]:
                        dataset.save_episode(parallel_encoding=False)
                        lineage.append({"episode_index": len(lineage), "source": str(source),
                                        "source_episode_index": ep, "frames": actual[ep],
                                        "labels": str(desired[ep][1]) if desired[ep][1] else None})
                        print(f"Saved {len(lineage)}/{expected}: {source.name} ep={ep} frames={actual[ep]}", flush=True)
                if count != len(rows) or count != json.loads((source / "meta/info.json").read_text())["total_frames"]:
                    raise ValueError(f"Video/Parquet frame count mismatch: {source}")
            if any(actual[ep] != desired[ep][0] for ep in desired):
                raise ValueError(f"Selected episode incomplete: {source}")
    finally:
        dataset.finalize()
    (output_root / "source_lineage.json").write_text(json.dumps(lineage, indent=2) + "\n")
    return lineage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.home()/"workspace/so101-data")
    parser.add_argument("--labels-dir", type=Path, default=Path(__file__).parent/"labels")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=30)
    args = parser.parse_args()
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"
    print(f"Merged {len(merge(args.source_root.expanduser().resolve(), args.labels_dir.resolve(), args.output.resolve(), args.expected))} episodes")
