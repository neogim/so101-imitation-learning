"""Make a new 168x224 local ACT dataset; leave the source untouched."""
import argparse
import json
import os
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

av.logging.set_level(av.logging.ERROR)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--expected", type=int, default=30)
    a = p.parse_args()
    source, output = a.source.resolve(), a.output.resolve()
    if output.exists() or source == output or source in output.parents:
        raise ValueError("Output must be a new path outside source")
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    info = json.loads((source / "meta/info.json").read_text())
    if info["total_episodes"] != a.expected or info["fps"] != 20:
        raise ValueError("Unexpected source")
    image_keys = sorted(k for k in info["features"] if k.startswith("observation.images."))
    if not image_keys:
        raise ValueError("No camera image features")
    source_shapes = {key: tuple(info["features"][key]["shape"]) for key in image_keys}
    if any(len(shape) != 3 or shape[2] != 3 for shape in source_shapes.values()):
        raise ValueError("Expected RGB video features in source")
    features = {key: info["features"][key].copy() for key in
                ("observation.state", "action", *image_keys)}
    for key in image_keys:
        features[key] = {**features[key], "shape": [168, 224, 3],
            "info": {**features[key]["info"], "video.height": 168, "video.width": 224}}
    data_files = sorted((source / "data").rglob("*.parquet"))
    video_files = {key: sorted((source / "videos" / key).rglob("*.mp4")) for key in image_keys}
    if len(data_files) != 1 or any(not files for files in video_files.values()):
        raise ValueError("Expected one source Parquet and MP4 for every camera")
    rows = pq.read_table(data_files[0], columns=["episode_index", "frame_index", "timestamp",
                                                      "observation.state", "action"]).to_pylist()
    repo_id = (f"local/muhi_act_success_{a.expected:03d}_168x224" if len(image_keys) == 1
               else f"local/muhi_act_success_{a.expected:03d}_{len(image_keys)}cam_168x224")
    output_ds = LeRobotDataset.create(repo_id, fps=20,
        root=output, robot_type="so101_follower", features=features, use_videos=True,
        video_backend="pyav", rgb_encoder=RGBEncoderConfig(vcodec="h264"), image_writer_threads=2)
    count = 0
    current_ep = 0
    current_frames = 0
    def source_frames(key):
        for path in video_files[key]:
            with av.open(str(path)) as video:
                yield from video.decode(video=0)
    try:
        for items in zip(rows, *(source_frames(key) for key in image_keys), strict=True):
                row, frames = items[0], items[1:]
                ep = int(row["episode_index"])
                if ep != current_ep:
                    if ep != current_ep + 1 or current_frames == 0:
                        raise ValueError("Episode ordering gap")
                    output_ds.save_episode(parallel_encoding=False)
                    print(f"Saved {current_ep+1}/{a.expected}: {current_frames} frames", flush=True)
                    current_ep, current_frames = ep, 0
                if int(row["frame_index"]) != current_frames or abs(float(row["timestamp"]) - current_frames/20) > .002:
                    raise ValueError("Frame/timestamp mismatch")
                if any((frame.height, frame.width, 3) != source_shapes[key]
                       for key, frame in zip(image_keys, frames, strict=True)):
                    raise ValueError("Source video shape differs from Dataset metadata")
                images = {key: np.asarray(Image.fromarray(frame.to_ndarray(format="rgb24")).resize(
                    (224, 168), Image.Resampling.BILINEAR)) for key, frame in zip(image_keys, frames, strict=True)}
                output_ds.add_frame({"observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                                     "action": np.asarray(row["action"], dtype=np.float32),
                                     **images, "task": "Pick up the Muhi bottle and place it into the target container."})
                current_frames += 1
                count += 1
        if current_ep != a.expected - 1 or count != info["total_frames"]:
            raise ValueError("Incomplete source")
        output_ds.save_episode(parallel_encoding=False)
        print(f"Saved {a.expected}/{a.expected}: {current_frames} frames", flush=True)
    finally:
        output_ds.finalize()
    (output / "source_lineage.json").write_text(json.dumps({"source": str(source),
        "transform": "PIL bilinear resize to 224x168", "source_shapes": source_shapes,
        "episodes": a.expected,
        "frames": count, "image_keys": image_keys,
        "action_requested": "excluded"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
