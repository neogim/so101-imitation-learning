"""Offline validation and self-contained HTML viewer for a short SO-101 dataset."""
import argparse
import base64
import io
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--body-unit", choices=["degrees", "normalized", "unknown"], default="unknown")
    a = p.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = json.loads((a.root / "meta/info.json").read_text())
    features = info["features"]
    required = ["observation.state", "action", "observation.images.iphone"]
    missing = [k for k in required if k not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    ds = LeRobotDataset("local/so101_probe", root=a.root.resolve(), video_backend="pyav")
    expected = [n + ".pos" for n in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]]
    errors, warnings, rows = [], [], []
    for key in required[:2]:
        if features[key].get("names") != expected:
            errors.append(f"Unexpected joint names/order: {key}")
    duplicates, last = 0, None
    for i in range(len(ds)):
        item = ds[i]  # Decode EVERY image: missing/corrupt videos must fail validation.
        state = np.asarray(item["observation.state"])
        action = np.asarray(item["action"])
        image = np.asarray(item[required[2]])
        if state.shape != (6,) or action.shape != (6,):
            raise ValueError(f"Wrong joint shape at frame {i}")
        if not all(np.isfinite(v).all() for v in [state, action, image]):
            raise ValueError(f"Nonfinite values at frame {i}")
        if image.shape != (3,480,640):
            raise ValueError(f"Wrong image shape at frame {i}: {image.shape}")
        if image.min() < 0 or image.max() > 1:
            raise ValueError(f"Expected decoded RGB float [0,1] at frame {i}")
        rgb = (image.transpose(1, 2, 0) * 255).round().astype(np.uint8)
        duplicates += int(last is not None and np.array_equal(last, rgb))
        last = rgb
        thumb = Image.fromarray(rgb)
        thumb.thumbnail((480, 360))
        buf = io.BytesIO()
        if i % max(1, len(ds)//250) == 0:
            thumb.save(buf, format="JPEG", quality=75)
        rows.append(dict(index=i, episode=int(item["episode_index"]), frame=int(item["frame_index"]),
                         t=float(item["timestamp"]), state=state.tolist(), action=action.tolist(),
                         image=base64.b64encode(buf.getvalue()).decode()))
    episodes = sorted(set(r["episode"] for r in rows))
    if episodes != list(range(info["total_episodes"])):
        errors.append("Episode indices/metadata are not contiguous")
    if not rows or len(rows) != info["total_frames"]:
        errors.append("Empty dataset or metadata frame count mismatch")
    periods = {}
    for ep in episodes:
        subset = [r for r in rows if r["episode"] == ep]
        t = np.array([r["t"] for r in subset])
        if [r["frame"] for r in subset] != list(range(len(subset))):
            errors.append(f"Noncontiguous frame indices in episode {ep}")
        if not np.isfinite(t).all() or not np.allclose(t, np.arange(len(t))/info["fps"], atol=1e-5):
            errors.append(f"Invalid nominal timestamp grid in episode {ep}")
        periods[str(ep)] = dict(frames=len(t), min_dt=float(np.diff(t).min()) if len(t)>1 else None,
                               max_dt=float(np.diff(t).max()) if len(t)>1 else None)
    ranges = {}
    for key in ["state", "action"]:
        values = np.array([r[key] for r in rows])
        if len(values):
            ranges[key] = {n: [float(values[:,j].min()), float(values[:,j].max())] for j,n in enumerate(expected)}
            if (values[:,5] < -1).any() or (values[:,5] > 101).any():
                warnings.append(f"{key}: gripper outside expected 0..100 range")
            if a.body_unit == "normalized" and (np.abs(values[:,:5]) > 101).any():
                warnings.append(f"{key}: body outside expected -100..100 range")
    # Count/decode each physical video and match per-episode v3 offsets to Parquet rows.
    import av
    import pyarrow.parquet as pq
    video_pts = {}
    for path in sorted((a.root/'videos').rglob('*.mp4')):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if (stream.height, stream.width) != (480,640):
                errors.append(f"Invalid video dimensions: {path}")
            pts = [float(f.pts*f.time_base) for f in container.decode(video=0)]
        video_pts[str(path.relative_to(a.root))] = np.array(pts)
    meta_rows = []
    for path in sorted((a.root/'meta/episodes').rglob('*.parquet')):
        meta_rows.extend(pq.read_table(path).to_pylist())
    video_counts = {}
    prefix = 'videos/observation.images.iphone'
    for meta in meta_rows:
        ep = int(meta['episode_index'])
        chunk, file = int(meta[prefix+'/chunk_index']), int(meta[prefix+'/file_index'])
        path = f'{prefix}/chunk-{chunk:03d}/file-{file:03d}.mp4'
        pts = video_pts.get(path, np.array([]))
        start, stop = meta[prefix+'/from_timestamp'], meta[prefix+'/to_timestamp']
        count = int(((pts >= start-1e-5) & (pts < stop-1e-5)).sum())
        video_counts[str(ep)] = count
        if count != periods.get(str(ep), {}).get('frames') or int(meta['length']) != count:
            errors.append(f'Video/Parquet/metadata length mismatch: episode {ep}')
    if len(meta_rows) != len(episodes) or sum(len(v) for v in video_pts.values()) != len(rows):
        errors.append('Video total or episode metadata count mismatch')
    # Verify physical Parquet global index and optional raw Leader action.
    raw = []
    for path in sorted((a.root/'data').rglob('*.parquet')):
        raw.extend(pq.read_table(path).to_pylist())
    if [r['index'] for r in raw] != list(range(len(rows))):
        errors.append('Global Parquet indices are not contiguous')
    for i, row in enumerate(raw):
        if 'action_requested' in row:
            v = np.asarray(row['action_requested'])
            if v.shape != (6,) or not np.isfinite(v).all():
                errors.append(f'Invalid action_requested at {i}')
    outcomes_path = a.root/'episode_outcomes.json'
    if outcomes_path.exists():
        outcomes = json.loads(outcomes_path.read_text())
        if len(outcomes) != len(episodes):
            errors.append('Outcome count mismatch')
        for row in outcomes:
            if row['frames'] != periods.get(str(row['episode_index']), {}).get('frames'):
                errors.append(f"Outcome frame mismatch: {row['episode_index']}")
    if duplicates:
        warnings.append(f"{duplicates} identical adjacent images; static scene or repeated camera frames")
    warnings.append("Real capture cadence, image/state skew and Wi-Fi latency cannot be recovered from nominal dataset timestamps")
    if a.body_unit == "unknown":
        warnings.append("Units cannot be inferred from numbers; retain recording config")
    report = dict(errors=errors, warnings=warnings, episodes=episodes, frames=len(rows),
                  fps=info["fps"], periods=periods, ranges=ranges, video_frames=video_counts,
                  body_unit_from_user=a.body_unit, gripper_unit="normalized 0..100 (SO-101 source)")
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    payload = json.dumps([r for r in rows if r["image"]]).replace("<", "\\u003c")
    html = '''<!doctype html><meta charset="utf-8"><title>SO-101 episode</title>
<style>body{font:16px sans-serif;max-width:960px;margin:30px auto}img{max-width:100%}input{width:100%}pre{white-space:pre-wrap}</style>
<h1>SO-101 episode</h1><p>Nominal dataset time; body units are in report.json. Gripper: normalized 0–100.</p>
<input id="slider" type="range" min="0" value="0"><img id="pic"><pre id="values"></pre>
<script>const rows=PAYLOAD;const s=document.getElementById('slider');s.max=Math.max(0,rows.length-1);
function show(){const r=rows[+s.value];if(!r)return;document.getElementById('pic').src='data:image/jpeg;base64,'+r.image;
const {image,...v}=r;document.getElementById('values').textContent=JSON.stringify(v,null,2);}
s.oninput=show;show();</script>'''.replace("PAYLOAD", payload)
    (a.output / "viewer.html").write_text(html)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
