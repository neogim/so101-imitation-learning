# 7D Cartesian Action Dataset候補の生成（オフライン）

この機能は、既存のSO-101 53エピソード・2カメラDatasetと、すでに計算済みの
7次元Cartesian Action sidecarを使い、**独立したDataset候補**を作ります。
**現時点ではACTで学習できることを保証していません。** 実機は動かしません。

## 事前に分かっていること

- 元データ：53エピソード、30,846フレーム、20fps、side+wrist
- Cartesian Action：形状 `[30846, 7]`、NaN/Infゼロ
- 把持53点でPlaco対Codexの中心化後3D RMSE = **3.68mm**。
  絶対位置平均差は約10mm。URDFと実機の較正一致は未確認。
- SO-101の腕は5自由度。6軸Cartesian差分を任意にIKで再現することはできない。
- IKによる新規撮影・合成エピソード生成はしていない。

## 1. まず元データの空き容量を確認

動画を独立コピーするため、元Datasetと同程度の空き容量が必要です。

```bash
cd ~/Documents/AI/physicalAI2026_docker/so101-imitation-learning
du -sh data/so101/muhi-side-wrist-success-053-20261025-168x224
df -h .
```

## 2. 専用環境で候補を生成

`so101-ik`環境（NumPy 2.3.5、PyArrow）のみで実行できます。
LeRobotのインストールや再学習は不要です。

```bash
conda activate so101-ik
git switch feature/cartesian-action-offline-export
git pull --ff-only origin feature/cartesian-action-offline-export

python scripts/so101_capture/build_cartesian_act_candidate.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025-168x224 \
  --sidecar data/analysis/so101_cartesian_action_53 \
  --output data/so101/muhi-side-wrist-cartesian-candidate-053
```

元Datasetの画像ファイルはコピーされ、`observation.state`（6次元）は保持。
`action`列を7次元に変更し、`meta/info.json`のfeature定義も変更。
Sidecarと各フレームを`episode_index`、`frame_index`、`timestamp`で照合し、
欠落・重複・順序不一致なら生成を止めます。

出力先に `NOT_TRAINING_READY` を付け、既存のaction統計を流用しないようにしています。
元データ、sidecar、元動画は一切更新しません。

## 3. 候補の件数・構造を確認

```bash
python - <<'PY'
from pathlib import Path
import json
import pyarrow.parquet as pq
p = Path("data/so101/muhi-side-wrist-cartesian-candidate-053")
info = json.loads((p/"meta/info.json").read_text())
files = sorted((p/"data").rglob("*.parquet"))
count = sum(pq.read_metadata(f).num_rows for f in files)
assert count == 30846
assert info["total_episodes"] == 53
assert info["features"]["action"]["shape"] == [7]
assert info["features"]["observation.state"]["shape"] == [6]
assert (p/"NOT_TRAINING_READY").exists()
print("Candidate generated:", count, "frames; shape 7; NOT training-ready")
PY
```

## 4. 次に必要な検証

候補Datasetをそのまま `lerobot-train` に渡さないでください。

1. LeRobot 0.6.2のDataset APIで、7D Action・元の6D State・動画を読めるか検証
2. Cartesian Actionに対応する正規化統計の再計算と検証
3. 既存のtrain/validation splitを保持しているか確認
4. ACTが7D Actionを学習・出力できるよう学習設定を準備
5. 推論時に同じFK座標系でデコードし、IKが求解可能かオフライン評価
6. 未解決の自己干渉警告／URDF座標系／関節制約を確認

この候補生成は**学習実行の前段階**です。実機制御コードは変更しません。
