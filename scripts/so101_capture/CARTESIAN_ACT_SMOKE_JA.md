# Cartesian ACT：学習45／検証8エピソード・10ステップスモークテスト

これは**Macで実行する手順**。GitHub上でMacのACT学習を実行したわけではない。

## 前提

- `lerobot` Conda環境 / LeRobot 0.6.2 / macOS MPS
- 候補Dataset：`data/so101/muhi-side-wrist-cartesian-candidate-053`
- 厳密な分割：train 45エピソード26,296フレーム、validation 8エピソード4,550フレーム
- trainのみの正規化統計：`data/analysis/so101_cartesian_trainonly_stats.json`（side/wristはtrainから256枚抽出）
- 入力：state 6、side/wrist各3×168×224。出力：Cartesian Action 7
- FK/IKによるロボット実機への指令は未実装。既存ACT checkpointに上書きしない

## 1. 最新ブランチを取得

```bash
conda activate lerobot
cd ~/Documents/AI/physicalAI2026_docker/so101-imitation-learning
git switch feature/cartesian-action-offline-export
git pull --ff-only origin feature/cartesian-action-offline-export
```

## 2. 独立した学習用Datasetを用意し、10ステップのコマンドを表示

```bash
python scripts/so101_capture/stage_cartesian_act_smoke.py \
  --candidate data/so101/muhi-side-wrist-cartesian-candidate-053 \
  --stats data/analysis/so101_cartesian_trainonly_stats.json \
  --output data/so101/muhi-side-wrist-cartesian-smoke-053 \
  --model-output outputs/act_cartesian_smoke10 \
  --steps 10
```

以下を行う。元の候補Datasetは一切変更しない。

- 新しいDatasetに動画・Parquet・メタデータを独立コピー
- meta/stats.json の Action、State、side、wrist をtrainのみの計算値に置換
- `episode_split.json` と統計監査のtrain/validationエピソードが完全一致するか確認
- trainエピソードを先頭45、validationを最後8にした明示的な `--dataset.episodes` を生成
- `--dataset.eval_split=8/53` を指定し、検証用エピソードを学習から除くことを狙う
- `--steps=10`、MPS、batch size 2、チェックポイントの別出力を指定
- 実際の学習はまだ開始しない

**注意**：`NOT_TRAINING_READY` は、**実機投入・本学習用として未承認**を示すマーカーとして維持する。スモークテスト専用のDatasetを明示的に使う。

## 3. Datasetの読み込みを先に検証

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python - <<'PY'
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root=Path("data/so101/muhi-side-wrist-cartesian-smoke-053").resolve()
ds=LeRobotDataset("local/muhi_cartesian_act_smoke_053",root=root,video_backend="pyav")
sample=ds[0]
assert len(ds)==30846
assert tuple(sample["action"].shape)==(7,)
assert tuple(sample["observation.state"].shape)==(6,)
assert tuple(sample["observation.images.side"].shape)==(3,168,224)
assert tuple(sample["observation.images.wrist"].shape)==(3,168,224)
assert len(ds.meta.stats["action"]["mean"])==7
print("LeRobot Dataset readback OK")
PY
```

## 4. 10ステップの学習を実行

手順2で表示された **`lerobot-train ...` コマンドをコピーして実行**。
すべてのデータがオフラインにある場合は先頭に
`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` を付けてもよい。

チェック項目：

- 45エピソードtrain、8エピソードevalの分割がログ上も正しい
- Actionが7次元、Stateが6次元のまま
- MPS上で学習lossが有限
- 10ステップ完了、モデル出力先 `outputs/act_cartesian_smoke10`
- **元の6D ACTモデルに上書きしない**

CLIが `--dataset.episodes`、`--dataset.eval_split` などを受理しなければ、
**学習を開始せず、エラーログを確認する**。
LeRobotの学習プロセスがユーザー指定のepisode順序を維持するかは
**Mac実行ログでの確認が必要**。特に評価エピソードの混入防止が重要。

## 5. うまくいった場合の残作業

スモークテスト成功は**学習・推論の品質を証明するものではない**。
本学習前にloss、検証データでの誤差、gripper予測、IK到達性、URDF/実機較正不一致、
自己干渉警告を別途評価する。SO-101は5自由度で任意の6D手先姿勢は再現できない。
まだACT出力をモータに接続しない。
