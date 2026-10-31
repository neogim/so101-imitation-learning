# Stage 1: 元6D関節Action版 ACT の学習・検証（J1〜J5ベースライン）

目的は、合成画像やIK再ラベルなしで、元の実演を使って腕（J1〜J5）の学習能力を切り分けること。**この段階の学習損失はJ1〜J6を均等に扱う標準ACT**。J1〜J5重み付け損失の導入やJ6の固定は行わない。後段の把持FTと重み互換を保つため出力6Dを維持。

## 実行前確認

```bash
conda activate lerobot
git switch feature/cartesian-action-offline-export
git pull --ff-only origin feature/cartesian-action-offline-export

python - <<'PY'
import json
from pathlib import Path
src=Path("data/so101/muhi-side-wrist-success-053-20261025-168x224")
split=Path("data/so101/muhi-side-wrist-cartesian-smoke-053/episode_split.json")
i=json.loads((src/"meta/info.json").read_text())
s=json.loads(split.read_text())
print("FPS",i["fps"],"ACTION",i["features"]["action"]["shape"],
      "STATE",i["features"]["observation.state"]["shape"],
      "TRAIN",len(s["train_episode_indices"]),"VAL",len(s["validation_episode_indices"]))
assert i["features"]["action"]["shape"]==[6]
assert i["features"]["observation.state"]["shape"]==[6]
assert set(s["train_episode_indices"]).isdisjoint(s["validation_episode_indices"])
assert len(s["train_episode_indices"])==45 and len(s["validation_episode_indices"])==8
PY
```

## 学習データのリーク防止

LeRobot Dataset のデフォルト `meta/stats.json` は53エピソード全体の統計である可能性がある。**train-only統計に差し替えた別Dataset**を作成してから使う。既存のCartesian用 `meta/stats.json` は7Dなので6Dに流用しない。

まず試験として元6D Datasetの`meta/stats.json`がオリジナルと共通になっている場合、`--dataset.use_imagenet_stats=false`だけでは漏洩は防げない。したがって現段階では**以下の学習コマンドをそのまま実行しない**。6D train-only statsを作る実装を用意するまでStage1の確定実験は保留。

## 予定学習コマンド（6D train-only Dataset作成後）

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 lerobot-train \
  --dataset.repo_id=local/muhi_joint6_trainonly_053 \
  --dataset.root="$(pwd)/data/so101/muhi-side-wrist-joint6-trainonly-053" \
  --dataset.video_backend=pyav \
  '--dataset.episodes=[0,1,2,3,4,6,7,9,10,11,12,13,14,16,17,19,20,21,22,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,48,49,50,52]' \
  --dataset.eval_split=0 --dataset.use_imagenet_stats=false \
  --policy.type=act --policy.device=mps --policy.push_to_hub=false \
  --policy.pretrained_backbone_weights=null --policy.chunk_size=50 --policy.n_action_steps=50 \
  --policy.dim_model=128 --policy.n_heads=4 --policy.dim_feedforward=512 \
  --policy.n_encoder_layers=2 --policy.n_vae_encoder_layers=2 --policy.optimizer_lr=1e-5 \
  --steps=20000 --batch_size=2 --num_workers=0 --eval_steps=0 --env_eval_freq=0 \
  --log_freq=100 --save_freq=2000 --save_checkpoint_to_hub=false \
  --wandb.enable=false --output_dir="$(pwd)/outputs/act_joint6_stage1_20k"
```

評価はheldout 8でJ1-J5の教師Action vs モデル予測を別に測定。ACT 7D evaluatorを6Dへそのまま使わない。**IK計算成功 ≠ ACT学習成功**。あとでJ6の追加学習を行う際、元6D checkpointとの互換性を利用する。

## 注意

- フル6Dを学習する構成は「J1〜J5に重み付き重点学習する」ものではなく、まず6D ACTの腕部分を計測するベースライン。
- URDFのJ2・J3で元Action/IK双方に関節制限違反があるので、この工程のモデルを実機投入しない。
- 既存7D Cartesian ACTと、6D joint ACTは出力の意味・尺度が異なるので直接重みを引き継がない。
