# Cartesian ACT オフライン評価（検証8エピソード）

## 目的
学習済みACTの **チャンク先頭の予測Action（7D）** と、同じフレームの教師Actionを比較する。実機は動かさない。IKは実行しない。

## 現時点での検証内容
- 評価エピソード：5,8,15,18,23,31,46,51 のみ
- 学習用45エピソードを評価に混ぜない
- 評価サンプルは各エピソード最大20フレーム、20フレームごと（160フレーム以内）
- 入力：side＋wrist画像、observation.state
- モデル：保存済み ACT checkpoint をロードし eval モードで推論
- 正規化：checkpoint内の pre/post processor を必ず経由
- **毎フレーム新しいチャンクを予測し先頭1個のみ比較**。n_action_steps=50の実行キューの評価ではない。
- GT: dataset の action[7]
- 保存：predictions.csv, metrics.json, gripper.png, dx/dy/dz.png
- 単位：XYZ(m)、回転ベクトル(rad)、グリッパ(0〜100)
- Metric: 次元別MAE/RMSE、gt/predのグリッパ>25件数

### コマンド

```bash
conda activate lerobot
git switch feature/cartesian-action-offline-export
git pull --ff-only origin feature/cartesian-action-offline-export

python scripts/so101_capture/evaluate_cartesian_act_offline.py \
  --dataset data/so101/muhi-side-wrist-cartesian-smoke-053 \
  --checkpoint outputs/act_cartesian_smoke10/checkpoints/000010/pretrained_model \
  --output outputs/act_cartesian_eval_smoke10 \
  --stride 20 \
  --limit 160
```

`--limit` は8エピソードのそれぞれから最大 `limit//8` 点を抽出。結果フォルダは存在しないこと。

### 結果の読み方
- `mae.dx` 等はメートル。mmに換算するなら1000倍する。
- `mae.drx` 等はラジアン。度に換算するなら180/π倍する。
- `mae.gripper` は0〜100単位の平均絶対誤差。
- グリッパ `gt_over25` が多いのに `pred_over25` がほぼ0なら前回6D ACTと同じ「閉じる指令を予測できない」兆候。
- ただし **10ステップモデルでは精度が悪いことが想定内**。評価経路の動作確認が目的。

### 注意事項
- この評価コードはLeRobot 0.6.2のユーザー環境にて**未実行**。import/API互換エラーがあればログを元に修正。
- チャンクの後続1〜49ステップの精度・閉ループ把持成功率は評価できない。
- 教師データの7D Cartesian ActionはFK計算に基づくが、IKを介して実機コマンドに戻せる保証はない。
- 実機投入には、IK求解・関節制限・衝突確認・実機較正が別途必要。
