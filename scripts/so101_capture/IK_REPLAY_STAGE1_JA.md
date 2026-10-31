# SO-101 Stage 1：IKオフライン再生検証

目的：実機を動かさず、既存の6D joint actionに対応するTCP目標をFKで計算し、観測stateから数値IKで再計算したJ1〜J5と元の指令を比較する。**新しい学習データを生成する工程ではない。**

## 実行

```bash
conda activate so101-ik
python -m pip install scipy
git switch feature/cartesian-action-offline-export
git pull --ff-only origin feature/cartesian-action-offline-export

python scripts/so101_capture/evaluate_ik_replay_offline.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025-168x224 \
  --split-dataset data/so101/muhi-side-wrist-cartesian-smoke-053 \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --episode 0 \
  --stride 20 \
  --max-frames 30 \
  --output data/analysis/so101_ik_replay_ep0
```

`--dataset`には必ず元の6D関節Action Datasetを指定する。7D Cartesian Datasetは不適合。`--split-dataset`は学習45／検証8の分割リストだけを参照。

出力：`summary.json`、`ik_replay.csv`、`ik_replay_detailed.json`。位置誤差の平均・P95・最大、5mm以内の点数、関節角の差、最適化成否を確認。

## 制限

- 位置優先IKで、姿勢まで満たしたという意味ではない。
- ターゲットは同一URDFのFK(action)から生成されるので、既知の軌道の「再求解」テストである。モデルの真の幾何較正や新軌道の性能を証明しない。
- 関節制限・自己干渉・障害物衝突・実機ゼロ点の較正をこのスクリプトでは保証しない。
- 各サンプルはobserved stateから初期化し、時系列IKとしての連続性評価はまだ行わない。
- URDFの絶対TCP位置には別実装との約10mm差があり、実機投入は禁止。
- 学習用の合成デモ画像・関節軌道は出力しない。Stage 2以降は本IK評価結果と既存データで判断する。
