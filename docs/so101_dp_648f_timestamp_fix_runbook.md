# 648フレーム再実行：タイムスタンプ修正版 / IK候補クラスタ分析方針

## 修正理由
0〜119では20Hzにほぼ一致したが、全0〜647のタイムスタンプは少なくとも一箇所で50ms±2μsに一致せず、以前の候補生成が ValueError: Expected recorded 20 Hz timestamps で終了した。元データを補間・修正するのではなく、**単調増加の実タイムスタンプを採用**し、最小/中央値/最大dtと非50ms区間数をsummaryに残す。DPの速度、加速度は以前から実測timestamp差を使っている。実機動作やACT生成なし。

## 実行
```bash
git switch feature/cartesian-action-offline-export
git pull --ff-only origin feature/cartesian-action-offline-export
python -m py_compile scripts/so101_capture/generate_independent_ik_candidates_offline.py scripts/so101_capture/select_ik_trajectory_dp_offline.py
test -f "$FOLLOWER_CALIBRATION_JSON" && echo "Calibration OK"
```

**旧v1出力が存在しても削除しない。新しいv2出力を使う。**

```bash
python scripts/so101_capture/generate_independent_ik_candidates_offline.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --split-dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --mappings 'shoulder-_elbow-' \
  --output outputs/so101-ik-independent-minusminus-000-647-v2 \
  --episode 0 --frame-start 0 --frame-end 647 \
  --offset-mm 3 --feasible-residual-mm 2 --max-nfev 100 \
  > outputs/so101-ik-minusminus-generation-console-v2.log
```

まず生成summaryの各方向 `witness_targets` と `timestamp_diagnostics` を確認。全648目標が見つからなくても、候補がないフレームは全区間の品質判定をFAILにする。欠損フレームを飛ばして合格にしない。

```bash
python scripts/so101_capture/select_ik_trajectory_dp_offline.py \
  --candidates outputs/so101-ik-independent-minusminus-000-647-v2/per_seed.json \
  --dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --calibration "$FOLLOWER_CALIBRATION_JSON" \
  --output outputs/so101-ik-dp-minusminus-000-647-v2 \
  --episode 0 --frame-start 0 --frame-end 647 \
  > outputs/so101-ik-minusminus-dp-console-v2.log

python scripts/so101_capture/summarize_dp_gate_offline.py \
  --summary outputs/so101-ik-dp-minusminus-000-647-v2/summary.json \
  --mappings 'shoulder-_elbow-'
```

ログの `timestamp_diagnostics` で dt_max を確認し、サンプリング欠落/時間の飛びなどが疑われる場合は、その区間を特に調査。一定周期の映像・制御だったことをこのログだけから断定しない。モータ制御上の安全基準として数値ゲートを使用しない。

## 候補クラスタ分析について
提案は妥当。局所的なJ5のバラつきは大きいが、それだけではIK解が「複数の枝（cluster）」に分かれるか不明。同一フレーム・同一方向の**5関節角候補**を関節角距離でクラスタリングし、各クラスタの代表解とフレーム間の接続性を調べると、枝飛びなのか候補の疎らさなのか区別できる可能性がある。ただし、固定閾値のクラスタはローカルな群分けに過ぎず、数学的に別のIK branchの存在を証明しない。優先順位は、まずタイムスタンプ問題を解消し648フレームの数値品質を確定すること。その後、FAILフレームに限定して必要なクラスタ分析を実施すること。
