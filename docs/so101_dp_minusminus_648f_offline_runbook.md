# SO-101: 仮説 -- のみ648フレーム全体をオフライン検証

対象：Episode 0、0〜647（実際のフレーム数が648であることを事前確認）。これはユーザーのローカルMacでのみ実行する数値計算。前段階0〜119の独立seed/DP2検証は--が6/6、++が2/6。++は拡張対象外。肩肘符号--は実機の正しいマッピングであると確認されたわけではない。

## Mac実行

```bash
git switch feature/cartesian-action-offline-export
git pull --ff-only origin feature/cartesian-action-offline-export
test -f "$FOLLOWER_CALIBRATION_JSON" && echo "calibration file found"
python -m py_compile scripts/so101_capture/generate_independent_ik_candidates_offline.py scripts/so101_capture/select_ik_trajectory_dp_offline.py scripts/so101_capture/summarize_dp_gate_offline.py
```

最初にソースデータのEpisode 0が648フレームであることをローカルのデータ情報から確認する（異なる場合は引数と計画を合わせる）。以前の31、120フレームの出力は書き換えない。保存先は一意の新規パスを使う。

```bash
python scripts/so101_capture/generate_independent_ik_candidates_offline.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --split-dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --mappings 'shoulder-_elbow-' \
  --output outputs/so101-ik-independent-minusminus-000-647-v1 \
  --episode 0 --frame-start 0 --frame-end 647 \
  --offset-mm 3 --feasible-residual-mm 2 --max-nfev 100 \
  > outputs/so101-ik-minusminus-generation-console-v1.log
```

生成結果のsummary.jsonで各6方向648/648 witnessがなければDP評価はできても全区間品質はFAILとなる。全方向を確認してから以下を実行。

```bash
python scripts/so101_capture/select_ik_trajectory_dp_offline.py \
  --candidates outputs/so101-ik-independent-minusminus-000-647-v1/per_seed.json \
  --dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --calibration "$FOLLOWER_CALIBRATION_JSON" \
  --output outputs/so101-ik-dp-minusminus-000-647-v1 \
  --episode 0 --frame-start 0 --frame-end 647 \
  > outputs/so101-ik-minusminus-dp-console-v1.log

python scripts/so101_capture/summarize_dp_gate_offline.py \
  --summary outputs/so101-ik-dp-minusminus-000-647-v1/summary.json \
  --mappings 'shoulder-_elbow-'
```

## 判定と重要な限界

ここでは一つの連続648フレーム軌道をDP2が選ぶため、区間境界のジャンプを見落とさない。各方向で候補カバー648/648、残差≤2mm、recorded Action偏差≤15°、jump≤5°、velocity≤100°/s、acceleration≤2000°/s²、orientation change≤15°で6/6合格することを数値品質ゲートとする。閾値は研究用。もしFAILなら不合格方向・フレームを診断する。閾値を緩めたり欠損を飛ばしてPASS扱いしない。

**位置のみIK**のためTCP姿勢は拘束されず、衝突・接触・トルクは未評価。Placoのニュートラル姿勢自己衝突警告は未解決。肩/肘モータ角からURDF角への物理的符号・ゼロオフセットは未検証。数値合格しても実機再生や画像を変更せずActionだけ改変するACTデータ生成は行わない。結果のコンパクト集計だけChatGPTに共有し、Google Sheets・レポートに記録する。
