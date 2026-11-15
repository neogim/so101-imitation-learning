# SO-101 把持単体・関節拘束による再構築（2026-11-15）

## 目的と制約
実機のムヒ把持をMuJoCoで再現するため、既存の `physical_grasp_table_bench.py` に**オプトイン**の `--constraint-arm-hold` を追加。既存デフォルトと過去実験を保存する。実機は動かさない。

- J1〜J5：pregraspの間だけ従来のIK/幾何配置。その最終姿勢でMuJoCo equality/joint拘束を有効化する。閉鎖中にJ1〜J5の`qpos/qvel`を書き戻さない。物理PD保持にも切り替えない。
- J6：既存の物理位置アクチュエータで閉鎖。ムヒと爪・机の重力、接触、摩擦は有効。
- Lift：両爪の物理接触が確認できた場合のみ5mmを試し、拘束の参照角度を更新する。物体をテレポートしない。
- **カメラ必須**：`wrist_camera_sim.py`のリスト治具（仮採寸15/35/35mm）、リスト3視点、サイド2視点と1440×780の5視点結合MP4、個別MP4は従来どおり。カメラ治具の衝突形状も維持。
- 接触警告は`--study-observe-through-contact`で記録し、数値発散・例外まで成功扱いしない。

## 初回実行（Mac / simのみ）

```bash
conda activate so101-real2sim
git fetch origin
git switch feature/so101-constraint-grasp-rebuild
git pull --ff-only origin feature/so101-constraint-grasp-rebuild
python -m py_compile scripts/so101_sim/physical_grasp_table_bench.py

OUT="outputs/so101-constraint-grasp-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/physical_grasp_table_bench.py \
  --simple-grasp \
  --fixed-jaw-surface-proxy --moving-jaw-surface-proxy \
  --precise-jaw-proxy --show-jaw-collision-overlay \
  --gentle-jaw-contact \
  --constraint-arm-hold --study-observe-through-contact \
  --target-grasp-depth-min-mm 8 --target-grasp-depth-max-mm 20 \
  --j6-position-speed-deg-s 35 --j6-actual-speed-limit-deg-s 35 \
  --lift-mm 5 --duration-s 7 \
  --render-every 5 --fps 30 --video --out "$OUT"
open "$OUT"
```

**注意：この新コードはまだMacのMuJoCo環境で実行されていない。** `--simple-grasp` は必要な最短時間にdurationを引き上げる場合がある。前回の `--physical-arm-hold` は付けない（両指定はエラー）。

## 最小合否

1. `summary.json`で`constraint_arm_hold_enabled=true`、`constraint_arm_hold_engaged=true`。
2. `table_contact_views.mp4`でサイド2・リスト3視点があり、ロボット/ムヒと同期して動く。個別リストMP4も保持。
3. 接触の直前/直後にJ1〜J5が突然大角度移動しない。接触力・物体速度が異常なら成功ではなく設計不具合と記録。
4. 物理接触力が両爪で持続するか、机から離れて5mm上昇するかを動画とCSVで区別。**未到達は未成功と記載**。

## 重要な残課題

- Equality jointの参照姿勢とソルバーの拘束力、モデル互換性をMac実測で検証する。関節拘束によって根本の接触0N問題まで解決すると断定しない。
- 現状のbenchはムヒをグリッパー近傍に検索配置する**把持単体診断**であり、Home→Pick→Placeや現実のWORLD固定配置の成功証明にはならない。
- カメラの物理位置は写真由来の暫定値。既存の視界を維持するが正確なキャリブレーションを済ませたという意味ではない。
- 動画でSTLとBOXの位置が一致しないなら、その局所箇所を先に修正。意味なくBOX・ガード・グラフを増やさない。
- 失敗が2回続いたら同じパラメータ調整ではなく、まずモデリングとMuJoCo拘束の成立条件を見直す。

参考： [MuJoCo joint equality](https://mujoco.readthedocs.io/en/latest/XMLreference.html#equality-joint) と [既存機能維持契約](../so101_simulation_preservation_contract.md)。
