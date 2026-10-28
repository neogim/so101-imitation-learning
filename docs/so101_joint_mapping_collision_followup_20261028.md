# SO-101 648f合格後：J2/J3変換式とニュートラル自己衝突の非駆動監査

## 2026-10-28 GitHubのURDF/コードから確定した事実
- URDF `data/urdf/so101/so101_new_calib.urdf` のJ2 `shoulder_lift` は親 `shoulder_link` →子 `upper_arm_link`、J3 `elbow_flex` は親 `upper_arm_link` →子 `lower_arm_link`。axisはそれぞれローカルZ（0,0,1）。joint originには個別のrpyがあり、単純なワールド座標Z回転ではない。
- `evaluate_mapping_hypotheses_offline.py` の変換は `motor_to_urdf_degrees(values, signs) = values*signs`。仮説 `shoulder-_elbow-` の符号はJ1..J5で `(+,-,-,+,+)`。ゼロオフセット項はない。従って648f 6/6 PASSは仮説の数値安定性を示すが、物理的な角度の符号・原点の証明ではない。
- URDFの `<visual>` と `<collision>` は同じSTLメッシュを多く参照。Placoでニュートラル時に6組の部品間衝突警告。メッシュが重なるのは隣接リンクの連結機構・詳細STL・パーツ名に関係するかもしれないが、未検証。
- URDFのJ3付近に「5-degree calibration offset applied to joint limits」と記述されている。しかし、これは正しいホームオフセット変換式が実装済みであるという証明ではない。limitsのコメントを角度補正式と混同しない。

## Mac一括検査（読み取り専用）
```bash
git switch feature/cartesian-action-offline-export
git pull --ff-only origin feature/cartesian-action-offline-export
python -m py_compile scripts/so101_capture/audit_urdf_collision_pairs_readonly.py
python scripts/so101_capture/audit_urdf_collision_pairs_readonly.py \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --output outputs/so101-urdf-collision-pairs-v1.json
```

このコマンドは衝突形状とリンク関係をまとめる。実際のmesh幾何交差の検証はしない。既存の `outputs/so101-ik-physical-mapping-metadata-v1.json` は保持する。結果出力は上書きしない。

## 実機の角度変換を確定するのに必要な根拠
1. 使用しているLeRobotのSO-101 follower実装とバージョン（ローカル環境）。サーボの生tickと行動角度変換の `homing_offset`、`drive_mode`、`range_min/max`、角度単位がどこで適用されているか。今回のGitHub repoの仮説関数だけでは分からない。
2. J2/J3の正方向と基準ゼロ位置を、既知の構造・設計図・既知姿勢と照合する。必要な場合も先に実機を動かさない読み出し方法を設計してから、明示的な承認を得る。
3. Placoの部品レベルの6警告をURDFのリンクペアと対応づけ、隣接リンク用の無視設定の有無を確認。実際の衝突有無はメッシュ交差・実機寸法・姿勢比較で別途確認。

## 停止条件
ロボットを接続・動作させるコマンド、校正やURDFの変更、ACTデータ生成は含めない。648フレームの数値PASSは研究用の品質ゲートであり、実機の再生許可ではない。結果ログを共有して次の調査範囲を確定する。
