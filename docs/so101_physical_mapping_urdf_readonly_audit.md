# SO-101：648フレーム数値PASS後の実機整合性監査（非駆動・読み取り専用）
2026-10-28。今回の調査は**ロボットに接続しない**。URDFやキャリブレーションを書き換えない。

## 現状
Episode0 f0–647、仮説shoulder-_elbow-：6方向×648フレームのDP2数値ゲートPASS。J2 shoulder_lift とJ3 elbow_flexの角度符号を負と仮定した場合の結果であり、正しい物理変換と確認されたわけではない。反対仮説++は120フレームで2/6PASS。位置のみIK、衝突、荷重、トルク、実機姿勢未検証。PlacoはニュートラルURDFに自己衝突の警告を表示。

## 既存ファイルの監査（実機操作なし）
```bash
git switch feature/cartesian-action-offline-export
git pull --ff-only origin feature/cartesian-action-offline-export
python -m py_compile scripts/so101_capture/audit_physical_mapping_urdf_readonly.py
python scripts/so101_capture/audit_physical_mapping_urdf_readonly.py \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --calibration "$FOLLOWER_CALIBRATION_JSON" \
  --output outputs/so101-ik-physical-mapping-metadata-v1.json
```

出力：J1〜J6のURDF関節名、関節軸、可動角制限、親子リンク、校正範囲メタ情報、URDFリンクのcollision要素数とgeometry種別。**校正範囲だけではモータ角度の増減方向とURDF正方向の対応は分からない**。衝突要素の存在だけでは自己衝突警告の正否は分からない。

## 次の検証項目
1. 既存のSO-101サーボ設定・キャリブレーション仕様から角度変換式を文書化。必要な証拠はサーボID/方向/校正中心/生tickからdegree変換する箇所、URDF axis/origin、既知姿勢での一致。J2/J3の符号とゼロ位置を別々に検証する。**ロボットへコマンドを送る前に**、設計担当者またはメーカー資料と照合する。
2. ニュートラル自己衝突警告は、URDFのcollision meshが外観用meshと重なる/隣接リンクを衝突対象に含めている/原点や寸法が違う等の可能性を調査する。リンクごとの詳細メッシュ・衝突除外設定・実姿勢を比較する。現段階の警告は実機の接触の証拠ではない一方、無視して安全と判断することもできない。
3. 数値DP2ゲートは研究用であり、実機駆動許可ではない。衝突、速度制限、トルク、TCP位置、キャリブレーション、非常停止の評価は未了。実機検証計画だけを別途作成する。

今回の出力ログをChatGPTに貼り付ければ、必要な資料不足とURDFモデルの検証箇所を整理できる。
