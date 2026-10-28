# SO-101：収録済み実姿勢 vs URDF FK（符号4仮説）比較

実機非接続、**観測した姿勢 `observation.state`** を使う。カメラに写っているロボットをAction（次の目標）で比較しないことが重要。評価対象はepisode0、代表7フレーム。既存データとURDFは変更しない。

```bash
conda activate so101-ik
git pull --ff-only origin feature/cartesian-action-offline-export
python scripts/so101_capture/compare_recorded_pose_fk_signs_offline.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --episode 0 --frames 0 60 120 240 360 480 647 \
  --output outputs/so101-fk-mapping-episode0-v1
```

出力：`fk_comparison.json` とフレームごとの `frame_####_fk.png`。4仮説（++、-+、+-、--）のURDF基準座標XZ/YZ投影を比較する。肩・肘・手首・TCPのリンク座標原点を結ぶ簡易図であり、実機形状の完全なレンダリングではない。ロボット駆動、学習、IK計算、映像ファイルの修正はしない。

**重要な限界：** 映像を自動的に対応付けたわけではない。画像座標へ重ね合わせるにはカメラ内部・外部パラメータ、同期、アームリンク形状が必要。リンク原点を結んだ図は見た目の比較を補助するだけで、単独で符号を決定しない。ゼロオフセットは全て0と仮定。事前の648f数値PASSは--が物理的に正しいという証拠ではない。比較には成功エピソードの該当フレーム実画像を別途確認する必要がある。

結果のpng、jsonと該当画像を照合して、初めて符号候補を絞る。カメラ側の向きが不明なら「未判定」と記録する。
