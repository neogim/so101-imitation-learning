# SO-101 side画像とURDFの幾何照合（探索的・非駆動）

## 目的
成功エピソード0の実画像と記録済み `observation.state` を使い、4種類のJ2/J3符号仮説を同一の固定カメラ姿勢で投影して比較する。**数値合格や単純FK画像の見た目から正しい物理符号を推定しない**。

## 入力
- データセット `data/so101/muhi-side-wrist-success-053-20261025`
- URDF `data/urdf/so101/so101_new_calib.urdf`
- side動画から抽出済みのEpisode0 frame 0/240/480（可能なら追加フレーム）。カメラが動いていないこと。
- 画像上の**URDFリンク原点と実物で同一物理位置と判別できる**特徴点の画素座標。メッシュ外観の角と関節回転中心を混同しないこと。

## 注釈フォーマット（例：値をでっち上げず、実測座標を入力）
```json
{
  "width": 320,
  "height": 240,
  "frames": [
    {"frame": 0, "points": {"shoulder_link": [0, 0], "upper_arm_link": [0, 0]}},
    {"frame": 240, "points": {"shoulder_link": [0, 0], "upper_arm_link": [0, 0]}},
    {"frame": 480, "points": {"shoulder_link": [0, 0], "upper_arm_link": [0, 0]}}
  ]
}
```
**上の0,0は構造の説明用ダミーであり、そのまま解析に使用しない。** 画像上で対応がはっきりしないリンクは注釈しない。8点以上の対応が必要。カメラ画像の実寸を入力する。カメラを再設置していた場合は同一カメラ姿勢の仮定が成立しない。

## 実行
```bash
conda activate so101-ik
git pull --ff-only origin feature/cartesian-action-offline-export
python scripts/so101_capture/fit_camera_fk_signs_offline.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025 \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --annotations outputs/so101-camera-fk-landmarks-v1.json \
  --episode 0 \
  --output outputs/so101-camera-fk-sign-fit-v1.json
```

## 判断基準
1. 全フレームで共通カメラ外部パラメータが妥当な位置・向きを示すか。
2. 再投影RMSEを4仮説で比較する。ただし、暫定焦点距離と未知ゼロオフセットが原因で正解の仮説も悪いRMSEになる場合がある。
3. 未使用フレームでの再投影を検証する。学習した3フレームだけの一致では過適合。
4. **最終的なJ2/J3の実機マッピング確定は、機械的な関節中心と校正ゼロ位置との照合まで保留する。**

重要：このコードは既存データ・URDF・実機モーターを変更しない。推定焦点距離と主点（画像中心）を仮定した探索的プロトタイプであり、カメラキャリブレーションの確定結果ではない。
