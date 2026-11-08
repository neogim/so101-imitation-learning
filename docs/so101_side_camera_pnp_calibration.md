# SideカメラのPnP校正（SO-101 Real-to-Sim）

## 目的と注意

繰り返しYaw/Xを手動調整する代わりに、**同一の実写画像内の2D点**と、**SO-101ベース／MuJoCoワールド座標系で独立に測った3D点**から、固定Sideカメラの位置と回転を推定する。

- 実機の再デモ収録やカメラ移設は不要。
- スクリプトはオフラインのPnP処理で、SO-101実機I/O、ROS、LeRobotのモータ操作は使わない。
- **既存のSide画像だけでは、正確な3D対応点やカメラ内部パラメータは自動的には決まらない。**
- 現時点の画像は背景・机縁・ロボットが見えるが、既知3D点が6点以上ある保証はない。存在しなければ校正用マーカーを既知位置に設置し、固定Sideカメラから静止画を1枚撮影する（デモ再収録ではない）。
- 少なくとも6つの異なる**非共面**の既知3D点を使う簡易PnPに限定。机の上の平面だけの4隅ではこの実装は受け付けない。
- PnPの低い再投影誤差は必ずしも真の物理精度を証明しない。3D座標、レンズ歪み、カメラ内部校正、座標系および未使用フレームでの検証が必要。
- 最初は \`+++++\` を暫定仮説として使うが、J2/J3の符号・原点は未確定。ロボットの関節原点から推定した点を採用する場合にはこの不確かさが直接影響する。**可能ならロボット本体とは独立した固定マーカーを使う。**

## Step 1：カメラ内部パラメータを準備する

実Sideカメラの撮影解像度に対応した \`fx,fy,cx,cy\` とレンズ歪みパラメータを用意する。既存の校正結果／実測値を優先。FOV=60°というMuJoCo仮設定を測定済みの内部パラメータと誤認しない。

例：**フォーマットだけを示す仮値。実測値に置換するまで入力しない。**

~~~json
{
  "width": 224,
  "height": 168,
  "fx": 180.0,
  "fy": 180.0,
  "cx": 112.0,
  "cy": 84.0,
  "distortion": [0, 0, 0, 0, 0]
}
~~~

実画像の縦横サイズ、画像のcrop/resize/rotateを確認すること。元の解像度で校正している場合、画像を縮小しているなら内部行列も対応してスケーリングする。収録データのmeta/info.jsonでheight=168、width=224と確認済み。上記fx/fy/cx/cyは説明用の仮値であり実校正値ではない。

## Step 2：対応点を準備する

テンプレートは \`scripts/so101_sim/side_pnp_points_template.csv\` 。以下の列を複製して埋める：

~~~csv
point_id,x_m,y_m,z_m,u_px,v_px
~~~

- \`x_m,y_m,z_m\`：MuJoCo側のworld座標系における実測3D座標（単位m）。3D形状の単なる目測値を入れない。
- \`u_px,v_px\`：**同一のSide画像**上で測ったピクセル座標。左上が(0,0)。画像が回転・crop・resizeされていれば座標の変換が必要。
- 見分けられる固定マーカー角・基準治具・既知位置の複数高さの点など、非共面で画面全体に分布した6点以上を使用。
- 点の3D座標と画面2D座標は**必ず同じ物理的点**。二つの画像を混ぜない。

独立した3D座標・実写上の対応点が準備できない場合は、いったんここで止める。**画像を何度も生成しても、対応点が増えるわけではない。**

## Step 3：専用Conda環境で依存追加

~~~bash
conda activate so101-real2sim
python -m pip install "scipy>=1.11" "opencv-python-headless>=4.8"
python -m unittest discover -s scripts/so101_sim -p 'test_*.py' -v
~~~

## Step 4：PnPの実行（点・内部パラメータが実測済みの場合のみ）

~~~bash
OUT="outputs/real2sim/side-pnp-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/fit_side_camera_pnp.py \
  --points outputs/real2sim/calibration/side_points_measured.csv \
  --intrinsics outputs/real2sim/calibration/side_intrinsics_measured.json \
  --config scripts/so101_sim/workcell.example.json \
  --output "$OUT"
open "$OUT"
~~~

入力画像がある場合にのみ \`--image /path/to/side_frame.png\` を追加。画像解像度を内部校正ファイルと照合し、\`overlay.png\` に黄（実測2D点）と赤（投影点）を描く。

出力：
- \`side_candidate.json\`：推定したワールド座標 \`pos_m\`、MuJoCoの \`quat_wxyz\`、ピンホール近似FOVを含む**派生設定**。常に \`calibrated=false\`。
- \`report.json\`：RANSAC内点数・再投影RMSE・個別点誤差。
- \`overlay.png\`：画像を指定した場合のみ保存。

このツールの推定では、OpenCVのカメラ座標系（+X右/+Y下/+Z前）からMuJoCoのカメラ座標系（+X右/+Y上/-Z前）へ変換する。

## Step 5：MuJoCoで検証

~~~bash
DATASET="data/so101/muhi-side-wrist-success-053-20261025-168x224"
CHECK="outputs/real2sim/side-pnp-validation-$(date +%Y%m%d-%H%M%S)"
python scripts/so101_sim/real_to_sim.py \
  --dataset-root "$DATASET" --episode 0 \
  --stride 80 --max-frames 3 \
  --config "$OUT/side_candidate.json" \
  --render --real-images --output "$CHECK"
open "$CHECK"
~~~

注意：既存の \`real_to_sim.py\` は、SideとWristの両方をレンダリングする。Wristは未校正。Sideの画角・机や物体の配置も未校正なので、一枚のスクリーンショットとのピクセル一致だけで採用しない。

## 研究の次の確認事項

- [x] SideカメラのX/Yaw手動探索を実施し、手動探索だけでは不足と判断
- [x] 非共面3D-2D点からSide姿勢を推定するPnPコードと合成データ単体テストを追加
- [ ] 実Sideカメラの内部校正結果と正しい撮影サイズを確認
- [ ] 非共面・既知の3Dマーカー位置を実測し、同一Side画像と対応付け
- [ ] PnPを実データで実行し、未使用点・フレームで投影誤差を確認
- [ ] J2/J3の角度マッピングを物理的に検証
- [ ] WristのHand-Eye校正へ進む

関係：[Draft PR #8](https://github.com/neogim/so101-imitation-learning/pull/8)／[Real-to-Sim計画](https://app.notion.com/p/40f85e16f4dc81368471db66b32bd029)

## iPhone → OBS Virtual Camera → LeRobot のSide内部校正

**Side = iPhone＋OBS、Wrist = Innomaker U20CAM。** 収録済みSideの実解像度は **幅224 × 高168**。OBSのクロップ・拡大縮小・回転およびLeRobot側リサイズの有無を確認する。iPhone本体の内部パラメータを無条件に転用しない。

既存デモの再収録は不要。ただし、内部校正には、**収録と同じOBSシーン／出力・LeRobotへの入力経路**で撮影したチェッカーボード画像を新規に12枚以上集めることを推奨。固定Sideカメラは動かさず、校正板の位置・向きだけを変える。撮影設定が元の53エピソード時と変わっていれば、新しい校正値を過去に直接適用できない。

[追加された内部校正スクリプト](https://github.com/neogim/so101-imitation-learning/blob/feature/mujoco-real-to-sim/scripts/so101_sim/calibrate_side_obs_intrinsics.py)

~~~bash
conda activate so101-real2sim
python -m pip install "opencv-python-headless>=4.8"

# 事前にチェッカーボード画像を読み取り用フォルダに保存
IMAGES="outputs/real2sim/calibration/side_chessboard_images"
OUT="outputs/real2sim/side-intrinsics-$(date +%Y%m%d-%H%M%S)"

# 以下の7×5は「内側の角の数」、20mmは仮の正方形一辺。
# 実際の校正板の寸法に置き換えて実行する。
python scripts/so101_sim/calibrate_side_obs_intrinsics.py \
  --images "$IMAGES" --columns 7 --rows 5 --square-mm 20 \
  --output "$OUT"
open "$OUT"
~~~

出力：intrinsics.json（PnP入力形式）およびreport.json。PnP前に内部校正の撮影サイズ、再投影RMS、画面内の点の分布、別画像での再現性をチェックする。

**重要：** 224×168に縮小した画像ではチェッカーボード角の精度が不足し得る。必要なら高解像度で校正し、既知のOBS／LeRobot変換を内部行列に反映する。校正写真は既存実験データとは別管理。
