# SO-101 Sideカメラ｜ChArUco内部校正の手順

## 目的と既存成果
固定iPhone 6 → OBS Sideソースの**最終640×480画像**に対応した内部パラメータ（fx/fy/cx/cy、歪み）を検証する。既存チェッカーボード16枚では fx≈540.17, fy≈539.19, cx≈317.35, cy≈231.55、RMS≈0.315px の**暫定値**を得ているが、デスクトップスクリーンショット中央の切り出し640×480だったため今回のArUco録画と幾何的に同一とは未確認。既存の記録は保全する。

ChArUcoは内部校正用。ロボットに貼ったID0/1/2のArUcoはEye-to-Hand外部校正用であり、役割を混同しない。

## ボード印刷
A4・7×9マス、1マス20mm・ArUco正方形14mm、DICT_4X4_50、全体140×180mm（内側コーナー6×8=48）。Gitで管理する原本：[`assets/calibration/charuco_7x9_20mm_14mm.svg`](../assets/calibration/charuco_7x9_20mm_14mm.svg)。SVGは140×180mmの実寸ベクターボード。A4で確認線のあるPDFと1400×1800 PNGは再生成可能。

```bash
python scripts/so101_capture/generate_charuco_board.py \\
  --out-dir outputs/charuco-board-01
```

出力 `charuco_A4_7x9_square20mm_marker14mm.pdf` は100%／実際のサイズで印刷する（用紙に合わせるをOFF）。印刷された100mm確認線と1マスを定規で測り、硬い平板に貼り付ける。既存のPDF/PNGをGitへバイナリとして直接格納したわけではない。**Git管理対象はSVG版ボードと再生成スクリプト**。

## 実機を動かさずOBSだけで収録
これまで使った**ローカル** `scripts/so101_capture/cameras.aruco-640.local.yaml` を使用。OBS WebSocket認証、既存source UUIDと設定照合が必要。個人のカメラ設定・パスワードをGitHubへ入れない。

```bash
conda activate lerobot
git pull --ff-only
python -c 'import cv2; print(cv2.__version__, hasattr(cv2.aruco,"CharucoDetector"))'
python scripts/so101_capture/capture_side_charuco_stills.py \
  --camera-config scripts/so101_capture/cameras.aruco-640.local.yaml \
  --out-dir outputs/so101-charuco-stills-01 --frames 25
```

上記は**プレフライトのみ**。次はOBSを開き、実際に撮影（ロボットのシリアルポートには接続しない）：

```bash
PYTHONPATH="$PWD/.local/camera-deps${PYTHONPATH:+:$PYTHONPATH}" \
python scripts/so101_capture/capture_side_charuco_stills.py \
  --camera-config scripts/so101_capture/cameras.aruco-640.local.yaml \
  --out-dir outputs/so101-charuco-stills-01 --frames 25 --execute
```

画面の中央・四隅・左右上下にボードを移し、距離や傾き（上下・左右）を変えて、各姿勢で静止してEnter。撮影終了は `q`。マーカーが小さすぎる／ピンボケ／反射／ボードが曲がる場合はやり直す。iPhone・OBSのクロップ・ズーム・回転を変更しない。出力先が既存ならスクリプトは上書きせず停止する。

PNG保存だがOBS APIから取得するのはJPEG screenshotで、圧縮による情報損失は残る。ArUco録画と同じAPIを使うための選択。

## オフライン内部校正
```bash
python scripts/so101_capture/calibrate_side_charuco_offline.py \
  --images outputs/so101-charuco-stills-01 \
  --out-dir outputs/so101-charuco-intrinsics-01 \
  --min-corners 10 --min-train 12
```

- 画像サイズが混在するとエラー。ChArUcoコーナーが少なすぎる画像は除外。
- 受理した画像の5枚ごとに1枚を保持用とし、残りで `cv2.calibrateCameraExtended` によるKと歪みを推定。学習12枚、保持3枚以上必要。
- `intrinsics.json`：幅、高さ、fx/fy/cx/cy、distortion。
- `report.json`：学習時RMSと保持画像のPnP再投影誤差。
- 保持画像でもボードのPnP姿勢は画像ごとに最適化するため、これは**物理3D精度の独立証明ではない**。画角端・距離・姿勢を変えて再検証する。
- このコードはカメラ接続もロボット操作もせず、入力画像を読み取り、新規出力先に保存するだけ。

## 完了条件と次工程
1. ボードの印刷実寸・剛性確認。
2. 640×480の実際のSide source screenshotで十分な数のコーナーを検出。
3. 画像別RMS・保持画像誤差・カメラ行列・歪みの妥当性を評価。
4. 旧16枚の結果と、画像幾何の同一性が説明できる条件で比較。
5. URDF角度ゼロ点・符号／ID2固定剛性を別途確認した後、ID0/1/2のPnPとEye-to-Hand推定へ。

[関連：ArUco収録／FK／重複監査](so101_aruco_calibration_capture.md) ／ [Draft PR #9](https://github.com/neogim/so101-imitation-learning/pull/9)


## 一人で撮影する方法：自動タイマー（2026-11-07追加）

ターミナルで1枚ずつEnterを押さなくてよい。OBSを起動し、印刷ボードを持ち、
**撮影開始まで8秒、以後3秒ごとに25枚**を自動保存する。
ボードは各シャッター時点で静止させる。連続移動するとモーションブラーで検出精度が落ちる。

まず **YAMLの実在場所を検索**：

```bash
find . -type f -name 'cameras.aruco-640.local.yaml' -print
```

発見したパスを正しく指定する。例えば `./scripts/so101_capture/cameras.aruco-640.local.yaml`
と表示された場合は次のコマンド（存在しなければ推測して使わない）：

```bash
CONFIG="$(find . -type f -name 'cameras.aruco-640.local.yaml' -print -quit)"
test -n "$CONFIG" && test -f "$CONFIG" || { echo 'ERROR: YAML not found'; exit 1; }
PYTHONPATH="$PWD/.local/camera-deps${PYTHONPATH:+:$PYTHONPATH}" \
python scripts/so101_capture/capture_side_charuco_stills.py \
  --camera-config "$CONFIG" \
  --out-dir outputs/so101-charuco-stills-timed-01 \
  --frames 25 --start-delay-seconds 8 --interval-seconds 3 --execute
```

このブロックの `exit 1` はシェルスクリプトのときに安全停止する。対話シェルで直接貼る場合は `CONFIG` の検出結果を確認してから実行すること。
自動撮影中はロボット／Follower／Leaderを接続・操作しない。途中停止はCtrl-C（その時点の画像は保存される）。出力先が存在すると実行拒否するので、新しい名前を指定する。

**この方式は動画ファイル保存ではなくOBS連続静止画取得**。連続録画して後からフレームを選ぶ方式も可能だが、今回のコードでは所定間隔の静止画を直接保存する。必要なら画面内のボード検出率を確認して追加収集する。
