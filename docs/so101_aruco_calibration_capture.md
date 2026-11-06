# SO-101 Side ArUco: 校正用の静止姿勢データ収集

## 方針と安全性

**既存の `capture_session.py` の20Hz Leader→Followerテレオペ／安全機構を使う。**
校正専用の制御経路やFollower USBへの別プロセス接続を作らない。
収集したデータセットを**後から** `extract_aruco_calibration.py` で解析し、
静止区間のSide画像と**同じフレーム**のFollower実測関節角度を抽出する。
これによりテレオペ制御ループへOpenCV検出処理を追加せずに済む。

**まだ実機動作確認はしていない。**
スクリプトはオフラインの抽出器であり、Eye-to-Handの6DoF外部パラメータを
推定するソルバーではない。

## 準備

- 対象：Followerに装着した `DICT_4X4_50` のID 0〜5。
- マーカー実寸：**黒い外枠の一辺14mm**（印刷全体18mm、白枠は各2mm）。
- **iPhone Sideカメラ・OBSのズーム／クロップ・ロボットベースを固定する。**
- `cameras.local.yaml` の `capture.width/height` は既存環境で320×240。
  校正用の最初の試験は640×480にする場合、**既存ファイルを上書きせず**
  専用のYAMLコピーに変更する。既存カメラ処理は320×240、448×336、640×480を許容する。
  1280×960は現実装で未対応。640×480にアップスケールするだけでは精度向上しない。
- OBS Side映像の撮影系が、チェッカーボード16枚の内部校正と同じ画像幾何か確認する。
  640×480で得た暫定内部パラメータは、640×480の同じ画像座標系でのみ直接使える。
- 各マーカーの**所属リンク名**を別途確定する。異なるリンクに貼ったIDをひとつの
  静止剛体ボードとして処理してはいけない。

## 実機収集：既存コマンドのみ

1. Macで既存のLeRobot/Capture依存関係、OBS、FollowerとLeaderを接続。
2. 別のテレオペ処理や `observe_aruco_follower_readonly.py` は起動しない。
3. 新規出力ディレクトリ名を用意。先に**ハードウェア非駆動のpreflight**を実行。

```bash
python scripts/so101_capture/capture_session.py \
  --root data/so101/aruco-calibration-20261107 \
  --camera-config cameras.aruco-640.local.yaml \
  --enable-cameras side \
  --seconds 120 \
  --num-episodes 1 \
  --task "Observe stationary ArUco poses for camera calibration"
```

4. **実機実行を明示的に許可できる状態でのみ**同じコマンドに `--execute` を付ける。
   実機を動かす前にLeader/Followerの初期姿勢と周辺干渉を確認する。
5. 既存のコンソール操作： `t` はテレオペ、`r` は安全確認とカウントダウン後に
   エピソード録画開始、`e` は録画終了、`h` は保持モード、
   `q` は `SAFE_RESET` から終了。**録画を開始しなければ抽出データは残らない**。
6. 録画中に約20〜30の異なる姿勢へゆっくり動かし、各姿勢で
   **1〜2秒静止**。上下・左右・前後・手首角を安全な可動範囲で分散。
   20Hzなら各ホールドは20〜40フレーム。120秒は最大録画時間の例であり、
   1回で足りなければ新規rootに分けて追加収集する。
7. 動作終了後、`e`、`h`、SAFE_RESETで `q` の順を守る。
   異常・引っ掛かり・熱・ケーブル張力があれば中止。
   **終了時のトルクは自動でOFFにならない**ことに注意。

## 抽出（実機・USB・カメラ不要）

```bash
python scripts/so101_capture/extract_aruco_calibration.py \
  --dataset data/so101/aruco-calibration-20261107 \
  --out-dir outputs/so101-aruco-calibration-20261107 \
  --window 12 --tolerance-deg 0.7 --distinct-deg 3 \
  --min-edge-px 8 --max-poses 30
```

出力：
- `manifest.json`: データセット名、各サンプルのepisode/frame/timestamp、
  実測6関節（J6のみ正規化値）、ArUco ID・四隅の画素座標・最小辺長、
  黒枠14mm、警告。
- `images/*.png`: 静止候補のSide画像（映像の再圧縮フレーム）。

実行ごとに**新規の出力フォルダ**を指定する。既存データを変更しない。
動画は既存LeRobotのMP4であり、照明やH.264の圧縮で小マーカーの
コーナー精度は変動する。結果を確認してから必要に応じて高画質撮影を検討する。

## 校正を始める前に確認すべきこと

1. 検出IDと各リンクの対応（J5で動くもの／動かないもの）。
2. 各ポーズで実測関節角度とArUco検出が同期しているか（**静止**でずれを抑える）。
3. 640×480画像のintrinsicsとの一致。映像系のクロップ／リサイズ変化がないか。
4. URDF/FKのJ2/J3符号、ゼロオフセット、およびマーカーのリンクへの剛体取り付け。
5. **マーカー→リンク変換が未知**なので、複数姿勢に基づくEye-to-Handソルバーの
   実装と未使用姿勢での再投影評価が別途必要。5枚見えても自動的に外部校正成功ではない。

## テスト

```bash
python -m unittest discover -s scripts/so101_capture -p 'test_extract_aruco_calibration.py'
```

テストは数値配列による静止姿勢判定のみで、モーター・カメラにはアクセスしない。
