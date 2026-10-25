# 承認済み依存の導入・実機なしの記録検証

2026-10-18。サーボへの接続・駆動、実機データセット収集は行っていない。
LeRobot本体とキャリブレーションファイルは変更していない。外部へのデータ送信なし。

## 環境

- 導入前に `pip freeze` を `dependency_plan/validation/pip-freeze-before.txt` に保存。
- 固定requirementsとconstraintsで再dry-run。21件が承認計画と完全一致したことをプログラムで照合。
- 承認コマンドでインストール成功。導入後の全distribution照合でも追加20件、変更はfsspec 2026.7.0→2026.2.0のみ。
- torch、torchvision、OpenCV、numpy、Feetech SDK、pyserial、deepdiff、LeRobotの版は維持。
- 導入後freezeは `dependency_plan/validation/pip-freeze-after.txt`。
- `python -m pip check`: exit 0、No broken requirements found。
- `lerobot-teleoperate --help`: exit 0。
- `lerobot-record --help`: exit 0。出力はvalidation配下に保存。

## カメラアダプター

- `test_obs_camera.py`: 3テスト成功（RGB、寸法/チャンネル拒否、接続時実画像検査）。
- `check_obs_modes.py --with-pyav`: index 1だけにアクセス。OpenCV+PyAV同時import状態で実施。
- OBS設定640×480/30、OpenCV申告1920×1080/60、実画像RGB 480×640×3。
- アダプターで90フレーム成功、約30.11 FPS。
- 標準カメラ処理の比較試験は既知の幅不一致で失敗。アダプター成功と区別する。
- 実測JSON: `/tmp/so101-obs-after-install/results.json`。

## 合成データの往復

`test_synthetic_dataset.py` はカメラ・ロボットドライバを使わず、赤/青の背景と動く緑マーカーのRGB画像、
既知の数式による6軸state/actionを200フレーム生成する。offline環境変数を設定しHubにはアクセスしない。

```bash
conda activate lerobot
python scripts/so101_capture/test_synthetic_dataset.py \
  --output /tmp/so101-synthetic-next
```

出力先は未作成の場所を指定。成功した出力先は `/tmp/so101-synthetic-stage3-verified`。

確認結果:
- 1エピソード、200フレーム、20 FPS（名目10秒）。
- Parquet 200行、MP4 H264 640×480、デコード200枚。
- `observation.state`と`action`は全200行で期待値と一致。グリッパを含む6軸。
- 画像はLeRobot読出時CHW float、赤/青のチャンネルと画素位置を検査。保存時HWC uint8。
- frame_index=0..199、episode_index=0、timestamp=0..9.95秒。
- MP4のPTSも0.05秒刻みで一致。
- `verify_dataset.py`を子プロセスで実行しexit 0、errors=[]、200フレームを確認。
- `verification/viewer.html` を生成。ブラウザGUIによる操作確認は未実施。
- 異常値/欠損のない合成データで成功。実機の追従精度や無線遅延を検証したものではない。

要約は `dependency_plan/validation/synthetic-roundtrip.json` と `synthetic-verification.json` に保存。

## 発生した問題と扱い

1. 最初の合成試験はMP4/Parquet保存後、既存HFキャッシュにlockファイルを書く際にサンドボックス権限で失敗。
   依存エラーではない。同じスクリプトを許可済みサンドボックス外・オフラインで再実行し成功。
   追加インストールやキャッシュ削除はしていない。
2. OpenCVとPyAVのwheelがそれぞれ同梱する `libavdevice.61.3.100.dylib` から、
   `AVFFrameReceiver` / `AVFAudioReceiver` のObjective-Cクラス重複警告が出る。
   両CLIのhelp、同時importしたカメラ取得、合成データ保存・読出は成功したが、警告そのものは未解消。
   ライブラリ削除・リネームや追加インストールはしていない。
   今後この重複が原因と疑われるクラッシュが出る場合は、カメラ取得と動画処理を別プロセスに分離する案を優先する。
   実機前には必要に応じ、両ライブラリを使う記録ランチャーのダミーデバイス試験で確認する。
3. PyAV読出で `No accelerated colorspace conversion found from yuv420p to rgb24` が出る。
   読出・RGB検査は成功。今回の成功だけから実機の20 Hz制御性能を保証しない。

## 残作業

依存復旧と依頼された合成試験は完了。実機は未接続。
既報のとおり、標準record_loopは送信前Actionを保存するので、実送信指令の戻り値を保存する独立ランチャー、
較正不一致時の停止保護、USBポート事前チェックを完成・検証してから実機承認を求める。
現在の標準recordコマンドをそのまま実機用の完成版とは扱わない。
