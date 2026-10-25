> 現行の複数エピソード収録・安全終了手順は [MULTI_EPISODE.md](MULTI_EPISODE.md) を参照。以下は過去の調査・単発記録の記録です。

# SO-101の最初の実機記録

2026-10-18。ランチャーは準備済み。サーボ接続・駆動・実機記録はまだ実施していない。

## 後続更新：開始時の自動位置合わせ

ユーザーの依頼により、record_episode.pyとteleop_only.pyの共通start_hardwareに低速位置合わせを追加。
以下の初回手順にある「15度差で停止」は旧仕様。現在は較正一致確認後、初期差が本体120度・
グリッパ100単位を超えない範囲で、目標値を最大0.15/周期（20Hz、最大3度または正規化単位/秒）で変化させる。
Leaderは開始合図まで静止させる。Leader移動3超、追従誤差8超、進捗停止3秒、全体60秒で中止。
一致を確認後、トルクと接続を維持したままテレオペまたは記録待機へ移行する。較正を書き換える処理ではない。
8件の実機なしテスト成功。最初の実機試行はFollower ID 2へのTorque_Enable=0書込みで
「There is no status packet」となり、位置合わせ開始前に中止。終了処理は追加エラーなく完了。
通信・電源状態を確認するまで再試行しない。

## 起動前の操作

1. Leader/FollowerをそれぞれUSB接続し、既存のテレオペ時と同じ電源・配線を確認。
   現在の読み取り専用チェックではFollowerだけ存在し、Leaderの指定ポートがない。
2. `ls /dev/tty.usbmodem*` で次の2ポートを確認する。このコマンドはポートを開かない。
   - Follower: `/dev/tty.usbmodem5B7B0098201`
   - Leader: `/dev/tty.usbmodem5B7B0185961`
   別名で現れた場合、勝手に別デバイスへ切り替えず対応を確認してからランチャーの定数を修正する。
3. iPhone 6 PlusのDroidCam映像をOBSに表示し、「仮想カメラ開始」を押す。
   OBSは640×480/30 FPS、Program出力。確認済みindex 1だけを使う。他のカメラ探索は行わない。
4. 腕を安定して置き、作業域とケーブルを確保する。Leader/Followerの初期姿勢を近づける。
   接続後に各回転軸の差15度以内・グリッパ差20以内を検査し、超えれば駆動前に中止する。
   力の入ったサーボを無理に手で動かさない。姿勢を揃える際は安全に電源を管理する。
5. 手元のターミナルで実行する。実行承認後、設定処理が終わるとすぐ10秒の記録が始まる。
   終了時はトルクを切るので、腕が倒れたり落ちたりしない状態にする。

## 実行コマンド

リポジトリルートで実行。新しい保存先は毎回生成し、既存ディレクトリは拒否する。

```bash
conda activate lerobot
cd /path/to/so101-imitation-learning
mkdir -p data/so101
SO101_RUN_ROOT="$PWD/data/so101/episode-$(date +%Y%m%d-%H%M%S)"

# 起動前チェックのみ：USBの存在・較正ファイル・保存先・OBS画像。
# シリアルポートを開かず、サーボ操作をしない。
python scripts/so101_capture/record_episode.py --root "$SO101_RUN_ROOT"

# 以下はユーザーの実機操作承認後に限って実行する。
python scripts/so101_capture/record_episode.py --execute --root "$SO101_RUN_ROOT"

# 正常終了後、別の新規ディレクトリへ検証結果を出力する。
python scripts/so101_capture/verify_dataset.py "$SO101_RUN_ROOT" \
  --output "${SO101_RUN_ROOT}-verification" --body-unit degrees
```

10秒・20 FPS・1エピソード、カメラindex 1・RGB 640×480に固定。
実画像とFeatureのサイズが一致しない場合は停止。LeRobot 0.6.2と独立OBSCameraを使用。
`--execute`がなければシリアル接続に進まない。`--simulate`では実カメラもポートも開かない。
Hubのofflineとtelemetry無効を設定し、push処理自体を実装していない。
`--execute`は操作承認の代用ではなく、誤起動を防ぐための追加条件。

## キャリブレーション・サーボ保護

- LeRobotの標準connect()を呼ばず、両バスのread-onlyハンドシェイクと既存is_calibrated判定を先に行う。
- 両方の較正が一致するまでconfigure()は呼ばない。不一致ならモータ設定・Goal_Position送信をしない。
- このプロセス内のデバイスインスタンスでcalibrate、_save_calibration、bus.write_calibrationを例外停止に置換。
  元のLeRobotソースやクラス全体は変更しない。較正ファイルのSHA256を前後比較する。
- 初期姿勢の差を検査した後、Followerのトルクを無効化し、現在位置を目標値にしてから標準configure()を行う。
  標準configureはモード・PID等を設定しトルクを有効にするので、この段階も実機承認が必要。
- 本体5軸はuse_degrees=true（度）。gripperは0〜100正規化値。
- NaN/Infの状態・要求指令は送信前に拒否。
- 通常終了・Ctrl-C・SIGTERM・例外時は切断を試みる。設定後はトルク無効化を試み、シリアルを閉じる。
  較正/姿勢不一致でconfigure前に中止する場合はトルク状態を変更せずポートだけ閉じる。
- Ctrl-Cは通信処理が戻ってから処理される場合がある。電源断・強制終了・USB障害時のソフト停止は保証できない。
- 中断時のディレクトリは未完成データとして残る。上書きや自動再開はせず新規rootでやり直す。

## Actionと同期

- `observation.state`: Followerの指令送信前の実測位置、float32[6]。
- `observation.images.iphone`: 同じループで取得した最新画像。
- `action_requested`: Leaderのget_action()、float32[6]。
- `action`: Followerのsend_action()の戻り値、float32[6]。

actionは実送信に使った目標値を表し、モータが到達した位置ではない。
サーボ内部量子化後のレジスタ読戻しを保証するものでもない。
現ランチャーではmax_relative_targetや独自変換を設定しないため、通常actionとaction_requestedは同じになる。
検証では差を意図的に作り、混同せず保存されることを確認した。

dataset.timestampは従来どおりframe_index/20。実際のループ開始・処理区間時間はcapture_timing.jsonに別途保存。
画像の露光時刻・Wi-Fi遅延まではこのログから分からない。
動画エンコードは10秒の制御終了後、サーボを切断してから行う。

## 実機なしの統合試験

`record_episode.py --simulate`は同じrecord_loop、LeRobotDataset、OpenCV RGB処理、PyAV H264を使用する。
ダミーRobotは送信値を要求値の半分にする。カメラ・シリアルは開かない。

最終テスト `/tmp/so101-launcher-integration-20hz`:

| 指標 | 実測 |
|---|---:|
| エピソード/フレーム | 1 / 200 |
| 制御ループ経過 | 10.0215秒 |
| 実効FPS | 19.9571 |
| 1周期の処理時間（待ち時間除外）平均 | 1.0385 ms |
| 処理時間p95 / 最大 | 1.7441 / 9.4062 ms |
| 処理時間50 ms超 | 0 |

通常sleepで約18.7 HzだったためLeRobot既存のprecise_sleepを使用。
実効FPSは待ち時間込みで測定。実機シリアルの遅延は含まれず、実機で20 Hz達成を保証するものではない。

`check_recorded_actions.py`で全200行のstateが前周期の送信結果、actionが要求の半分であることを検査。
Parquet、H264 MP4 640×480/200枚、動画PTSとdataset timestampの0.05秒刻みを確認。
既存verify_dataset.pyも終了コード0、errors=[]、200フレームで成功。
検証ビューアーは `/tmp/so101-launcher-verification-20hz/viewer.html`。
較正不一致、初期姿勢不一致、較正操作禁止、既存root拒否の4テストが成功。

OpenCVとPyAVのObjective-Cクラス重複警告は残るが、今回の同一プロセス統合試験ではクラッシュなし。
ライブラリの追加・更新・削除はしていない。

## 残っている項目

- LeaderのUSBポートが現在見えない。接続・ポート名確認が必要。
- 実機較正一致、初期姿勢、実機制御周期はまだ未確認。承認後の起動時に検査する。
- OpenCV/PyAVの警告は未解消。ダミー試験成功だけで長時間安定性は断定しない。
- ROS2/ACTは今回の対象外。まずこの1エピソードを収集・検証する。
