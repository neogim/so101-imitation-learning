> 現行の複数エピソード収録・安全終了手順は [MULTI_EPISODE.md](MULTI_EPISODE.md) を参照。以下は過去の調査・単発記録の記録です。

# SO-101収集環境の調査（2026-10-18）

**最新状況:** 独立ランチャーと実機なしの統合試験が完了。[実機記録手順](RECORDING.md)を参照。
実機操作は承認待ち。承認済み依存の導入、両CLIのhelp、合成200フレームの保存・読出・検証に成功。
[導入後の検証結果](STAGE3.md)を参照。[第2段階の変更計画](STAGE2.md)は計画時点の記録。
以下は初回調査の記録。

実機の接続・サーボ駆動・記録は未実施。LeRobot本体、Conda環境、キャリブレーションは変更していない。

## 確認結果

- Python: Conda環境 `lerobot` の `python`（検証時は3.12.13）。
- LeRobot: メタデータ0.6.2、editable install先 `~/workspace/lerobot`。
- Git: `v0.6.1-29-g223a8ad1`。調査開始時の作業ツリーはclean。リリースタグ0.6.2と同一という意味ではない。
- OpenCV 4.13.0、torch 2.11.0、torchvision 0.26.0、numpy 2.2.6。
- `lerobot-record --help` は `ImportError: 'datasets' is required but not installed` で停止。
- datasets / pandas / pyarrow / av が未導入。pynputもないが、この版にはTTY入力の代替処理があり、それだけを致命的エラーとはしない。
- ローカルpyprojectのdataset extra: datasets >=4.8,<5、pandas >=2,<3、pyarrow >=21,<30、av >=15,<16、ほかプラットフォーム別依存。依存解決・インストールは未実施。環境再構築は不要。導入前に現状を保存し、既存パッケージを固定したdry-runで変更一覧を確認する。
- Follower指定ポートは存在。Leader `/dev/tty.usbmodem5B7B0185961` は調査時点で見えない。
- 既存キャリブレーションは `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower_arm.json` と `teleoperators/so_leader/my_leader_arm.json` に存在。内容変更なし。

## カメラの確認

`lerobot-find-cameras --help` は成功し、`opencv`、`--output-dir`、`--record-time-s`、`--warmup-s` を確認。
ただし、このコマンドは全カメラを順に開く。iPhone 15の連係カメラにもアクセスし得るので、今後は既知のOBSインデックスだけを開く。

最初はカメラ権限エラー。許可後は3台が列挙された。インデックス2はOBSとは異なる映像だったため対象外。
OBSロゴで同定したインデックス1を単独確認し、仮想カメラ開始後にユーザー提示と同じ紫の布・机を取得した。

| OBS index 1 | 結果 |
|---|---|
| Backend | AVFoundation (1200) |
| OpenCV申告値 | 1920×1080 / 60 FPS |
| 実画像 | 640×480、3チャンネル |
| 5秒の受信 | 151フレーム、約29.94 FPS |
| 向き | OBSプレビューと一致。左右反転の確定には文字・左右マーカーを撮影する |

画像は `/tmp/so101-obs-only-1/first.png`, `last.png`、測定は `probe.json`。
これはホスト受信頻度であり、iPhoneセンサの露光時刻・実効更新頻度・Wi-Fi遅延ではない。
再接続後はインデックスが変わり得る。全探索せず、OBSと同定できた対象を使用する。

```bash
conda activate lerobot
# このREADMEがあるディレクトリで実行。出力先は毎回未作成の場所にする。
python probe_obs_raw.py --index 1 --output /tmp/so101-obs-recheck
```

**現状の阻害要因:** LeRobot OpenCVCameraで640×480を設定すると、CAP_PROPが1920×1080を返し接続失敗。
デフォルト接続でも実画像の寸法と申告値が合わず読み取りスレッドが失敗する。
OBS出力と仮想カメラのプロパティを一致させて再検証するか、実フレームを基準に扱う独立Camera実装が必要。
既存LeRobotの検査を無条件に除去する変更はしない。
一致を解消した後の本体経路確認:

```bash
python probe_camera.py 1 --width 640 --height 480 --camera-fps 30 --output /tmp/so101-obs-validated
```

現状では上の本体経路確認は失敗することを実測済み。カメラ30 FPSとデータセット20 FPSは別設定。

## 記録コマンド案（未実行・現状では実行不可）

以下のキーはローカル設定クラスで存在確認した。ただし記録CLIのhelpとパーサ通過は依存不足で確認できていない。
「現在の環境で動作確認済みコマンド」ではない。依存・カメラ寸法・Leader接続を解決し、実機動作についてユーザー確認を得てから実行する。
カメラ値は上の検証に成功した値に合わせる。

```bash
conda activate lerobot
cd ~/workspace/lerobot
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B7B0098201 \
  --robot.id=my_follower_arm \
  --robot.use_degrees=true \
  --robot.cameras='{iphone: {type: opencv, index_or_path: 1, backend: 1200, width: 640, height: 480, fps: 30, color_mode: rgb, rotation: 0}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5B7B0185961 \
  --teleop.id=my_leader_arm \
  --teleop.use_degrees=true \
  --dataset.repo_id=local/so101_iphone_test \
  --dataset.root="$HOME/workspace/so101-data/iphone-test-001" \
  --dataset.no_stamp=true \
  --dataset.single_task='Move the gripper slowly toward the object and return.' \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=10 \
  --dataset.reset_time_s=0 \
  --dataset.fps=20 \
  --dataset.video=true \
  --dataset.rgb_encoder.vcodec=h264 \
  --dataset.push_to_hub=false \
  --display_data=false \
  --play_sounds=false
```

保存先は未作成の新規ディレクトリにする。既存データへresumeしない。H264エンコーダはav導入後、実機なしの合成画像で書込・読出確認する。
標準connectはキャリブレーション不一致時にcalibrateを呼び得る。確認画面が出たら再較正・書込みに進まず中止。
本番前に独立ランチャーでcalibrateを例外停止に置き換える保護を入れると、ファイルやモータの較正値変更を防げる（未実装）。
接続時はトルクやPID設定も扱うので、単なる接続テストでもユーザーの実機承認後に行う。
20 FPS×10秒は約200フレームの目安。ループ遅延で減ることがあるのでCycleTimerの実測ログを保存する。

## 保存形式と同期の限界

| Feature | 形式 | 意味 |
|---|---|---|
| observation.images.iphone | RGB uint8 H×W×3を動画化 | カメラ画像 |
| observation.state | float32[6] | Follower Present_Position |
| action | float32[6] | Leader位置から作った目標関節位置 |
| timestamp | 秒 | frame_index / dataset.fps（実測でない） |
| frame_index / episode_index / index / task_index | 整数 | フレーム・エピソード・全体行・タスクの参照 |

6要素の順序は shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos。
use_degrees=trueでは先頭5軸が度、グリッパは常に0..100の正規化値。falseなら先頭5軸は-100..100。
単位は数値だけで推測せず、この記録設定をデータセットと一緒に保管する。ACTの学習時の統計正規化とは別。
速度・トルクは標準のこのObservationにはない。

処理順は、Follower関節読取 → 最新カメラフレーム取得 → Leader読取 → Action送信 → add_frame。
画像・state・actionは同じ行にまとまるが、同時露光・同時測定ではない。画像はバックグラウンド取得した最新値で、同じ画像を再利用する場合がある。
`read_latest`はホスト側フレームが500 msより古いと停止するが、Wi-Fi経由で届く古い内容の検知にはならない。

重要: record_loopのコメントと実コードに差があり、`_sent_action`は保存されず、`action_values`（teleop processor後）が保存される。
標準identity処理かつmax_relative_targetなしでは同じ目標になる。クリップや独自processor導入時は実送信値との差を別途記録する。
モータ内部の量子化・機械的な追従誤差までActionが表すわけではない。

v3形式: meta/info.json、タスク・エピソードのメタデータ、data配下のParquet、videos配下のMP4。
LeRobotDatasetで読み出した画像は通常CHW float [0,1]。保存時のHWC uint8との違いに注意。

## 検証と可視化

```bash
python verify_dataset.py "$HOME/workspace/so101-data/iphone-test-001" \
  --output /tmp/so101-validation-001 --body-unit degrees
```

全フレームの画像デコード、Feature/関節順、NaN/Inf、6軸形状、連続frame_index、名目timestamp、
エピソード数・フレーム数・範囲・グリッパ・同一連続画像を検査。
report.jsonと、画像・状態・Actionを同じフレームのスライダーで見られるviewer.htmlをローカル生成する。
CDNや外部送信なし。10秒の短い試験専用で、長時間データはHTMLが大きくなる。
範囲外や同一画像は警告であり、物理的異常・停止画像を断定しない。急な動作と追従誤差はviewerで確認する。
実際の周期・露光時刻・各モダリティのずれは標準データだけでは検証不可。
この検証スクリプトは構文確認済み。依存不足と未収集のため実データでの試験は未実施。

## ROS2への発展案（未実装）

| Topic | メッセージ案 | 時刻・単位 |
|---|---|---|
| /camera/image_raw | sensor_msgs/msg/Image、rgb8 | 光学frame_id。可能なら露光時刻。現経路ではホスト受信時刻を代用と明記 |
| /joint_states | sensor_msgs/msg/JointState | 関節読取時刻。5回転軸は度→rad。速度/effortは測定しないので空 |
| /so101/leader_command | 独自Stampedメッセージ | Header、cycle_id、関節名、要求/実送信値、単位、グリッパ正規化値、送信時刻 |

JointStateのpositionはradまたはmなので、グリッパ0..100をそのまま混ぜない。校正済み変換でrad/mにできるまでは別の正規化値Topic/独自メッセージに保存する。
全6軸の元のLeRobot値も独自メッセージで保持すれば、後から忠実に変換できる。
根拠: [JointState定義](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/JointState.msg)。

同じ取得ループからpublishとdataset記録へ分岐し、シリアルポートを別ノードで二重読取しない。
cycle_id、画像受信時刻、関節読取開始/終了、Leader読取、送信開始/終了を保存する。
周期計測はmonotonic、ROS Headerは共通のROS clockに統一し、両者の対応点も記録する。
macOSとROS2側が別ホストなら時計同期とネットワーク遅延を測定。全モダリティに同一時刻を機械的に上書きしてずれを隠さない。

rosbag2には元メッセージとepisode開始/終了・task・設定を残す。bag受信時刻とHeader取得時刻を区別する。
変換時は20 Hzの基準時刻へリサンプルし、過去の最新画像を年齢上限付きで選択。stateは測定値の補間を検討し、Actionは送信済み指令のゼロ次保持。
cycle_idでの対応付けを優先。近似時刻同期だけでWi-Fi遅延は除去できない。
許容差・欠落・補間・重複のルールと件数を保存。QoSはpublisher/subscriber/recorderで整合させる。
[message_filters公式説明](https://github.com/ros2/message_filters/blob/rolling/doc/index.rst)、[rosbag2公式説明](https://github.com/ros2/rosbag2/blob/rolling/README.md)。

## ローカル実装の確認箇所

`~/workspace/lerobot/src/lerobot/` 以下:
- configs/dataset.py: DatasetRecordConfig、configs/video.py: RGBEncoderConfig
- scripts/lerobot_record.py: RecordConfig、record_loop、record
- robots/so_follower/config_so_follower.py、so_follower.py
- teleoperators/so_leader/config_so_leader.py、so_leader.py
- cameras/opencv/configuration_opencv.py、camera_opencv.py
- cameras/configs.py: backend=1200、rotation=0/90/180/-90
- datasets/dataset_writer.py: timestamp生成
- utils/feature_utils.py: float32配列化、関節順

次の作業は依存導入の変更計画、OBSプロパティ不一致解消、実機なしの記録/デコード確認、較正保護ランチャー。
その結果を確認してからユーザー承認のもとで1エピソード実機収集する。
