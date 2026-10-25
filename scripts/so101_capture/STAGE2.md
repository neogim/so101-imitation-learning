# 第2段階：カメラ解決結果と依存導入計画

2026-10-18。インストール・サーボ接続・キャリブレーション変更は未実施。

## カメラ解決結果

OBS basic.iniと稼働ログで、基本解像度640×480・出力640×480・30 FPS・仮想カメラProgram出力を確認。
OBS側はすでに目的の設定で、設定ファイルを書き換える必要はなかった。
[OBS公式の出力モード説明](https://obsproject.com/kb/virtual-camera-guide)。

インデックス1だけをAVFoundationで開き、以下を実測した。

| 経路 | 要求設定 | OpenCV申告 | 実画像 | 受信FPS |
|---|---|---|---|---|
| OpenCV設定なし | OBS側640×480/30 | 1920×1080/60 | 640×480 | 30.01 |
| OpenCV set | 640×480/30 | 1920×1080/60 | 640×480 | 30.03 |
| OpenCV set | 640×480/20 | 1920×1080/60 | 640×480 | 30.00 |
| LeRobot標準 | 640×480/30 | 1920×1080/60 | 接続失敗 | — |
| 独立アダプター | 640×480/30 | 1920×1080/60 | RGB 480×640×3 | 30.13（90フレーム） |

set()はtrueを返すが、get()は変わらない。FPSを20に要求しても実受信は約30のまま。
LeRobot標準の失敗箇所は `camera_opencv.py:297` の `_validate_width_and_height()`。
寸法を省略すると申告値を採用し、後段 `_postprocess_image()` の実画像寸法検査で失敗する。
FPSも `_validate_fps()` で要求値と申告値を厳密比較している。
根本原因の範囲はOBS仮想カメラとAVFoundation/OpenCV間のプロパティ報告。
どのコンポーネントの不具合かまでは、この計測だけで断定しない。

`obs_camera.py` のOBSCameraは標準OpenCVCameraを継承し、接続時のプロパティ交渉だけを置換する。
index=1・backend=1200・RGB・640×480・rotation=0を固定し、実画像が一致しなければ停止。
以後のフレームも親クラスの寸法・チャンネル検査と500 msの鮮度検査を使用する。
リサイズ、切り抜き、追加反転はしない。OBS内で既にrotation=180の変換があるため二重回転しない。
fps=30は期待値であり、プロパティ申告値の保証ではない。今回の実測値はホスト受信周期であり露光周期ではない。

証拠: `/tmp/so101-obs-stage2/results.json`, `adapter.png`。
画像はユーザー提示と同じ紫の布・机であることを確認。index 0/2にはアクセスしていない。
接続後90フレーム成功。RGB変換、異なる解像度・チャンネルの拒否、接続時の実画像検査の3テストも成功。
アダプターを実機記録ランチャーへ組み込む作業は依存導入後に行う。

## 依存変更計画（未実行）

現状 `pip check` は成功。ただしoptionalなdataset extraは未導入なので、記録CLIは起動しない。
全既存パッケージを固定したdry-runではdatasets 4.8.0〜4.8.5とfsspec 2026.7.0が競合。
datasetsが要求するfsspec上限は2026.2.0。fsspecのみ変更を許可したdry-runは成功した。

主要追加:

| パッケージ | 固定版 | 用途 |
|---|---|---|
| datasets | 4.8.5 | LeRobotDatasetの行データ |
| pandas | 2.3.3 | メタデータ処理 |
| pyarrow | 26.0.0 | Parquet |
| av | 15.1.0 | MP4エンコード・デコード |
| jsonlines | 4.0.0 | dataset関連処理の依存 |

間接追加15個（全て固定）:
aiohappyeyeballs 2.7.1、aiohttp 3.14.5、aiosignal 1.4.0、attrs 26.1.0、dill 0.4.1、
frozenlist 1.8.0、multidict 7.1.0、multiprocess 0.70.19、propcache 0.5.4、
python-dateutil 2.9.0.post0、pytz 2026.5、six 1.17.0、tzdata 2026.5、xxhash 4.0.1、yarl 1.25.1。

既存変更は **fsspec 2026.7.0 → 2026.2.0の1個のみ**。
fsspecはファイルシステムアクセス層なので、他の処理で新しい版固有の機能を使っていれば影響し得る。
既存torchの要件 `>=0.8.5` とhuggingface_hubの要件 `>=2023.5.0` は満たす。
torch 2.11.0、torchvision 0.26.0、OpenCV 4.13.0.92、numpy 2.2.6、
feetech-servo-sdk 1.0.0、pyserial 3.5、deepdiff 8.6.2を含む他の既存版はconstraintsで固定。

`lerobot[core_scripts]`全体は導入しない。表示無効・PyAV指定のローカル記録/検証を対象にする。
torchcodecは公式dataset extraに含まれるが、ローカル実装のPyAV経路では必須でないため今回追加しない。
rerun/foxglove/pynputも追加しない。TTY入力の代替機能を使う。
これは実行経路を絞った導入案であり、全extraの導入ではない。help/importの最終確認は導入後に必要。

承認後のコマンド（リポジトリルートで実行）:

```bash
python -m pip install \
  --only-binary=:all: \
  -c scripts/so101_capture/dependency_plan/proposed-constraints.txt \
  -r scripts/so101_capture/dependency_plan/requirements.txt
```

全21件（新規20＋変更1）の固定版一覧はrequirements.txt、導入前全版はexisting-constraints.txtに保存。
承認後もまず同じ固定ファイルでdry-runを再確認し、追加変更が生じた場合は止める。
環境全体の一括更新、LeRobot再インストール、git pullは行わない。

## 導入後に行う検証・記録準備

1. `pip check`、`lerobot-record --help`、torch/OpenCV/サーボ関連版が維持されたことを確認。
2. 合成RGB画像・ダミー6軸state/actionで20 FPSのLeRobotDatasetを作成し、H264/Parquetの往復とverify_dataset.pyを確認。ハードウェアには接続しない。
3. 独立ランチャーでFollowerのcamerasにOBSCameraを設定。画像Featureは `observation.images.iphone`、640×480で既存検証スクリプトと一致させる。
4. 記録ループは `sent_action = robot.send_action(requested_action)` の戻り値を `action` として保存する。標準record_loopの送信前action_values保存は採用しない。
5. 要求Actionと実送信Action、各取得/送信のmonotonic時刻は別ログにも残す。返り値は指令値であって、実測追従位置・サーボ内部量子化後のレジスタ読戻しではない。
6. Follower/Leaderのcalibrateを独立サブクラスで例外停止にする。単に `connect(calibrate=False)` で不整合を無視しない。保存済み較正と一致しなければ停止。
7. `ls /dev/tty.usbmodem*`で両ポート存在を確認。これはポートを開かない。シリアル接続自体はユーザーの実機承認後。

予定ポート: Follower `/dev/tty.usbmodem5B7B0098201`、Leader `/dev/tty.usbmodem5B7B0185961`。
双方use_degrees=true: 本体5軸は度、gripperは正規化0..100。
記録条件は20 FPS・10秒・1エピソード・新規ローカルroot・Hub offline/uploadなし。
完成したランチャーと実行コマンドは、依存導入と合成データ検証の成功後に提示する。現段階で実機記録成功とは扱わない。
