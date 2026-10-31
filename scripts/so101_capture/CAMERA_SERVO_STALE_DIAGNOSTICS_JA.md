# SO-101 Macカメラ選択・six-axis snapshot遅延 調査報告

## 結論

今回の `--camera-config ... --enable-cameras side,wrist` 経路では、SideはOBSの
`source_name + source_uuid + expected_settings(cur_dev_ip)`、WristはAVFoundationの
Unique IDで選択される。Wristの `AVCaptureDeviceInput` はUnique ID一致後の1台だけであり、
コード上、FaceTime HDを入力へ追加するフォールバックはない。

ただし、従来はSwift helperのstderrを `/dev/null` に捨てていたため、実行時に選択された
`localizedName/uniqueID/deviceType` をログから証明できなかった。今回、列挙候補、選択、
input追加、`startRunning()` 前後を記録する診断ログを追加した。

`stale six-axis register snapshot` の `sample_age_s` はキャッシュ済みSnapshotの年齢ではない。
Followerに対し `Present_Position → Goal_Position → Torque_Enable → Status` を同期・逐次で
6軸一括読取りした**全所要時間**である。4回目の読取り終了時刻から最初の読取り開始時刻を
引いており、100 ms超過はこの一連の読取りが遅かったことを示す。Alignment後のSnapshotは
毎回新規に同期読取りされ、Alignment中の古いSnapshotを再利用する経路はない。

カメラ処理はAlignment前からバックグラウンドで動くため、CPU/USB/OBS負荷がスケジューリングへ
影響する可能性はある。しかしエラーが `Teleop active.` 直後の最初のtickで出る場合、ACTの
MPS推論はまだ始まっていない。したがって第一候補は4回のシリアル読取り自体の一時遅延であり、
カメラとの因果関係は計測なしには確定できない。

## カメラ処理フロー

```text
cameras.local.yaml
  ├─ side: obs_source
  │    → OBS WebSocket接続
  │    → GetInputListでname+UUIDを一致確認
  │    → GetInputSettingsでcur_dev_ipを確認
  │    → GetSourceScreenshotをbackground threadで反復
  └─ wrist: avfoundation_unique_id
       → Swift helperを起動(uniqueID引数)
       → DiscoverySessionで候補列挙
       → uniqueID完全一致のdeviceだけを選択
       → AVCaptureDeviceInput(selected device)を1個追加
       → AVCaptureVideoDataOutputを追加
       → session.startRunning()
       → BGRA frameをstdout、診断をstderrへ出力
```

`CameraRig.connect()` はSideを先に、Wristを後に接続する。実行ログの
`SO101_AVFOUNDATION_DIAGNOSTIC` で `selected`、`input_added`、`capture_running` の
Unique IDがすべて `0x11240000c456366` か確認する。最終 `handoff_report.json` の
`camera_identity_diagnostics` にも保存される。

DiscoverySessionの生成と `devices` 参照はデバイスを「検索する」APIであり、コード上の
capture開始点は選択デバイスをInputへ追加した後の `startRunning()` である。列挙だけでMacの
緑インジケータが点灯するかはOS/driver実装の実機確認事項であり、静的解析からは断定しない。

## six-axis Snapshot処理フロー

```text
HandoffController.tick()
  → HardwareAdapter.snapshot()
      → t0 = perf_counter()
      → Follower sync_read(Present_Position, 6軸, retry=0)
      → Follower sync_read(Goal_Position,    6軸, retry=0)
      → Follower sync_read(Torque_Enable,    6軸, retry=0, raw)
      → Follower sync_read(Status,           6軸, retry=0, raw)
      → t1 = perf_counter()
      → sample_age_s = t1 - t0
  → _check_snapshot()
      → finiteかつ 0 <= sample_age_s <= 0.100 を要求
      → 超過時はsnapshot_rejectedイベントを記録して従来通りFAULT
```

Leaderの `request()` は別の逐次処理で、TELEOPの正常tickではFollower Snapshot検査後に
Leader `Present_Position`を読み、Follower Goalを書き込む。並列通信ではない。Alignment中も
Leader Present、Follower Present、Follower Goal writeを同一制御threadで逐次実行する。

今回の診断では4レジスタごとの所要時間、全体時間、失敗レジスタを
`handoff_report.json > servo_snapshot_diagnostics` に最大2000件保存する。通信例外は
`result=error` として失敗レジスタを残す。単なる全体遅延は4読取りが成功したまま
`threshold_exceeded=true` になるため、通信停止と区別できる。

## 原因候補と切り分け

|候補|コード上の根拠|現時点の判定|確認方法|対策候補（未適用）|
|---|---|---|---|---|
|SwiftがFaceTime HDを選択|Unique ID完全一致後、そのdeviceだけをInputへ追加|可能性低。フォールバックなし|native診断のselected/input_added/capture_runningを照合|一致しなければ即停止する追加gateを検討|
|AVFoundation列挙が内蔵カメラを列挙/driver probe|Discovery typeにbuiltInWideAngleCameraを含む|列挙はする。点灯との因果は未確認|カメラ専用previewでイベント時刻と緑インジケータを照合|Unique ID直接初期化または外部type限定を別変更として評価|
|古いOpenCV index=1経路|`obs_camera.py`にlegacy index=1が残存|今回のcamera-config経路では不使用|起動ログにCameraRig identityが出ることを確認|legacy利用時だけ明示的警告/廃止を別途検討|
|PyAVが実カメラを開く|PyAV利用箇所はDataset動画ファイル処理|今回のlive経路には証拠なし|process open filesを確認|不要|
|OBSが別シーンでFaceTimeを使用|本コードは指定InputのUUID/settingsのみ確認|OBS設定はrepo外のため未確認|OBSのSources/ScenesをGUIまたはWebSocket read-onlyで棚卸し|不要ソースをOBS側で無効化（確認後）|
|4回のFollower sync_read合計が100 ms超過|`sample_age_s=t1-t0`|最有力。今回の直接的停止条件|read-only latency JSONで各register分布を測定|閾値変更前に分布、失敗率、USB状態を収集|
|Alignment後に古いSnapshotが残る|HandoffController生成後、tickごとに新規snapshot()|コード上否定|sampled/completed monotonic時刻を確認|不要|
|MPS初回推論が原因|推論後にはfresh snapshotを再取得する実装|Teleop直後の障害原因ではない|イベントphaseとsnapshot時刻を照合|推論時間と通信時間を別系列で維持|
|カメラ/OBS/Swift負荷の間接影響|camera workerはAlignment前から並行動作|可能性あり、未確定|カメラのみ、servoのみ、両方のread-only計測を比較|優先度/QoS変更は根拠取得後に別提案|
|本当のシリアル通信停止|sync_read例外または特定registerの異常長時間|未確認|`result=error`, `failed_register`, exceptionを確認|USB/電源/ケーブルを調査。自動retryは追加しない|

## `.externalUnknown` と `.external`

AppleはmacOS 14で `.externalUnknown` をdeprecatedとし `.external` を推奨している。
ローカルSwift compilerも同じwarningを出す。ただし `.external` は新しいSDK/OS向けであり、
対象MacのmacOS deployment条件を確認せず置換すると古い環境との互換性を変える可能性がある。
今回は取得方式を変えない要件に従い置換していない。将来はavailability分岐、または
`AVCaptureDevice(uniqueID:)` の直接利用を、実機A/B確認付きの別変更として扱う。

参考：Apple公式 `AVCaptureDevice.DiscoverySession` はdevicesを検索・選択するAPIとして説明し、
`externalUnknown` はdeprecatedで`external`利用を案内している。

- https://developer.apple.com/documentation/avfoundation/avcapturedevice/discoverysession
- https://developer.apple.com/documentation/avfoundation/avcapturedevice/devicetype-swift.struct/externalunknown

## Mac実機での読み取り専用確認

以下はロボットを動かさない。servo診断はFollowerのUSBをread-onlyで開くため、他のLeRobot
processを停止してから実行する。電源・アーム支持は通常の安全手順を維持する。

```bash
# 1. カメラだけを20 Hzで10秒計測（モータportを開かない）
PYTHONPATH="$PWD/.local/camera-deps:$PWD/scripts/so101_capture" \
python scripts/so101_capture/preview_cameras.py \
  --camera-config scripts/so101_capture/cameras.local.yaml \
  --enable-cameras side,wrist --seconds 10 --hz 20 --diagnose-stale \
  --output "outputs/camera-identity-$(date +%Y%m%d_%H%M%S)"

# stderrのselected/input_added/capture_runningを確認
# 各行のunique_idが0x11240000c456366であること

# 2. Followerの4レジスタ×6軸を100回read-only計測
PYTHONPATH="$PWD/.local/camera-deps:$PWD/.local/act-deps:$PWD/scripts/so101_capture" \
python scripts/so101_capture/diagnose_servo_snapshot_read_only.py \
  --samples 100 --interval-ms 50 \
  --output "outputs/servo-snapshot-readonly-$(date +%Y%m%d_%H%M%S).json"

# 3. OSが認識するカメラ一覧（capture開始なし）
system_profiler SPCameraDataType

# 4. カメラを使用中のprocess確認（preview実行中と停止後を比較）
log stream --style compact \
  --predicate 'subsystem == "com.apple.cmio" OR subsystem == "com.apple.avfoundation"'
```

OBSについては、現在選択中のSceneだけでなく全Scene/GroupのSourcesを確認し、FaceTime HD、
OBS Virtual Camera、Continuity Cameraを参照するVideo Capture Device sourceがないか確認する。
OBSの設定はrepoに含まれないため、この調査では未確認である。

## 100 ms閾値について

現時点では適否を決める実測分布がない。20 Hz制御周期は50 msであり、4読取りに100 ms以上
かかる状態は既に2周期分を超えるため、安全側に停止する現設計には合理性がある。一方、4回の
逐次読取り時間を「Position測定の年齢」と同一視しているため、最初のPresentは後続3読取り分だけ
古くなる。閾値緩和、Status読取り頻度低減、bulk read化、別thread化はいずれも制御・安全挙動の
変更であり、read-only分布と障害時ログを取得するまで実施しない。

## 未解決事項

- 実際に点灯したのがMacの緑インジケータか、Continuity/OBS/DroidCam側の表示か。
- native診断で選択された実デバイスが毎回InnoMakerになるか。
- OBSの全Scene/GroupにFaceTime HD sourceが存在するか。
- servo-onlyとcamera同時実行で4-register latency分布が変化するか。
- 100 ms超過時に特定registerだけが遅いか、全体的なscheduler delayか、例外を伴うか。

安全閾値、カメラ方式、ACT時間・速度、Goal/Present、Calibration、frame鮮度・同期・重複検出は
今回変更していない。

## テスト時に確認した既存不一致

対象69テストのうち67件は成功した。失敗2件は今回未変更のfirst-motion初期Gapに関する既存不一致で、
HEADの実装定数 `TRIAL_MAX_INITIAL_GAP=2.0` に対してテストが1.5°超での拒否を期待している。
安全挙動を今回の調査に紛れて変更しないため未修正とした。診断追加テスト、100 ms拒否テスト、
カメラidentity保存テストはすべて成功した。Swift helperはコンパイル成功し、macOS 14以降で
`.externalUnknown` deprecated warningが出ることも確認した。
