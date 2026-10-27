# SO-101 ACT実行時 stale snapshot 追加調査

## 調査範囲

- `outputs/**/handoff_report.json`: 73件
- Step 16000 continuous 60s系: 21件
- `outputs/**/preview.json`: 13件
- `outputs/servo-snapshot-readonly-20261028_014018.json`: 100 sample
- コード基準: LeRobot 0.6.2、commit `79ddf43`以降

Dataset、checkpoint、カメラ鮮度条件、100 ms閾値、Goal/Present・Calibration・通信停止条件は変更していない。

## 結論

### 確定事項

1. 73件のHandoffレポート中、`stale six-axis register snapshot` は1件だった。
2. 該当実行は `so101-act53-step16000-continuous-60s-20261028_012240` で、停止phaseは
   **TELEOP**。Leader handoff、CHECK、ACT、MPS推論の前である。したがって、MPS推論が始まる前でも
   100 ms超過は発生し得る。
3. 同実行は260回のTELEOP cycleを正常通過した後、cycle 260で停止した。正常260 cycleの
   work timeはP50 5.149 ms、P95 10.057 ms、P99 13.467 ms、最大19.334 msだった。
   異常cycleだけ136.160 msで、直前cycle startからのintervalも88.060 msだった。
4. 最初のTELEOP cycleからfault eventまでは、記録済みcycle intervalと最後のwork timeの合計で
   約14.242秒。Alignment完了時刻そのものはJSONにないため、「Alignment完了から」の厳密値は
   測定不可である。
5. 単独read-only 100 sampleは平均4.155 ms、P95 4.980 ms、最大6.906 ms、通信error 0、
   100 ms超過0。恒常的な低速通信ではない。
6. stale発生ログは診断commit `79ddf43`以前に作られている。したがって、超過時のレジスタ別時間、
   Snapshot完了からvalidationまでの時間、同時刻のカメラfetch時間は**測定不可**である。
7. 過去13件のcamera previewでは、Wrist direct captureのUnique IDは
   `0x11240000c456366`、Side UUIDは`4cf2bd63-541c-4346-a008-48683c9bf487`。
   FaceTime HDが選択された記録はない。ただし旧ログはnative device詳細追加前なので、
   `localizedName`までの証明は測定不可である。

### 有力仮説

異常cycleは通常値から単発で約120 ms以上跳ねている。候補は次のいずれか、または複合である。

- 4回の`sync_read`のうち1回がpacket受信待ちになった。
- Python process/threadがcamera worker、OBS応答処理、画像decode/hash、OS schedulerにより
  `sync_read`中またはレジスタ間で長時間descheduleされた。
- USB serialとUSB Wrist cameraのhost controller/帯域/driver処理が競合した。
- macOS全体の一時的なscheduler/IO stall。

現存ログだけではこの4候補を順位付けできない。異常cycleの直前intervalも88 msだったため、
Snapshot読取り開始前からprocess schedulingの遅れがあった可能性はあるが、intervalには前cycle後の
sleepとOS schedulingが含まれ、原因の断定材料にはならない。

### 現時点で弱い仮説

- **MPS推論そのもの**: stale実測はTELEOP phaseで発生しており、この時点で推論未開始。
  ACT phaseの将来の遅延要因にはなり得るが、今回の1件の直接原因ではない。
- **古いSnapshotの再利用**: `HardwareAdapter.snapshot()`は各tickで同期読取りを新規実行する。
- **恒常的に遅いservo bus**: 単独100 sampleの最大は6.906 ms。

## 過去ログの正常・異常比較

|項目|stale実行|同実行の正常TELEOP cycle|過去の正常Step 16000実行|
|---|---:|---:|---:|
|phase|TELEOP|TELEOP|TELEOP/CHECK/ACT|
|cycle数|261|260|実行ごとに異なる|
|cycle work P50|測定対象1件|5.149 ms|Snapshot単独分布は測定不可|
|cycle work P95|測定対象1件|10.057 ms|同上|
|cycle work P99|測定対象1件|13.467 ms|同上|
|cycle work最大|136.160 ms|19.334 ms|全cycle最大にはMPS推論を含み、比較不可|
|4-register合計|100 ms超と判定|個別値は未保存|arm時の単発値のみ。分布は測定不可|
|レジスタ別内訳|測定不可|測定不可|測定不可|
|camera同時刻|測定不可|測定不可|測定不可|
|MPS実行中か|いいえ|いいえ|phaseによる|

Step 16000 continuous系21件の内訳は、60秒正常完了10件、OBS screenshot response fault 3件、
stale snapshot 1件、hardware startup段階の拒否/失敗7件。正常ACT runの
`cycle_work_max_ms`には初回/定期MPS推論が含まれ419.97〜895.35 msに達するが、実装は推論後に
fresh Snapshotを再読取りするため、この値をservo Snapshot時間として扱ってはならない。

過去レポートの`act_armed.snapshot.sample_age_s`は一部runに1点だけ存在し、おおむね
3.7〜6.3 msだった。しかし全cycle分ではなく、stale cycle自身には値が残っていないため、
分布や異常時比較には使用できない。

## Camera処理とserial通信

### Side / OBS

`OBSSourceCamera`はbackground Python threadでWebSocket `GetSourceScreenshot`を反復する。
応答後にbase64 decode、Pillow JPEG decode、NumPy copy、SHA-256を実行する。network waitや一部native
処理がGILを解放しても、JSON/base64/Python bookkeeping等はGILを必要とするため、main threadの
schedulingへ短時間影響する可能性がある。既存previewではsample copy P95は概ね0.17〜0.26 msだが、
これはservo同時実行ではなく、background fetch全体のGIL影響を直接測った値でもない。

### Wrist / AVFoundation

capture callback、scale、BGRA生成は別Swift processで行われ、PythonとはGILを共有しない。
Python側のreader threadはpipe read、NumPy channel変換、resize、copy、SHA-256を行うため、Python
scheduling負荷は残る。またUSB cameraならservo USB serialとのmacOS USB stack/host controller競合は
GILとは別に起こり得る。どの物理controllerへ接続されているかは現存ログから測定不可。

### MPS

Torch/MPS推論はACT/CHECK時にCPU preprocessingとGPU command submissionを含み、長いcycleを作る。
ただしcontinuous ACTでは推論後に新しい4-register Snapshotを取得し直す。今回のstaleはTELEOPで、
MPSは未開始である。

### Serial retry / timeout

全Snapshot readは`num_retry=0`。LeRobot `sync_read()`は1回だけ`GroupSyncRead.txRxPacket()`を呼び、
失敗時は自動retryせず例外にする。SDKはbroadcast request後、6 servoのresponseをPython/pyserialで
順に読む。packet timeoutは期待packet長から設定される。

## 時間内訳と追加診断

今回、動作を変えず次を追加した。

|診断値|意味|限界|
|---|---|---|
|`sample_start_lateness_ms`|50 ms予定時刻からread-only sample開始までの遅延|OS sleep/schedulerを含む|
|`sync_read_ms`|Pythonから`bus.sync_read()`を呼び、返るまで|serial TX/RX、SDK decode、GIL待ちを分離できない|
|`python_processing_ms`|`sync_read()`返却後の件数/辞書変換|非常に小さい処理のみ|
|`inter_register_gap_ms`|前register処理完了から次register開始まで|main threadのdescheduleを検出可能|
|`snapshot_to_validation_ms`|4読取り完了から`_check_snapshot()`開始まで|実機ACT reportに保存|
|camera fetch統計|request/decode/total/capture-to-host|sensor exposure timestampはSideで取得不能|

「実際のwire上のserial通信時間」と`sync_read()`内のPython SDK処理を完全に分けるには、LeRobotまたは
scservo SDK内部のTX、各servo RX、decode境界へのinstrumentationが必要である。これは依存packageへの
patchまたはlow-level wrapperとなるため今回は行っていない。今回の境界で、まず次を区別できる。

- `sync_read_ms`だけ大: serial/SDK内部
- `inter_register_gap_ms`が大: Python thread deschedule/GIL/scheduler
- `sample_start_lateness_ms`が大: sample開始前のprocess scheduling
- `python_processing_ms`が大: 返却後Python処理
- `snapshot_to_validation_ms`が大: Snapshot完成後のmain-thread遅延

## A〜D読み取り専用実験

全条件でGoal、Torque、configure、Alignment、Calibration writeを行わない。各条件を最低3回、可能なら
1000 sampleで実行する。実験間は他のcamera/LeRobot processを停止し、同じUSB接続を維持する。

```bash
export PYTHONPATH="$PWD/.local/camera-deps:$PWD/.local/act-deps:$PWD/scripts/so101_capture:${PYTHONPATH:-}"

# A: camera OFF / ACT OFF / servo read-only
python scripts/so101_capture/diagnose_servo_snapshot_read_only.py \
  --samples 1000 --interval-ms 50 \
  --output "outputs/servo-A-$(date +%Y%m%d_%H%M%S).json"

# B: Side only
python scripts/so101_capture/diagnose_servo_snapshot_read_only.py \
  --samples 1000 --interval-ms 50 \
  --camera-config scripts/so101_capture/cameras.local.yaml --enable-cameras side \
  --output "outputs/servo-B-side-$(date +%Y%m%d_%H%M%S).json"

# C: Wrist only
python scripts/so101_capture/diagnose_servo_snapshot_read_only.py \
  --samples 1000 --interval-ms 50 \
  --camera-config scripts/so101_capture/cameras.local.yaml --enable-cameras wrist \
  --output "outputs/servo-C-wrist-$(date +%Y%m%d_%H%M%S).json"

# D: Side + Wrist
python scripts/so101_capture/diagnose_servo_snapshot_read_only.py \
  --samples 1000 --interval-ms 50 \
  --camera-config scripts/so101_capture/cameras.local.yaml --enable-cameras side,wrist \
  --output "outputs/servo-D-both-$(date +%Y%m%d_%H%M%S).json"
```

Camera有効条件ではstderrの`SO101_AVFOUNDATION_DIAGNOSTIC`とJSONの`camera`を確認する。
Wristの`selected/input_added/capture_running`がすべてInnoMakerのUnique ID
`0x11240000c456366`であること。FaceTime HDのUnique IDが`selected`以降へ現れた場合は実験を中止する。
`discovered`にFaceTimeが現れるだけでは選択・capture開始を意味しない。

比較CSV:

```bash
python scripts/so101_capture/compare_servo_snapshot_diagnostics.py \
  --input A=outputs/servo-A-....json \
  --input B=outputs/servo-B-side-....json \
  --input C=outputs/servo-C-wrist-....json \
  --input D=outputs/servo-D-both-....json \
  --output outputs/servo-A-D-comparison.csv
```

CSVには条件ごとのmean/P50/P95/P99/max、100 ms超過数、error数、レジスタ別統計、camera identityが入る。

## 条件E

通常ACT実行は自動実行しない。ユーザーの既存安全確認、明示的approval、手動`x`/`q`操作を維持する。
次回runの`handoff_report.json`では以下を確認する。

1. `original_exception.mode_before`
2. `servo_snapshot_diagnostics`末尾の`total_duration_ms`
3. 同じsampleの4件の`sync_read_ms`、`python_transform_ms`、`inter_register_gap_ms`
4. `events`の`snapshot_rejected.snapshot_to_validation_ms`
5. `camera_identity_diagnostics`の選択device
6. fault近傍の`cycle_timings`

## A〜D比較表（実測記入欄）

|条件|Camera|N|P50 ms|P95 ms|P99 ms|max ms|>100 ms|error|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|A|OFF|100（既存）|4.021|4.980|5.923|6.906|0|0|
|B|Side|未実施|—|—|—|—|—|—|
|C|Wrist|未実施|—|—|—|—|—|—|
|D|Side + Wrist|未実施|—|—|—|—|—|—|

既存AのP99は保存済みraw 100 sampleをnearest-rank法で再集計した実測値であり推測値ではない。
更新後scriptで再実行したA〜DはP99をsummaryへ直接保存する。

## 次の判断基準

- Bだけ悪化: OBS/JPEG/GIL側を重点調査。
- Cだけ悪化: AVFoundation Python pipe処理またはUSB camera/serial競合を重点調査。
- Dだけ悪化: 複合負荷・scheduler/USB帯域を重点調査。
- A〜Dすべて正常、Eだけ悪化: ACT preprocessing/MPS、Goal writeを含む制御cycle、またはpowered servo状態を比較。
- `inter_register_gap_ms`大: Python scheduling/GIL寄り。
- 特定`sync_read_ms`大: serial/SDK/USB寄り。
- 全条件で稀に大: macOS/USB/電源/ケーブルを含む単発stallの長時間測定が必要。

100 ms閾値緩和、retry追加、read頻度削減、thread優先度変更、bulk化は、この比較結果を得るまで実施しない。
