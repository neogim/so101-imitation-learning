> B案の表示オプションと成功デモ1本の手順は [GRIPPER_DISPLAY.md](GRIPPER_DISPLAY.md) を参照。

> 通信異常時は保持指令も禁止します。最新仕様は [COMMUNICATION_SAFETY.md](COMMUNICATION_SAFETY.md) を参照。

> 追従誤差診断・カウントダウン・終了時のグリッパ目標維持の更新は [GRIPPER_DIAGNOSTICS.md](GRIPPER_DIAGNOSTICS.md) を参照。終了時の仕様はそちらを優先。

# SO-101 複数エピソード収録（2026-10-18）

この手順が現在の `record_episode.py` の仕様。RECORDING.md / STAGE*.md は過去の調査記録。
今回の検証は合成カメラ・ダミーRobotのみ。実機・OBS接続・自律動作は未実行。
`~/workspace/so101-data/pick-place-003`、LeRobot本体、依存関係、較正ファイルは変更しない。

## 変更設計

| 状態 | Leader追従 | Dataset記録 | 安全条件・遷移 |
|---|---|---|---|
| RECORDING | ON、目標変化量制限 | ON | `e`（最短1秒経過後）または上限時間でTELEOP_RESET |
| TELEOP_RESET | ON、同じ制限 | OFF | `h`で現在位置を保持目標として1回送信しSAFE_RESETへ |
| SAFE_RESET | OFF | OFF | 位置・トルクレジスタを監視。通常周期でGoal_Positionを送らない |
| COUNTDOWN | OFF | OFF | `r`後の補助状態。Leaderが3単位超動けばキャンセル |
| STOPPED | OFF | OFF | 終了。通信を閉じる。自動トルク解除なし |

初回の較正確認・configure・既存の低速位置合わせは1回のみ。エピソード間はUSB接続を維持する。
SAFE_RESETで少なくとも0.5秒監視し、Leader/Followerの全軸差が3以下なら `r` / `t` を許可。
差が大きい場合は**再開拒否**。保持されたFollowerに合わせてLeaderだけを動かし、再操作する。
エピソード間の自動低速位置合わせは追加していない。大差を急に追従する経路は設けない。

既存の初回初期化には短いトルク解除区間があるため、起動時は腕を安定させる。
初回は最大3度/秒（グリッパ3正規化単位/秒）で既存の位置合わせを行う。
途中停止条件・較正書込み禁止ガードは既存と同じ。初回のLeaderは静止させる。

制御スレッドだけがRobotへ指令を送る。保存スレッドはフレームと境界をFIFO順に処理し、
各エピソードでsave_episode、最後にfinalize。保存中もTELEOP_RESET/SAFE_RESETを継続。
次の記録は直前のsave_episodeが完了するまで拒否する。キュー上限64フレーム。
満杯時にフレームを黙って捨てず停止する。保存スレッド失敗も主制御へ通知する。

## 実機前の確認

1. この新版での実機動作についてユーザーの明示的承認を得る。
2. USBポートを `ls /dev/tty.usbmodem*` で確認。Follower末尾`5B7B0098201`、Leader末尾`5B7B0185961`。
   別名なら対応を確認する。別の機器へ自動切替しない。
3. iPhone 6 Plusの映像をOBSで確認し、仮想カメラを開始する。アダプターは確認済みindex 1限定。
   640×480の画像を毎フレーム検査する。実画像の回転・作業域もOBSで確認する。
4. 作業域とケーブルを確保し、Leader/Followerを近い姿勢に置く。電源と接続を確認する。
5. 実ターミナルで実行する。パイプ入力・IDEの非TTYコンソールでは実機モードを拒否する。
6. 起動時の位置合わせが終わるまでLeaderを静止させる。SAFE_RESET表示後、保持を目視確認する。

## 実行コマンド（実機承認後のみ）

```bash
conda activate lerobot
cd /path/to/so101-imitation-learning
python scripts/so101_capture/record_episode.py \
  --root "$HOME/workspace/so101-data/muhi-session-001" \
  --seconds 40 --num-episodes 5 --min-seconds 1 --countdown 3 \
  --task 'Pick up the Muhi bottle and place it into the target container.' \
  --execute
```

保存先は存在しない新規ディレクトリが必須。親ディレクトリは既存で書込み可能なものを使う。
既存なら上書きせず起動を拒否する。Hub送信の実装はなく、HFをoffline設定にする。
`--execute`を付けない場合はUSBパス・較正ファイル・保存先だけ検査し、カメラもシリアルも開かない。
`--simulate`では実機・カメラを開かず、自動キー列で連続模擬収録する。

## 収録時の操作

キーはEnter不要（備考の確定のみEnter）。大文字はShift付き。

| キー | 操作 |
|---|---|
| `r` | SAFE_RESETから姿勢確認→カウントダウン→記録開始 |
| `e` | RECORDINGを正常に早期終了。成功にはしない。最初の1秒は無効 |
| `h` | TELEOP_RESETからSAFE_RESETへ |
| `t` | SAFE_RESETから姿勢確認後にテレオペ再開（記録しない） |
| `p` / `P` | SAFE_RESET中、直前エピソードのPick成功 / 失敗 |
| `l` / `L` | Place成功 / 失敗 |
| `s` / `f` / `u` | 全体成功 / 失敗 / 未判定 |
| `a` | 学習採用を切替。Pick・Place・全体が成功、かつ正常終了のみ許可 |
| `n` | 備考入力。Enterで保存、Escでキャンセル。保持監視は継続 |
| `q` | SAFE_RESETからセッション終了。予定本数前でも終了可能 |
| Ctrl-C | 異常・安全終了。通常のエピソード完了とは別扱い |

`r`→収録→`e`または上限→**追従は続いている**→Leaderで退避→`h`→実際の静止を確認→
ムヒと透明容器を戻す→結果を入力→Leaderを保持姿勢に近づける→`r`。
最終エピソード後も勝手に追従を切らない。退避→`h`→結果入力→`q`で終了する。
結果入力を忘れたエピソードは未判定・不採用。結果変更時は採用を解除するため、最後に`a`を押す。

## 安全条件と限界

- 全周期で有限値・関節キー・既存較正から変換した関節範囲、FollowerのTorque_Enableを確認。
- 追従目標の変化率は既定30度/秒（グリッパ30単位/秒）。`--slew`で変更可能。
  これは**目標値の変化制限**であり、実機の物理速度や衝突回避を保証するものではない。
- 目標と実測の差8超、保持姿勢から2超のずれ、250ms超の制御間隔で停止する。
  度とグリッパ正規化値は別単位として扱う。較正端点の数値許容差は0.2。
- SAFE_RESETではLeader由来のActionを送らない。明示的な保持開始と終了時に限り現在位置を目標にする。
- 異常時は追従を止め、現在位置を取得できる場合に保持指令を1回試みる。自動リトライ駆動をしない。
  通信・トルク・電源異常時には**位置保持を保証できない**。
- 終了時はシリアルを閉じるが、トルクOFF指令を送らない。終了後の連続監視も行わない。
  安全な支持を確保してからユーザーが電源を管理する。保持表示だけを信用して腕の下へ手を入れない。
- 初期較正不一致ではconfigureも保持指令書込みも実施せず中止。較正ファイルのハッシュを前後確認する。
- Ctrl-C/通信例外では受信済みフレームを中断エピソードとして保存してfinalizeする。
  保存系自体の破損、ディスク容量不足、強制終了、停電では完全な保存を保証できない。
  `writer_error.txt` / `session_error.txt` / `shutdown.json`を確認し、途中ファイルを勝手に消さない。

## データと検証

原本：RGB `(480,640,3)`、`observation.state` / `action` / `action_requested` 各float32[6]。
回転5軸degrees、グリッパ0〜100。`action`は制限適用後のsend_action戻り値（送信目標）であり、
モータが到達・受理したことの確認値ではない。`action_requested`はLeaderの元の位置指令。
エピソードごとのframe_indexは0開始、timestampはframe_index/20。全体indexは連番。
これらの公称時刻はiPhoneの撮影時刻を意味しない。`capture_timing.json`にはホスト単調時計による
状態取得・送信後・画像キュー投入時刻と処理時間を別に残す。Wi-Fi遅延はここから求められない。

`episode_outcomes.json`に終了理由manual/time_limit/interrupted、実経過時間、公称時間、フレーム数、
Pick/Place/全体結果、採用、備考を保存。`session_events.json`にモード遷移を保存する。

```bash
python scripts/so101_capture/verify_dataset.py \
  "$HOME/workspace/so101-data/muhi-session-001" \
  --output "$HOME/workspace/so101-data/muhi-session-001-check" --body-unit degrees
python scripts/so101_capture/export_act_dataset.py \
  "$HOME/workspace/so101-data/muhi-session-001" \
  --output "$HOME/workspace/so101-data/muhi-act-train-001"
```

検証は全画像をデコードし、エピソード境界、公称時刻、全体index、関節・指令の有限値、
動画のPTS区間とParquetのフレーム数を照合。HTMLは約250枚のプレビューを含む。
ACT用派生Datasetは成功・採用済みのエピソードのみ別ディレクトリへ再保存し、統計を再計算する。
原本を編集しない。派生には画像・state・actionと標準フィールドだけを残し、source_lineage.jsonで対応を保存する。
LeRobotの特徴量推定はactionで始まるキーをACTIONと扱うため、action_requestedを残したまま学習に渡さない。
学習用と評価用はエピソード単位で分ける。学習デモを使った動作確認は未学習条件での評価成功率とは別扱い。

## 非実機テストの再実行

```bash
python -m unittest discover -s scripts/so101_capture -p 'test_*.py' -v
python scripts/so101_capture/record_episode.py --simulate \
  --root /tmp/so101-new-session-test --seconds 2 --num-episodes 3 --min-seconds .5 --countdown .2
python scripts/so101_capture/test_session_roundtrip.py --output /tmp/so101-new-fault-test
python scripts/so101_capture/act_smoke.py /tmp/so101-new-fault-test/interrupt-act \
  --output /tmp/so101-new-act-smoke
```

シミュレーションだけが自動キー・自動の成功ラベルを使う。実機では自動採用しない。
旧`check_recorded_actions.py`は旧DummyRobotの「送信=要求×0.5」専用検査として維持。
新セッションの検証にはverify_datasetとtest_capture_sessionを使う。
ACT smokeは外部ダウンロードなし、CPU、縮小Transformer、ランダム初期化、1ステップのみ。
標準の本学習やMPS/GPU性能・成功率を証明するテストではない。生成された正規化Actionを実機へ送ってはならない。
