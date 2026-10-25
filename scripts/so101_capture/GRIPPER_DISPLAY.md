# B案：取得済みグリッパ値の表示

追加シリアル読取りなし。A案の接触モデルを呼ばず、位置制御・停止しきい値8・通信ラッチを変更しない。
実機操作は未実施。日時2026-10-18。

## 表示の意味

```text
[GRIP FRESH] mode=RECORDING cycle=187 unit=normalized_0_100
measured=23.571 prev_goal=16.222 error_prev=7.350 error_limit=8
leader_now=9.611 sent_now=14.722
measured_utc=... t_mono=... age_ms=25 event=normal | NOT a grip-force safety indication
```

上はmuhi-session-002の保存値を参考にした表示例。新版で実機取得した画面ではない。

- measured：その周期に読んだFollower実測値。
- prev_goal：その周期の位置取得前に保持していた前回送信目標。
- error_prev：abs(measured−prev_goal)。今回の目標との差と混同しない。
- leader_now：その周期で取得したLeader値。未取得なら`--`。前回値で埋めない。
- sent_now：その周期で実際に送信した目標。未送信なら`--`。
- measured_utc/t_mono：ホスト側の位置読取り完了時刻（UTC・単調時計）。サーボ内部計測時刻ではない。
- age_ms：描画時点で位置読取り完了から経過した時間。値は同時取得ではなく同じ制御周期内の順次取得。
- FRESH：位置情報の鮮度だけ。安全な把持力、接触、通信以外の安全性を認定する表示ではない。
- STALE：500ms超、通信異常、当該周期に有効な新しい位置がない場合、または表示ストリーム終了。

追従誤差停止の例：measured=23.163、prev_goal=14.722、error_prev=8.442、leader_now=--、sent_now=--。
停止を判断する周期では新しいLeader指令を読んでおらず、新しい目標も送っていない。
終了後の最終行にはSTREAM_ENDEDとSTALEを付ける。modeは値を取得した当時のモードであり、
そのモードで動作を続けているという意味ではない。終了後の連続監視はしない。

## 非ブロッキング構成

- 制御ループは小さな辞書をコピーして表示専用の容量1キューにput_nowaitするだけ。
- 満杯時は古い表示フレームを捨てる。最大2回の非待機試行で打ち切る。
- 表示スレッドが2〜5Hzで最新値を描画する。新規値がなくても再描画し、鮮度の低下を示す。
- 出力は複製したstderrのFDにos.write。制御側printのPython stdoutロックを共有しない。
- 出力先が詰まっても制御側は待たない。終了時の表示待ちは最大0.3秒で、制御と接続を終了した後に行う。
- 表示例外で制御やDataset保存を中断しない。表示のフレーム破棄はDatasetのフレーム破棄ではない。
- 最終表示用辞書、破棄数、描画数、表示エラー、未終了workerの有無をgripper_display.jsonへ保存。
  書込み自体が失敗してもDatasetのfinalizeを試みる。

端末/パイプ/OS自体の表示遅延は制御できない。ageは描画文字列を作った時点の値であり、
出力先でさらに滞留した場合は画面が古くなり得る。動かない画面を安全確認に用いない。
画面に最終行を出せない場合もあり、fault_diagnostics.jsonとcommunication_diagnostics.jsonが診断原本。
汎用OSのため、ハードリアルタイムの周期保証ではない。

## CLI

既定OFF。`--gripper-display`でON、`--no-gripper-display`でOFF。
`--gripper-display-hz`は2〜5、既定4。ON/OFFでサーボへの読取り/指令を増減しない。
TTYではグリッパ状態を同じ行に更新する。狭い端末では折り返す場合がある。
端末行の`M/P/E/L/S`は順に実測位置・前回目標・その誤差・Leader要求・今回送信目標。
キーを押すと表示を約1.5秒休止し、KEY RECEIVEDとKEY RESULTを見えるようにする。
キーは制御スレッドだけが標準入力から読み、状態遷移時に端末入力を消去しない。
受信したキー・周期・遷移先・拒否理由は保存先の`key_events.json`にも記録する。
実機の`keyboard_diagnostics.json`には入力ポーリング回数と読んだバイト数を残す。
`r`が通らない場合はKEY RECEIVEDの有無を見てから、KEY RESULTの0.5秒待ち・姿勢差・前エピソード保存中を確認する。
ノート本文の文字はキー診断ログに記録しない。
備考入力中も監視は継続する。表示が入力を見にくくする場合は次回起動時にOFFを選べる。

## 最初の成功デモ1本（ユーザー承認後のみ）

```bash
conda activate lerobot
cd /path/to/so101-imitation-learning
python scripts/so101_capture/record_episode.py \
  --root "$HOME/workspace/so101-data/muhi-display-demo-001" \
  --seconds 40 --num-episodes 1 --countdown 10 \
  --gripper-display --gripper-display-hz 4 \
  --task 'Pick up the Muhi bottle and place it into the target container.' \
  --execute
```

保存先は新規のみ。今回はこの実機コマンドを実行していない。
起動後、SAFE_RESETで鮮度と値を確認し、r→10秒→記録開始。
ゆっくり閉じ、誤差の増加をさらに締め込んで補わない。表示の確認でロボットから注意をそらし過ぎない。
完了したらeで記録終了（追従は継続）。退避・物体の支持を確保しh→結果入力→q。
成功ならp/l/sで各結果を設定し、aで採用する。失敗・未判定は不採用。

危険時は記録より安全を優先。Ctrl-Cは通常ループを終了するが、グリッパには最後の目標が残り得る。
通信異常時は終了処理も新しい指令を送らない。物理的な停止・力解除・位置保持は保証しない。
eは安全停止キーではなく、hも把持力の除去手段ではない。

収録後はverify_dataset.pyで別出力先へ検証し、動画・関節・Actionと診断JSONを確認する。
成功の再現性を確認するまでは本数を増やさない。

## 今回のファイル

- gripper_display.py（新規）：描画・容量1キュー・表示スレッド
- capture_session.py：取得済み値のスナップショット、CLI、表示終了処理
- test_gripper_display.py（新規）：11テスト
- GRIPPER_DISPLAY.md（新規）：本手順
- MULTI_EPISODE.md：案内リンク
- validation/display_20261018/（新規）：テスト結果・周期計測・Dataset検証結果

以前の未コミット差分は維持。接触モデル・LeRobot本体・較正・既存実機Datasetは今回変更していない。
