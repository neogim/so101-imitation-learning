# 通信異常の停止と診断（2026-10-18）

実機接続なし。LeRobot本体・較正・既存実機Datasetへの変更なし。今回の差分は未コミット。
通信失敗後の終了処理については、以前の保持指令送信の説明より本書を優先する。

## 実装

SessionのIOをGuardedIOで包む。シリアルアクセスは引き続き主スレッドだけ。
Follower Present_Position/Torque_Enable、Leader Present_Positionは追加リトライ0回の読取りを行う。
最初の例外でラッチし、その時点からすべての新規Actionを禁止する。
同じ読取りを最大1回だけ診断目的で再試行し、その結果をログに残して必ず例外終了する。
再試行の結果を制御・記録へ返さず、古い状態の代用も行わない。
診断中も送信禁止。以降の通常読取りも禁止。Action送信例外もラッチし、送信を再試行しない。

終了処理はラッチ済みなら保持用の再読取り・Goal_Position書込みを行わず、STOPPING→STOPPEDで終了する。
終了処理の読取りで初めて失敗した場合にも、同じラッチ・診断・送信禁止が適用される。
トルクOFF指令は送らない。完了済み・途中のフレームは保存し、最後にfinalizeする。

**禁止するのは新しい指令の送信。サーボは最後の目標へ動き続けたり保持したりし得る。**
物理的な停止・位置保持・把持力の安全性を保証しない。プロセス終了後の監視もない。
通信異常ではない終了時の保持処理は従来どおり（グリッパは直前目標を維持）。
初期化中にConnectionError/OSErrorで失敗した場合も、従来の後始末の保持指令を実行しない。
詳細なGuardedIO計測対象はSession開始後。初期化前のエラーはsession_error.txt等で確認する。

ラッチを解除するキー・APIはない。r/tによる再開も禁止する。
通信回復を検出しても自動追従・自動記録・自動再設定・自動再較正しない。
原因確認とユーザー承認後、別の新規保存先でプロセスを起動し直す。

## communication_diagnostics.json

- register、role、servo_ids、mode、cycle（0開始）
- started_monotonic / finished_monotonic / duration_s、session_elapsed_s
- 同レジスタのlast_success_monotonicと全読取りのlast_any_read_success_monotonic
- UTC失敗検出時刻、例外型・本文、制御用/診断用の区別、attempt=1/2
- extra_read_retries=0、diagnostic_retry_limit=1
- 実baudrate、pyserial timeout、SDKのpacket_timeoutスナップショットとソース式での計算値
- 設定クラスのnum_read_retriesと、実際にこの経路へ渡す値0
- 最大256件の読取り履歴（SAFE_RESETで約6.4秒）、最大128周期（約6.4秒）の処理時間・周期間隔
- 正常/失敗周期を含むcycle_history、失敗前の履歴スナップショット、再試行結果
- レジスタ別全期間の試行数・失敗数・所要時間合計/最大値

失敗周期はfinallyで記録する。従来のcapture_timing.jsonに欠けていた失敗周期もここで確認できる。
ラッチ後、終了・接続クローズ後にJSONを保存し、動画保存完了を待つ前に出力する。
途中で電源断やプロセス強制終了、ディスク障害が起きた場合の保存は保証しない。
SDKはタイムアウト判定時にpacket_timeoutを0へ戻すことがあるため、スナップショット0を
「タイムアウト無効」と解釈しない。計算値と実際のduration_sを合わせて見る。

診断再読取り成功例（合成試験、実機の回復記録ではない）：

```json
{
  "latched": true,
  "automatic_resume": false,
  "failure": {
    "register": "Present_Position",
    "role": "follower",
    "servo_ids": [1, 2, 3, 4, 5, 6],
    "mode": "RECORDING",
    "ok": false,
    "attempt": 1,
    "exception": "injected position failure"
  },
  "diagnostic_retry": {"ok": true, "diagnostic": true, "attempt": 2}
}
```

完全なJSONはvalidation/communication_20261018/recovered_read_example.json参照。

## 頻度・タイムアウト・負荷

現時点ではPresent_Position、Torque_Enableを各20Hzのまま維持。期限も変更しない。
既定1Mbps・6台時のSDK式は位置約50.51ms、トルク約50.45ms。
1回の期限だけでも50ms周期を超え得る。最大1回の診断再試行により約101ms以上を要し得るが、
最初の失敗で制御停止は確定しており、その間にActionを出さない。OS待ち等を含む実時間の上限保証ではない。

毎周期トルク読取りはソフトウェア監視方針であり、LeRobotの記録APIの必須条件ではない。
例えば5Hzに下げると、位置20Hzを維持したまま同期読取りは40→25回/秒になるが、
トルク異常のポーリング間隔は50→200msへ悪化する（さらに通信・OSの遅延が加わる）。
今回は低頻度化していない。まず実機ログでレジスタ別の時間・失敗位置を測る。

合成3エピソードの処理時間：全制御平均0.721ms、最大7.163ms、50ms超過0。
合成位置読取り173回・トルク172回、位置の計測区間平均約0.022ms、トルク約0.008ms。
この区間には診断記録の準備も含む。**実機シリアルの所要時間を示す数値ではない。**
既存実機ログは両読取りを含む周期平均2.31msだったが、個々の読取り所要時間は復元できない。
今回の計測で次の承認済み実機試験から分離できる。

## テスト結果

- 単体45件成功（以前34件＋通信関連11件）。
- 位置の単発/連続失敗、トルクの単発/連続失敗、診断成功/失敗。
- 最初の失敗後、診断中・回復後・終了処理中の送信禁止。解除キーなし。
- 履歴上限、正常周期、失敗周期、読取り設定、設定取得失敗時の診断、JSON保存。
- 実Dataset統合4シナリオ：Ctrl-C相当、Leader通信失敗、Follower位置失敗後に診断成功、トルク連続失敗。
  各シナリオで完了1＋中断1エピソードを保存・読出し、全画像/動画数/Parquet/時刻を検証。
  学習派生には成功1件のみ。変換前後の合成原本ハッシュ一致。
- 3エピソードの正常CLI合成記録も完了。構文検査・差分空白検査成功。

## SAFE_RESETで確認待ちにする将来案

現在は確実に終了し、勝手に復帰しない方式。単発エラーでも再起動が必要になる。
将来のFAULT_WAITでは、同じ送信禁止ラッチを維持しつつ低頻度の診断読取りと操作者確認を待てる。
ただし、接続継続は安全な保持を意味しない。復帰には最新位置、トルク、姿勢差、周囲の安全確認と
明示承認が必要で、連続成功だけでは解除しない。今回FAULT_WAIT・復帰機能は実装していない。

## 次の無負荷実機試験（承認後のみ）

1. USBポート・電源・支持・ケーブル・OBS index 1を確認し、競合する別のロボット制御プロセスを終了する。
2. アームの可動域に物体や人の手を置かず、最後の目標へ動き続ける可能性を前提に電源対応手順を確認する。
3. 新規保存先で15秒上限・1エピソードを起動。初期化中はLeaderを静止させる。
4. SAFE_RESETで少なくとも60秒待ち、位置・トルクの監視を確認する。通信異常ならそこで試験終了。
5. 異常がなければrで10秒カウントダウンを確認し、無負荷で低速の小さな操作を行う。
6. 通常終了後は退避→h→q。異常時はr等で再開せず、支持・電源を適切に管理しJSONを確認する。
   エラー再現目的のUSB抜差しや電源断を動作中に行わない。
7. communication_diagnostics.jsonのレジスタ別時間、周期、失敗/診断結果を評価し、次の試験を判断する。

```bash
conda activate lerobot
cd /path/to/so101-imitation-learning
python scripts/so101_capture/record_episode.py \
  --root "$HOME/workspace/so101-data/no-object-test-002" \
  --seconds 15 --num-episodes 1 --countdown 10 \
  --task 'Test SO-101 teleoperation without an object.' --execute
```

存在する保存先は拒否する。実機試験はこのコマンドをまだ実行せず、明示承認後に行う。

## 今回の変更ファイル

- communication_guard.py（新規）：送信禁止ラッチ、診断再読取り、履歴・設定・統計
- capture_session.py：IO保護、周期計測、終了時送信禁止、診断JSON出力
- test_communication_guard.py（新規）：通信関連11件
- test_session_roundtrip.py：実Datasetを使う障害ケース追加、送信禁止・JSON検証
- COMMUNICATION_SAFETY.md（新規）：本書
- MULTI_EPISODE.md / GRIPPER_DIAGNOSTICS.md：現行通信仕様への案内
- validation/communication_20261018/（新規）：結果・合成診断例
