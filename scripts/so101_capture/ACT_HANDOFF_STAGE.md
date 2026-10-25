# B方式：同一接続のLeader→ACT引継ぎ（実機試験済み）

`act_handoff_launcher.py` の通常起動は、10秒間の無負荷実機テレオペで成功した `record_episode.start_hardware(..., guarded_teleop_start=True)` を再利用する。モデルは `--checkpoint` で指定し、190本モデルは `outputs/so101-act-muhi-190-step20000-20261022_1326/checkpoints/014000/pretrained_model`。OBS index 1の640×480 RGBからPIL bilinearで168×224へ変換する。6軸の順序は `act_action_guard.NAMES`。回転5軸は度、グリッパは0–100。

通常起動では独立した読み取り専用プローブを再接続しない。成功したテレオペと同じ接続内で較正・初期姿勢差を確認し、FollowerトルクOFF後の新鮮なPresent生値を一括Goalへ一度だけ送る。書込みから20ms待って最初の読戻しを行い、さらに20ms待って全6軸のGoal/Present/Torque/Statusを再確認する。Goal/Present差と位置変化は5生tick以内、Status=0、Torqueは全軸で一致することを要求する。同期後の全軸Torque=1は実機で再現した事実として警告・記録するが、正常仕様とは断定しない。部分反映・通信異常ならconfigure/Alignmentへ進まず、Goal書込みや読取りを自動再試行しない。

Followerの`configure()`は従来のLeRobot実装を使用し、例外時だけトルク再有効化しない局所的なcontext managerを適用する。正常終了後に再び6軸を確認してから既存の低速Alignmentへ進む。初期化時の生値は`startup_checkpoints.json`に残す。**観測専用でも初期化とAlignmentで実機駆動が起こり得る。** Goal書込み後の予期しないトルクON原因は未解明。ソフトウェア検知は事後的で、読取りまでの動作や保持力を防げない。腕の支持、周囲退避、独立したサーボ電源遮断手段と操作者が必要。

Alignment後、1つの20 HzスレッドだけがFollowerのGoalを送る。`x`でLeader送信を止め、`CHECK`中はLeader/ACT両方の送信を禁止する。送信処理は同期呼出しなので未処理Goalキューはない。6軸のGoal、Present、Torque、Statusを読戻し、既存較正、鮮度、Goal/Present追従誤差8、直前送信Goalとの一致を検査する。実画像でACTを一度ウォームアップし、再度新鮮な6軸値を検査する。観測専用モードではさらに2秒・40周期の推論を行い、ACT Goal送信は0件のまま終了する。

ACT書込みモードでは、同じゲートが `ACT` 所有権になった次の周期からだけ送信する。モデル重みは変えず `n_action_steps=5` とし、50 Actionを予測して先頭5個だけを使用する。ガードは較正範囲、グリッパ0–100、設定された有限の速度上限、追従誤差8、100 msの測定鮮度を監視する。通常モードの既定速度は回転10度/秒・グリッパ20正規化値/秒で、最大30単位/秒まで指定できる。閉じ側追加制限は未測定のため `None`。通信・トルク・Status・推論・周期の異常はラッチし、復帰やGoal書込みを自動実行しない。終了時もGoal・トルクを書かず、接続のみ閉じる。**ソフト停止後もサーボの保持トルクが残り得る。** 物理停止や保持は保証されない。

## 過去のGoal同期専用試験（次回は使用しない）

前回のGoal生値0を単に起動拒否するだけでは前進しないため、通常起動とは独立した**一括Goal同期のみ**の試験段階を設けた。`--goal-sync-only`だけは保存Goalが較正範囲外でも、Torque=0、Status=0、Presentが較正範囲内、較正一致、試験前の位置安定を確認して進める。Followerだけ接続し、Leader、OBS、ACTモデルは使わない。新鮮なPresent生値を一度だけ一括`sync_write`でGoalへ送り、直後、約20/100/500ms後の6軸Goal/Present/Torque/Statusを`startup_checkpoints.json`へ記録する。Goal/Present差は5生tick以内、位置変化も5生tick以内を確認する。**成功してもready=Falseのまま終了し、configure、トルクON、Alignment、テレオペ、ACTには進まない。** 通常起動では保存Goalの較正範囲外を引き続き拒否する。

一括書込みがトルクONを引き起こすかは未解明。ソフトウェアの異常検知は事後的であり、書込みから検知までの動作を止められない。試験条件は、腕を確実に支持できること、ムヒなどの物体を置かないこと、周囲に人・障害物がないこと、USBとは独立したサーボ電源遮断手段とその操作者がいること。どれかを満たせなければ**実施しない**。Goal読戻し不一致、Torque/Status変化、位置変化超過、通信例外、予期しない音・動作があれば、ソフトウェアは追加書込みをせず、操作者は腕を支持してサーボ電源を物理的に遮断する。切断後の保持力と物体保持は保証しない。通信失敗時は一括書込みの到達が不明なので再実行しない。

次は**別途承認後にだけ使う同期専用コマンド**。出力先は未使用のディレクトリを指定する。`--hardware-init-approved`、`--hardware-act-approved`は付けない。

```sh
conda activate lerobot
cd /path/to/so101-imitation-learning
python scripts/so101_capture/act_handoff_launcher.py \
  --mode hardware --goal-sync-only --hardware-goal-sync-approved \
  --ack-arm-supported --ack-clear-workspace --ack-power-stop-ready \
  --output outputs/so101-act-goal-sync-only-001
```

判定は6軸すべてGoal読戻しが送信したPresentと一致（許容5生tick）、Torque=0、Status=0、位置変化5生tick以内、通信異常なし、かつ物理的な動作なし。満たしても予期しないトルクONの原因が解明されたわけではない。次のconfigure・Alignment試験はログと安全対策を再評価し、別途承認する。

1. 別途承認後、Leader/FollowerのUSB、較正、OBSの実際のカメラインデックス、選択したCheckpointと出力先の未使用を確認する。現在の `obs_camera.py` はindex 1を指定している。ムヒを置かず、腕を支え、周囲を空け、独立したサーボ電源遮断手段を用意する。
2. 観測専用モードで起動する。既存初期化とAlignmentは駆動を伴う。Leaderで小さくテレオペし、異常がなければ`x`を押す。ここからLeader Goalは止まる。
3. `CHECK`で2回の6軸読戻しとACTウォームアップを行い、続いてGoal送信を禁止したままデフォルト2秒・40周期の`OBSERVE_ACT`を実行して自動終了する。`handoff_report.json`の`post_handoff_goal_write_attempts=0`、`last_owner=CHECK`を確認する。`cycle_timings`には全制御周期の実測間隔と処理時間、`observe_act`イベントには5フレーム境界と予測Actionを残す。`observed_rgb_640x480.png`と`model_rgb_224x168.png`を見て映像の向きと前処理を確認する。ウォームアップ周期は50 msを超え得るため、通常の観測周期と区別して評価する。

各5フレームchunkの再予測で実際に渡した画像は `act_input_000_640x480.png` / `act_input_000_224x168.png` などとして終了後に保存する。`observe_act` イベントには同じframe番号、画像取得のmonotonic/Unix時刻、6軸Present/Goal、逆正規化済み予測Action、各軸の予測・Goal絶対差、カメラインデックスとbackendを記録する。OpenCV経由でデバイス名を取得できない場合は `device_name=null` とする。ウォームアップ画像は別の `observed_rgb_640x480.png` であり、最初の予測chunkの画像とは区別する。
4. 異常なら新規Goal送信せず終了。腕を支持し、必要なら独立した物理サーボ電源遮断を行う。ACT Goal書込み試験には進まない。

次は**将来の観測専用切替用レビューコマンド**であり、別途、初期化・Alignmentを含む実機駆動について明示承認を受けた後にのみ使う。起動時の確認入力は `YES`。`--hardware-act-approved`と`--no-observe-only`は付けない。

```sh
conda activate lerobot
cd /path/to/so101-imitation-learning
python scripts/so101_capture/act_handoff_launcher.py \
  --mode hardware --hardware-init-approved --observe-only --observe-seconds 2 \
  --device mps \
  --output outputs/so101-act-handoff-observe-001
```

M1の通常実行環境でMPSが使用可能。実機の観測専用切替では40周期完了、切替後Goal書込み0件を確認したが、最大処理時間は約474msで50ms超過周期もあった。同様の遅延が駆動中に発生した場合は鮮度ガードで停止する。Goal=0の古い保存値だけでは起動拒否せず、成功済みテレオペと同じ一括Goal同期後の読戻しで判定する。その後の190本モデルでは初回1秒試験でACT Goal送信6回、通常40秒試験で732回を記録した。`sync_goal_to_present.py` の旧単軸手順は引き続き実機実行禁止。

## 初回自律駆動試験（別途実行承認後のみ）

通常の観測専用経路は維持する。初回試験は完了済みで、再実施しない。`--first-motion-trial` では `--no-observe-only --hardware-act-approved` と物理安全確認の3フラグをすべて要求する。CHECK中にACTをウォームアップし、最初の5-Action chunkを送信禁止のまま準備する。初回Action前に6軸Goal/Presentを再確認し、回転5軸の予測差が各5度超、初期Goal/Present差が回転軸で各1.5度超・グリッパで1.5 normalized_0_100超、観測が100ms超なら送信を拒否する。グリッパGoalは切替時の値に固定する。駆動は最大1秒、回転5軸の指令速度3度/秒、切替時からの位置差と往復を含むGoal総移動量は各軸1度以内。制限到達後は新しいGoalを送らず接続を閉じるが、保持トルクは残り得る。
