# SO-101 Leader–Follower J2/J3 実機マッピング検証：安全な段階実施

## 対象と目的
ローカル `lerobot` conda環境で、既存校正ファイルを維持したままJ2/J3の`q_URDF = s*q_LeRobot + b`を検証する。既存データEpisode 0 の数値DP 648f 6/6 PASSは実機用マッピングの証明にはならない。

### フェーズA：読み取り専用事前点検
Followerを机へ固定し、落下防止の支持をつける。可動域から容器・ムヒ等を除き、非常停止/電源遮断を確保する。別のテレオペ/ROS/ACT実行を停止しシリアルポートの二重接続を避ける。
```bash
conda activate lerobot
git pull --ff-only origin feature/cartesian-action-offline-export
python -m py_compile scripts/so101_capture/preflight_leader_follower_mapping_readonly.py
python scripts/so101_capture/preflight_leader_follower_mapping_readonly.py \
  --output outputs/so101-leader-follower-preflight-v1.json
```
出力は各関節のLeader・Follower度数、差分、トルク状態、校正ファイルSHA256。**完全読み出しだけ**で`configure()`、`send_action()`、`Goal_Position`、トルクON/OFF、校正書き込みをしない。既存校正一致しなければ停止。失敗時も接続を閉じる（`disable_torque=False`）。

**差が小さいというだけでテレオペ開始を許可しない。** 実機の姿勢と接触リスクの確認、開始時の位置合わせ処理・トルク切替の点検が必要。

### フェーズB：既存の動作制御を別途レビュー
`record_episode.py` の `start_hardware()` はキャリブレーション確認後にLeader/Followerをconfigureし、FollowerにGoalを書き込んで位置合わせする。つまりこの関数を呼ぶことは**read-onlyではない**。特に従来の初期姿勢差上限は関節120度/グリッパー100であり、小振幅の校正試験用安全限界として利用してはならない。`synchronize_positions()` は位置合わせで`Goal_Position`を繰り返し書き込む。いきなり実機へ適用しない。

実機運動開始前の停止条件例：
- Leader/Followerの初期姿勢差が小さく、安全な開始姿勢を確保できない場合
- トルク状態が不明・接続中エラー・校正不一致
- 可動域/アームの固定/緊急停止方法を確認できない場合
- Leader動作で複数関節が想定外に動く場合
- 実機に追従誤差・引っ掛かり・発熱が発生した場合

### フェーズC：操作承認後のみ
既存の実績ある安全なLeader–Followerテレオペを低速・短時間に限定し、J2を単独で少し動かし停止、次にJ3で同じことをする。制御中に別のシリアル読取ツールを起動しない。トルクON/OFFや追従を自動化する新コードはこの変更に含まない。記録前後のFollower観測角と同じ時刻の実機sideカメラ像/正方向の動きを記録。LeRobot角度だけからURDF正方向を断言しない。

### フェーズD：オフライン解析
2つ以上の既知姿勢を使ってJ2/J3符号を物理的に照合。ゼロオフセットは1姿勢から推定できても、角度基準と関節中心の幾何を検証していなければ採用しない。安全な実機FK再生や学習データ書き換えは禁止のまま。
