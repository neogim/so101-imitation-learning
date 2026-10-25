# Leader-Follower から ACT への引継ぎ案（実機未接続）

推奨は B: 正常に動作中の収録セッションを同一プロセス・同一シリアル接続のまま維持し、Leader Goal の送信権を停止してから ACT に渡す方式。既存の `start_hardware()` と `synchronize_positions()` を一度だけ使用し、引継ぎ時には `configure()`、Goal 再同期、トルク切替を実行しない。ただし最初の起動には、原因未解明の予期しないトルク ON リスクが残る。Leader/Follower が一致しても、カメラ、対象物、学習デモの初期姿勢が一致するとは限らない。

現状は設計と合成テストのみ。`teleop_act_handoff.py` はシリアルポートを開かず、実機ランチャーにもまだ接続していない。`record_episode.py` を `q` で終えると `Session.stop()` が新しい保持 Goal を送る場合があり、さらに接続を閉じる。`teleop_only.py` は終了時にトルクを無効化する。いずれも終了後に別プロセスで ACT を開始する方式には使わない。

同じ20 Hz制御スレッドからだけ単一の Goal writer を呼ぶ。`TELEOP` から `CHECK` に移した瞬間に Leader/ACT の両方の書込みを禁止する。未処理Goalがないこと、較正の一致、通信健全性、100 ms以内の6軸Goal/Present/Torque/Status、Torque=1、Status=0、較正範囲、Goal/Present誤差<=8、Goal読戻しと直前送信Goalの一致を確認する。ACTガードの初期Goalには実際に読み戻したGoalを使い、初回も30単位/秒・20 Hzなら各軸最大1.5単位の変化に制限する。通信異常・追従誤差超過・確認失敗はラッチして新しいGoalを送らず、通信が戻っても自動復帰しない。

実機統合時は `HardwareIO.send()` への全経路をこの単一writerへ集約し、`Session` の `TELEOP_RESET` から専用の切替状態へ遷移させる。既存の `SAFE_RESET` への遷移は保持Goalを書き込むため、引継ぎの直前に使わない。ACT推論は `CHECK` 中に実行してもよいが、書込みは検査合格後のみ。実機移行前に切替チェックポイントをJSONへ記録し、先行するLeader指令の送信停止と未処理キューゼロを実測する。カメラの向き・前処理、6軸の順序と単位、学習初期姿勢も別途確認する。

比較: A は収録の既存起動コードをACTランチャーから再利用できるが、FollowerトルクOFF→Goal一括書込み→`configure()`の経路を通る。`configure()`中の例外でもLeRobotの `torque_disabled()` はfinallyでトルクONを試みる。Bはこの起動経路を一度通るが、動作中の引継ぎで新たなGoal同期やトルク切替を要しない。Cの独立した一括Goalプローブは、書込みと読戻しの検証には役立つが、ACT起動前に追加のGoal書込みを生み、初期化後のトルク状態も別途確認する必要がある。単軸Goal書込み後の予期しないトルクONは未解明であり、A/B/CのいずれもGoal書込み時のトルク変化を排除できない。

次の最小実機試験は別途承認後に、腕を支え物理電源遮断手段を用意して、既存のLeader-Follower起動・Alignmentを行い、ACT送信を一切有効化しない `CHECK` への切替だけを試す。Leader指令が停止したこと、6軸の生Goal/Present/Torque/Status、較正値、Goal/Present誤差、通信状態、観測画像とACT推論結果を記録する。異常時はGoal/トルクの追加書込みなしにラッチして終了する。ソフト停止後も保持トルクは残り得るので、腕を支えた上で独立した物理電源遮断手段を使用する。ACTへのGoal送信試験はさらに別の承認段階とする。
