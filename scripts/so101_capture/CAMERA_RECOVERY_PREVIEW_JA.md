# SO-101 カメラ待機改善・side/wristプレビュー

## 対象と制約

- 独立ブランチ `feature/camera-recovery-dual-preview`。元Dataset、ACT、モーター制御ガードは変更しない。
- CameraRigのペア取得待機上限を50msから**120ms**へ変更。実カメラ撮影時刻ではなくホスト利用可能時刻で照合。
- 鮮度100ms、ホスト到着時刻差75ms、未使用新規フレーム要求は維持する。
- 120ms以内に新鮮なペアを得られないときは従来どおりラッチ停止。**FAULT後の自動再開・一時停止状態からの復旧は今回未実装**。
- 長い待機は制御周期を伸ばす可能性がある。250ms制御ギャップ等の既存ガードを弱めない。
- previewは**専用プロセス**でカメラを開く。AVFoundationのwristを既存record_episodeやACTと**同時起動しない**。
- OBSにはsideのiPhone6_Fixedだけを残してよい。wristはPython直接取得。
- 画像の表示はロボット指令と無関係。出力の緑/橙は画像のホスト鮮度の目安であり、光学露光からの遅延を保証しない。

## カメラだけを表示

既存収録・ACTプログラムを終了し、OBSとiPhone DroidCamを起動してから:

```bash
conda activate lerobot
cd ~/Documents/AI/physicalAI2026_docker/so101-imitation-learning
export PYTHONPATH="$PWD/.local/camera-deps:$PWD/.local/act-deps:${PYTHONPATH:-}"

python scripts/so101_capture/preview_side_wrist_live.py \
  --camera-config scripts/so101_capture/cameras.local.yaml \
  --enable-cameras side,wrist \
  --preview-fps 5
```

2画面の別ウィンドウが表示される。**q**かEscで終了する。
本ツールはシリアル通信も関節Goal送信も行わない。
Macのカメラアクセス許可が必要な場合がある。

## テスト

```bash
python -m unittest discover -s scripts/so101_capture -p 'test_preview_side_wrist_live.py' -v
python -m unittest discover -s scripts/so101_capture -p 'test_multi_camera.py' -v
```

## 実機収録検証

通常の収録コマンドをそのまま使う。新しいペア待機上限は内部処理のみに適用される。
変更前と比較し、`capture_timing.json` のカメラ待機時間、control interval P95/P99、
stale/pairing fault数、20Hz実効レートを確認する。

**注意**: 今回は「失敗時に収録を一時停止して再開」する機能ではない。
それには撮影区間の切断・分割、サーボ保持の実測確認、操作承認付きの明示再開が必要。
学習用エピソードのtimestampや20fps制約を崩さずに実装する必要がある。
