# SO-101 DP2 ヤコビアン診断 → 648フレームへの進行判定（2026-10-28）

## 目的と現在の確定結果
120フレームのDP2は、仮説++で2/6合格（x−, z+）、仮説--で6/6合格。++の失敗は主にフレーム51〜99の wrist_roll と shoulder_lift のジャンプ・速度・加速度超過。以前の詳細診断により、x+ f75→76, f78→79, f79→80、y+ f87→88, f97→98、y− f54→55, f63→64、z− f51→52, f73→74 などを確認した。

次は**位置ヤコビアン3×5**のSVDで、数値的な特異値の小ささが不連続と同じ位置・時刻で発生するかを調べる。これは原因候補の相関確認であり、物理的な特異点を確定するものではない。衝突・トルク・実機角度符号の確認ではない。

## Macでの実行（Codex不要）

```bash
git switch feature/cartesian-action-offline-export
git pull --ff-only origin feature/cartesian-action-offline-export
python -m unittest discover -s scripts/so101_capture -p 'test_audit_dp_jacobian_offline.py' -v
python -m py_compile scripts/so101_capture/audit_dp_jacobian_offline.py
```

問題の4方向について、同じ保存済みDP2軌道を使用。新しいIK計算は不要。出力先は必ず未使用のパスにする。

```bash
for direction in 'x+' 'y+' 'y-' 'z-'; do
  safe=${direction//+/_plus}; safe=${safe//-/_minus}
  python scripts/so101_capture/audit_dp_jacobian_offline.py \
    --paths outputs/so101-ik-dp-000-119-stage-v2/selected_paths.json \
    --urdf data/urdf/so101/so101_new_calib.urdf \
    --mapping 'shoulder+_elbow+' --direction "$direction" \
    --start 50 --end 100 \
    --output "outputs/so101-ik-dp-000-119-stage-v2/jacobian_${safe}_50_100.json"
done
```

各方向で最小特異値の最小フレームと、暫定の閾値1e-5 m/rad未満となるフレームを表示する。方向ごとの結果をChatGPTへ貼り付けてよい。必要に応じ、既存のdiagnostics_plus_plus.jsonの違反フレームと付き合わせる。**閾値1e-5 m/radは探索的な数値定義であり、特異点や機械性能の物理的な閾値ではない**。SVDの最大条件数、特異値の連続的な変化、姿勢の揺らぎなどを解釈するには追加の比較が必要。

## 推奨する次段階と停止条件
1. 4方向のJacobian結果を回収し、数値特異性がwrist_roll/shoulder_liftの急変と時間的に重なるか判定する。位置のみIKでは手首姿勢が拘束されていない点に注意し、3x5ヤコビアンの条件数だけでwrist_rollの変動理由を断定しない。
2. それでも仮説++の4/6失敗は継続するため、全仮説AND条件による648フレームの拡張はSTOP。仮説--は単独で120フレーム6/6だが、仮説の物理的正しさは未確認。次回の拡張は**--だけ**を対象とした事前登録済みのオフライン数値評価として別タスクで設計し、初期フレームの一部分で先にゲートを確認する。
3. 許可されていない作業：ロボット接続、モータ操作、URDF・校正値の改変、ソースデータ変更、ACT学習データ生成、648フレームの無条件一括実行。

保存済みデータとGitHubコードだけで調査できる。ローカル出力（selected_paths、calibration）はgitignoreに保持してアップロードしない。
