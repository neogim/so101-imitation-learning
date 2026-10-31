# 次の推奨作業：120フレームの不連続位置を特定する（オフライン）

現時点の確定結果：フレーム0〜119、肩＋肘＋ DP2は2/6合格（x−、z+）、肩−肘− DP2は6/6合格。両仮説で全720/720のIK到達候補あり。120フレームを再実行する必要はない。全仮説ANDのゲートはSTOPなので648フレームへまだ進まない。

## Macで実行

```bash
git switch feature/cartesian-action-offline-export
git pull --ff-only origin feature/cartesian-action-offline-export
python -m unittest discover -s scripts/so101_capture -p 'test_diagnose_dp_discontinuities_offline.py' -v
python scripts/so101_capture/diagnose_dp_discontinuities_offline.py \
 --paths outputs/so101-ik-dp-000-119-stage-v2/selected_paths.json \
 --mapping 'shoulder+_elbow+' --method dp_second_order \
 --output outputs/so101-ik-dp-000-119-stage-v2/diagnostics_plus_plus.json
```

結果は方向ごとの閾値超過イベント（jump>5°、速度>100°/s、加速度>2000°/s²）の先頭12件を記載する。特にx+、y+、y−、z−で最初の違反フレーム、関節、seed名、関節制限への接触の有無を確認する。

この簡易診断はデータセットの20Hz一定間隔を使った有限差分であり、既存の集計と丸めや微小な時刻差によって数値が異なることがある。姿勢・残差・偏差の判定は元のsummary.jsonが正。新しいIK候補は生成しない。実機操作・ACTデータ変更はしない。

## 判定

不連続原因が特定できるまで、648フレーム計算は実行しない。肩−肘−は数値的に有望だが、実機角度の正しい符号かどうかは全く未確定。過去にPlacoでニュートラル姿勢自己衝突の警告あり。実機での動作可能性や安全性に読み替えない。

コンパクトな出力をChatGPTに貼り付け、問題フレームの原因候補を切り分けた後、肩−肘−単独の長区間評価に移る。
