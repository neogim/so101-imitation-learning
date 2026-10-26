# ACTグリッパ独立評価と把持フェーズ強化の実験計画

## 現状
7D Cartesian ACT、53成功エピソード30,846フレーム、45 train／8 validation。10-step smoke modelの独立評価は160サンプルでgripper MAE=12.77、GT>25が34件、予測>25が0件。これは**10-step初期モデルの評価**であり20k-stepモデルの学習失敗を意味しない。

## 1. 学習前のラベル診断
```bash
conda activate lerobot
git switch feature/cartesian-action-offline-export
git pull --ff-only origin feature/cartesian-action-offline-export
python scripts/so101_capture/audit_gripper_grasp_windows.py \
  --dataset data/so101/muhi-side-wrist-cartesian-smoke-053 \
  --output data/analysis/gripper_grasp_windows_audit
```
出力`gripper_episode_audit.csv`に各エピソードのgripper>25フレーム割合、初回>25位置、前後3秒の候補区間。>25は「把持成功の瞬間」ではなく、あくまで**教師指令値**の暫定閾値。目視または成功メタデータとの照合が必要。

## 2. 学習済みモデルのグリッパ独立評価
```bash
python scripts/so101_capture/evaluate_cartesian_act_offline.py \
  --dataset data/so101/muhi-side-wrist-cartesian-smoke-053 \
  --checkpoint outputs/act_cartesian_smoke10/checkpoints/000010/pretrained_model \
  --output outputs/act_cartesian_eval_smoke10_v2 \
  --stride 20 --limit 160
```
`metrics.json`内の`gripper_diagnostics`から、物理単位での独立MSE・MAE・RMSE・bias・教師指令>25の検出Recall・グリッパの学習統計標準偏差で規格化したMSEを取得。これらはACTトレーニング時のL1＋KLの内訳とは異なり**推論出力に対するグリッパ単独誤差**。

グラフは`python -m pip install matplotlib`を先に実行する（未導入の場合）。既存出力ディレクトリは上書きしない。

## 3. 改善案の検証順序
A. 同じDataset、正規化統計、学習設定のまま20k step等の長い学習を実施し、同じ検証160点で評価。10step checkpointと比較。まず学習不足なのかを検証する。

B. グリッパ閉鎖直前・閉鎖中の区間を重点サンプリングする場合、画像・state・Actionの時刻対応を維持し、**学習側45エピソードだけ**から抽出する。既存デモの切り出しは新しい独立サンプルを増やすわけではなく分布を再重み付けすること。検証側8エピソードは変更しない。

C. 把持専用ACT：接近→位置合わせ→閉鎖→短い持ち上げまでを連続デモとして学習する。区間開始条件や終了条件、必要な画像履歴（n_obs_steps等）とAction chunk50による後続ラベルの欠落を扱う必要がある。単に閉鎖フレームのみで学習しない。グリッパ閾値の自動切り出し結果を**必ず動画で目視確認**してから、新たなtrain-only Datasetを作る。

D. 学習目的関数のグリッパ加重はB/Cを評価した後、ACTの実装で正規化後のAction次元別L1とVAEのKLを別々に計測してから検討。標準化済みグリッパ誤差に過度な重みを付けると位置・姿勢の精度が劣化しうる。

## 4. 評価の限界
- 本コードはACT予測chunkの**先頭1 action**をteacher-forcedで比較。ロボット実行キューや閉ループの評価ではない。
- 10-stepモデルの結果だけではグリッパの改善方法は確定しない。
- 7D ActionはFKから生成したラベル変換で、IKデータ拡張ではない。
- 5自由度のSO-101では任意の6D手先姿勢をIKで再現できない。実機投入は引き続き禁止。
