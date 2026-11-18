# Phase 6G｜J4を逆に曲げない把持：両爪同時接触の局所姿勢最適化（2026-11-18）

## 背景
前回の18候補IK探索で選択されたJ4非負の姿勢は J4=+38.22°、J2=−89.46°。開爪時の固定爪―ムヒ距離2.93mmに対し、可動爪―ムヒ距離31.59mm。物理保持中の両爪同時接触は0/500で、把持できなかった。一方、従来middle（J4=−94.99°）は500/500だが最大XY変位27.55mmで、持ち上げは未検証。

## 今回追加した処理
- GitHub: `scripts/so101_sim/j4_bilateral_contact_refinement.py` と `phase6g_fixed_workspace_grasp_depth.py`
- 18候補は初期スクリーニングのまま残し、上位最大4候補についてJ4≥0°の制約を保った**関節角の局所最適化**を追加。
- 固定爪と可動爪の、**同一J6閉鎖率88%または100%におけるムヒとの符号付き最短距離**を同時に評価。片側だけが近い状態を改善目標にしない。
- MuJoCo `mj_geomDistance` の世界座標系の最近接点を使い、円筒軸中心から見た左右接触方向の余弦（−1に近いと反対側、+1に近いと同じ側）を評価。ただし接触面の物理法線を直接使った厳密な最適化ではなく、**近接点の幾何ヒューリスティック**。
- 開爪時の物体との貫入を罰則化。選択した姿勢は同一のJ6動力学で動画、保持中両爪接触回数、XY変位を評価する。
- `j4_bilateral_refinement.json`（4シードの補正前後、隙間、半径方向の余弦、選択理由）、`j4_bilateral_refinement.csv`。
- 新しい専用Colab Notebook: [SO101_Phase6G_J4_BilateralRefinement_20261118.ipynb](https://drive.google.com/file/d/1qLBaTVz6iLnv0lemfCG1C19YPQIsUaQB/view)。Notebookにスコア・両爪の距離の解説付きグラフと2条件の比較動画を追加。

## 固定する条件
- 従来 `middle` の関節角、IK、物理シミュレーションの実行条件は変更しない。
- ムヒの初期世界座標、表示用STL、衝突用12×3メッシュ、摩擦係数、J6時間設定は変更しない。
- **SIM-ONLY。実機を操作しない。**
- `--no-j4-bilateral-refine`で前回の18候補選択を復元可能。変更前バックアップブランチ `backup/phase6g-before-j4-bilateral-refinement-20261118`。

## 判定
`preflight_paired_contact` は幾何予測であり、**物理的な把持成功を意味しない**。最優先は新しい`middle_j4_nonnegative/grasp.mp4`と`comparison.csv`の`bilateral_hold_steps`、`max_xy_displacement_mm`。特に「両爪接触0なのに移動が小さい」を成功と判定しないこと。

新しいMuJoCo実行はまだ行っていない。Colabで結果取得後、成功／失敗を判断する。
