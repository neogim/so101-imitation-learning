# SO-101 Side｜ChArUco 40枚の解析記録と次工程（2026-11-07）

## この資料の位置づけ

既存のSideカメラ固定（iPhone 6→OBS）、640×480のChArUco撮影 **25枚＋15枚**を解析。**内部パラメータは暫定候補であって、確定済みではない。Eye-to-Handは未解決。** 実験結果は軽量なJSON・CSVをGit管理し、元画像40枚は別保管とする。

### 関連ファイル

- [40枚の診断JSON](../assets/calibration/side_charuco_20261107_40frame_diagnostic.json)
- [40枚の検出数CSV](../assets/calibration/side_charuco_20261107_detection_counts.csv)
- [制約付き内部パラメータ暫定候補](../assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json)
- [ZIPから再計算するスクリプト](../scripts/so101_capture/audit_charuco_two_sessions.py)
- [ArUco 2解候補のPnP診断](../scripts/so101_capture/audit_aruco_pnp_ambiguity.py)

## 画像の出自・再現性

| セッション | 元ZIP（Git未格納） | 撮影枚数 | 10点以上検出 |
|---|---|---:|---:|
| 初回 | `so101-charuco-stills-timed-02.zip` | 25 | 22 |
| 画面周辺の追加 | `so101-charuco-stills-edges-01.zip` | 15 | 15 |

元ZIPのSHA-256は診断JSON内に保存。**ZIPと40枚のPNG自体はコミットしていない**。バイナリがない状態では解析値をゼロから再計算できないため、元ZIPをローカルで保全すること。

初回の除外：`side_charuco_010.png` 0点、`022` 2点、`024` 0点。計37枚を受理。OpenCV 4.13.0／`DICT_4X4_50`／7×9マス／20mm/14mmボード。セッション順→ファイル名順に並べ、受理画像の5枚ごとに1枚を保持用とした（学習30枚、保持7枚）。

### 40枚の再解析コマンド（ローカルで元ZIPを用意した場合）

```bash
conda activate lerobot
python scripts/so101_capture/audit_charuco_two_sessions.py \
  --first-zip /path/to/so101-charuco-stills-timed-02.zip \
  --edge-zip /path/to/so101-charuco-stills-edges-01.zip \
  --out-dir outputs/charuco-20261107-cross-session-check-01
```

`/path/to/` は実際の場所に置き換える。結果：`combined_report.json`。以前の結果と比較する際にはOpenCVバージョンも記録し、画像処理条件が変わっていないか調べる。

## 数値結果（単位px）

| モデル | 学習RMS | 保持RMS | fx/fy | cx/cy |
|---|---:|---:|---|---|
| 標準5係数 | 0.28893 | 0.45009 | 541.916 / 541.956 | 324.663 / 222.589 |
| k3固定・接線歪み0 | 0.30154 | 0.45204 | 543.173 / 542.957 | 316.395 / 229.548 |
| 旧チェッカーボード16枚 | 約0.315 | 比較不能 | 約540.17 / 539.19 | 約317.35 / 231.55 |

前回25枚だけで5係数を学習し今回15枚へ適用するとRMS約2.510px、一方k3固定・接線歪み0なら約0.418px。**旧25枚に対する歪みモデルの過適合が疑われる**が、撮影ごとのOBS／光学条件の同一性は未検証。

両モデルの合計37枚の保持RMSはほぼ同じ。したがってより制約したモデルを**PnP診断の暫定入力**として用いる。外部校正へ採用確定した意味ではない。固定条件を変えると主点が動く理由も検証対象。

## 次工程：ArUco 22静止姿勢のPnP診断

対象：`outputs/so101-aruco-calibration-20261108/manifest.json`（Mac上の既存ローカル出力）。Marker 14mmの四隅と、暫定K／歪みから `SOLVEPNP_IPPE_SQUARE` の**2つの姿勢解**を出し、距離・再投影RMS・両解のRMS差をIDごとに確認する。

```bash
git pull --ff-only
mkdir -p outputs/so101-aruco-pnp-audit-01

python scripts/so101_capture/audit_aruco_pnp_ambiguity.py \
  --manifest outputs/so101-aruco-calibration-20261108/manifest.json \
  --intrinsics assets/calibration/side_charuco_20261107_restricted_intrinsics_candidate.json \
  --marker-mm 14 \
  --out outputs/so101-aruco-pnp-audit-01/report.json
```

- 全てローカルファイルの**読み取り専用**（結果だけ新規作成）。ロボットを動かさない。
- 小さなArUcoは平面PnPの向きが不安定になり、再投影誤差が低くても真の3D姿勢は不正確なことがある。**2解の差が小さいと曖昧**。
- ID0/1（gripper_link）、ID2（wrist_link）は仮対応。3/4/5は未確認リンクにつきEye-to-Handの基準にしない。
- ChArUcoによる内部校正とEye-to-Handを混同しない。PnPの結果確認後、ロボットFKの符号／原点、剛体取付、リンク→マーカー外部変換とBase→Side変換を分離する推定へ進む。
- 本実験結果はGit管理するが、**元ZIPと元のArUco `manifest.json` は公開していないため、リポジトリ単体で生データ再現はできない**。

## 未解決・合否判定

- [x] 40枚での内部校正RMS／主点の変化を数値化
- [x] 個別検出数、元ZIP SHA-256、2モデルをGitに記録
- [x] 暫定Kを明示してPnP診断コードを用意
- [ ] OBS設定／画角／焦点が2セッションとArUco録画で同じと確認
- [ ] ボードの実測20mm・平面性を確認
- [ ] ArUco 22姿勢のPnP（2解）診断をMacで実行・レビュー
- [ ] LeRobot角とURDFのゼロ点・符号を物理検証
- [ ] 外部校正＋未使用姿勢の再投影評価＋MuJoCo実写比較
