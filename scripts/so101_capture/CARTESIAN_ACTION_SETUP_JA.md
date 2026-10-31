# SO-101 FK／Cartesian Action解析：環境構築・使用手順

更新：2026-10-26 / 対象：macOS Apple Silicon、Python 3.12、LeRobot 0.6.2

## 0. 目的と現在の実装範囲

SO-101で収録済みの **2カメラ・53エピソード（20fps）** から、
各フレームの関節 state と関節 Action を順運動学（FK）で計算し、
TCPの差分（3次元位置・3次元回転）とグリッパ目標値を**別のParquetファイル**に出力する。

- 使用コード：`scripts/so101_capture/export_cartesian_action_sidecar.py`
- 元データ：`data/so101/muhi-side-wrist-success-053-20261025-168x224`
- 出力例：`data/analysis/so101_cartesian_action_53/`
- **元Dataset、画像、checkpoint、収録・実機制御コードは変更しない。**
- **まだACTが直接学習可能なDatasetではない。** データ拡張によるエピソード増加でもない。
- IKによる関節指令の復元、ACT再学習、ロボット実行、MoveIt連携は**未実装**。

本資料は現状のオフライン機能を対象とする。SO-101実機は接続・起動しない。

## 1. 重要：2つのConda環境を分ける理由

| 環境 | 用途 | NumPy |
| --- | --- | --- |
| `lerobot` | 従来のACT学習・実機収録・推論 | `2.2.6`（LeRobot 0.6.2は `numpy<2.3`） |
| `so101-ik` | FK／Cartesian Action解析 | Placo 0.10.2の依存先cmeel-boost 1.89.0が `numpy>=2.3,<2.4` を要求 |

**同一環境で `pip install placo` と `pip install numpy==2.2.6` を交互に行わない。**
既存 `lerobot` 環境ではNumPyを戻した後もPlacoの依存競合が残る場合がある。
`pip check` が警告を出した事実を隠さない。

重要：**エクスポータはPlacoを直接利用する方式に変更済み。LeRobotのインストールは不要。**
そのため、**PlacoとPyArrowを入れた専用環境で実行する。**
また、LeRobot 0.6.2のパッケージ要件とPlacoの依存要件は矛盾するため、
**専用環境で `python -m pip check` とFK実測照合が必要**。
以下では安全に専用環境を作り、必要な機能をチェックするところまでを案内する。
組合せが未解決なら元環境を改変せず、依存を切り離す実装変更を先に行う。

## 2. ACT側の既存環境を確認する（変更しない）

```bash
conda activate lerobot
python -V
python -c 'import numpy, torch, lerobot; print("NumPy:", numpy.__version__); print("Torch:", torch.__version__)'
python -m pip check
```

過去のログでは、`numpy==2.2.6` へ戻したあとPlaco依存に関する
`cmeel-boost` の競合警告が残っている。
**この警告だけでACTの実行不可と断定はできないが、依存関係は不整合である。**
本番用途の環境を修復する場合は既存依存の記録を取り、別途慎重に実施する。

## 3. FK／IK専用Conda環境を準備する

既存のACT環境はそのまま保持する。

```bash
conda create -n so101-ik python=3.12 -y
conda activate so101-ik
python -m pip install "placo==0.10.2" pyarrow
python -c 'import numpy, placo, pyarrow; print("NumPy:", numpy.__version__); print("Placo import OK")'
python -m pip check
```

変換スクリプトにLeRobotは不要になった。
次のPlaco読み込み確認を実施する。

```bash
python -c 'import placo; print("Placo import OK")'
```

`ModuleNotFoundError: placo` 等になった場合は**そこで停止**する。
`pip install lerobot` を無条件に実行しない（依存競合の可能性）。
この問題を解決するには、エクスポータをPlaco直接利用に改修するか、
検証済みの分離構成を別途用意する必要がある。

## 4. GitHubコードを取得する

```bash
cd ~/Documents/AI/physicalAI2026_docker/so101-imitation-learning
git status
git fetch origin
git switch feature/cartesian-action-offline-export
```

`git switch` に失敗する場合は、ローカル変更の有無を先に確認し、
未コミット変更を上書きしない。

単体テスト（ロボット接続・Placo不要）：

```bash
python -m unittest discover -s scripts/so101_capture -p 'test_export_cartesian_action_sidecar.py' -v
```

過去のMac実行では **5件成功**。ただし、これは模擬FKによる単体テストであり、
URDFによる実データ変換の成功は意味しない。

## 5. SO-101 URDFをダウンロードする

使用候補はTheRobotStudio SO-ARM100リポジトリ内の
`Simulation/SO101/so101_new_calib.urdf`。
ローカルに置き、元Datasetとは別管理する。

```bash
cd ~/Documents/AI/physicalAI2026_docker/so101-imitation-learning
mkdir -p data/urdf/so101
curl -fL \
  https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101/so101_new_calib.urdf \
  -o data/urdf/so101/so101_new_calib.urdf
ls -lh data/urdf/so101/so101_new_calib.urdf
head -n 5 data/urdf/so101/so101_new_calib.urdf
```

`curl -fL` が失敗したら保存先のファイル内容とHTTP応答を確認する。
URDFがメッシュを参照している場合、FKライブラリによっては関連する
メッシュ／assetも必要になる。

**キャリブレーション上の注意**：

- このURDFが実機のゼロ位置、関節符号、関節順序、TCPと一致するかは**未確定**。
- Codex解析では、J2/J3の一部Datasetフレームが上流URDFの関節範囲外。
- Datasetの関節値にhoming_offsetを二重加算しない。
- `gripper_frame_link` をTCPとして使う際は指先位置とのずれを検証する。
- 実機の安全な可動範囲を広げる目的でURDF限界を書き換えない。

## 6. 53エピソードのCartesian Actionを生成

**実行条件**：専用環境でPlacoのimportが成功し、
URDFの関節名・基準座標・FK結果が独立解析と一致することを確認済み。

```bash
conda activate so101-ik
cd ~/Documents/AI/physicalAI2026_docker/so101-imitation-learning

python scripts/so101_capture/export_cartesian_action_sidecar.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025-168x224 \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --output data/analysis/so101_cartesian_action_53 \
  --expected-episodes 53
```

出力先は**新規ディレクトリ**である必要がある。すでに存在する場合、
スクリプトは上書きを拒否する。元Datasetの内部への出力も拒否する。

期待される生成物：

```text
data/analysis/so101_cartesian_action_53/
├── cartesian_actions.parquet
└── manifest.json
```

注意：Parquetは **53エピソード／30,846フレームと一致するはず**だが、
FK・URDF整合性や実行環境の検証が済むまでは生成成功を保証しない。
生成前にエラーになった場合、元Datasetは変更されない設計である。

## 7. 結果確認

```bash
python - <<'PY'
import json
from pathlib import Path
import pyarrow.parquet as pq

p = Path("data/analysis/so101_cartesian_action_53")
meta = json.loads((p / "manifest.json").read_text())
table = pq.read_table(p / "cartesian_actions.parquet")
print("episodes:", meta["source_episodes"])
print("frames:", meta["source_frames"])
print("parquet rows:", table.num_rows)
print("schema:", table.schema)
print("first row:", table.slice(0, 1).to_pylist())
assert meta["source_episodes"] == 53
assert meta["source_frames"] == 30846
assert table.num_rows == 30846
PY
```

さらに重要な点として、位置・回転差分の分布を見て、
極端な値や関節制限外の指令を検出する必要がある。
**Cartesian Actionと画像の時間対応を変えず、元データと照合すること。**

## 8. ターゲットの意味を間違えない

各フレームの`observation.state[t]` と `action[t]` はそれぞれ
測定関節状態と関節の**絶対Goal**である。

変換後の7要素は：

```text
delta_x_m, delta_y_m, delta_z_m,
delta_rotvec_x_rad, delta_rotvec_y_rad, delta_rotvec_z_rad,
gripper_goal_0_100
```

`delta_tcp` は `FK(action[t]) - FK(state[t])` に対応する。
**隣のカメラフレームまでに手先が実際に動いた距離ではない**。
グリッパ値も絶対Goalである。

そのため、これは現時点では**新しい学習表現を調べるための教師ラベル候補**。
ACTを使ってそのまま学習・推論する機能は含まれない。
5自由度アームのため、任意の6次元手先姿勢をIKで実現できるわけでもない。

## 9. 失敗したときの確認先

| 症状 | 確認すること |
| --- | --- |
| `ModuleNotFoundError: placo` | `conda activate so101-ik` とインストール状況 |
| `ModuleNotFoundError: lerobot` | 古いブランチ上のコードの可能性がある。`git pull` と import 行を確認 |
| `pip check` NumPy競合 | `lerobot` と `so101-ik` の混在を解消する |
| URDF file not found | ファイル名・パス・curl実行結果 |
| FK frame / joint error | URDFの関節名と`gripper_frame_link`、基準座標を照合 |
| `FileExistsError` | 新しい出力ディレクトリ名を指定する（元出力を上書きしない） |
| FK位置がCodex解析と不一致 | キャリブレーション・関節順序・TCP基準の差。**学習へ進まない** |

## 10. Gitの取り扱い

解析CSV、生成したParquet、URDFなどは `data/` に保存する。
このリポジトリでは `data/` はGit管理対象外。

PR #2が検証完了するまでは、`main`へのマージを急がない。
実機制御の変更はこのPRに含めない。

## 11. 今後必要な別実装

1. Placo直接FKによる53エピソード変換は完了。独立FK照合が残る。
2. 実機キャリブレーションとURDFのFK一致検証。
3. side／wrist画像の時系列対応を保持した**新しい7D Action Dataset**の作成。
4. 7D ACTの学習・検証、IKデコーダ、shadow modeの構築。
5. 関節・速度・追従誤差・カメラ障害の安全ガードを通した実機検証。

現段階で「IKデータでACTの性能が改善した」と主張することはできない。

## 12. 実行記録（2026-10-26：53エピソードの変換成功）

### 12.1 STLメッシュが見つからない場合

URDFを単独ダウンロードしただけでは、次のエラーが出る場合がある。

```text
ValueError: Mesh assets/base_motor_holder_so101_v1.stl could not be found.
```

これはPlacoがURDF内の相対参照 `assets/*.stl` を解決できないため。
公式SO-ARM100リポジトリの `Simulation/SO101/assets` を、URDFと同じ階層へ配置する。

```bash
cd ~/Documents/AI/physicalAI2026_docker/so101-imitation-learning
mkdir -p data/urdf/so101

curl -fL https://github.com/TheRobotStudio/SO-ARM100/archive/refs/heads/main.zip -o /tmp/so-arm100-main.zip
unzip -oq /tmp/so-arm100-main.zip "SO-ARM100-main/Simulation/SO101/assets/*" -d /tmp/
cp -R /tmp/SO-ARM100-main/Simulation/SO101/assets data/urdf/so101/

ls -lh data/urdf/so101/assets/base_motor_holder_so101_v1.stl
```

期待する配置：

```text
data/urdf/so101/
├── so101_new_calib.urdf
└── assets/
    ├── base_motor_holder_so101_v1.stl
    └── （その他STLファイル）
```

配置後、次の手順で読み込みだけを試せる。

```bash
conda activate so101-ik
python - <<'PY'
import placo
robot = placo.RobotWrapper("data/urdf/so101/so101_new_calib.urdf")
print("URDF loading OK")
print(robot.joint_names())
PY
```

### 12.2 53エピソードの変換に成功したログ

実行コマンド：

```bash
conda activate so101-ik
python scripts/so101_capture/export_cartesian_action_sidecar.py \
  --dataset data/so101/muhi-side-wrist-success-053-20261025-168x224 \
  --urdf data/urdf/so101/so101_new_calib.urdf \
  --output data/analysis/so101_cartesian_action_53 \
  --expected-episodes 53
```

2026-10-26の実機未接続オフライン実行で、処理が例外なしに完了し、
以下の値を `manifest.json` 相当の標準出力で確認した。

| 項目 | 結果 |
| --- | --- |
| エピソード数 | **53** |
| フレーム数 | **30,846** |
| FPS | **20** |
| FK対象フレーム | `gripper_frame_link` |
| TCP平行移動差分の最大値 | **0.05614867966585311 m（約56.15 mm）** |
| TCP回転差分の最大値 | **0.22448111425603512 rad（約12.86°）** |
| `trainable_leRobot_dataset` | **false** |

最大値は `FK(action[t])` と `FK(state[t])` の**同時刻のGoal差分**であり、
ロボットが隣接するフレーム間で実際に移動した距離・回転角ではない。
これらの値だけでは変換の幾何学的正確さ、データ品質、把持精度は判断できない。

出力先：

```text
data/analysis/so101_cartesian_action_53/
├── cartesian_actions.parquet
└── manifest.json
```

### 12.3 Placoの自己干渉警告について

実行時に以下の警告が出たが、今回の**FK変換自体は完了した**。

```text
WARNING: Robot has the following self collisions in neutral position:
 -base_link_2 collides with shoulder_link_2
 -shoulder_link_0 collides with upper_arm_link_1
 -upper_arm_link_0 collides with lower_arm_link_0
 -lower_arm_link_2 collides with wrist_link_1
 -wrist_link_0 collides with gripper_link_1
 -gripper_link_0 collides with moving_jaw_so101_v1_link_0
```

これは**URDFの中立姿勢における衝突形状の判定**であり、
実際のSO-101がこれらの箇所で物理的に干渉していると断定するものではない。
隣接リンクの接触やコリジョンメッシュ設定に由来する可能性もあるが、
**原因は未確認**。一方、衝突安全性が確認できたという意味でもない。

- 今回は読み取り専用FKなので、警告があってもデータ出力は完了できた。
- **IKで生成した軌道を実機へ送る前には、衝突形状・関節制限・姿勢・速度の確認が必須。**
- 安全確認前に自己干渉判定を無効化したり、警告を無視して実機を動かさない。

### 12.4 生成物の検証

```bash
conda activate so101-ik
python - <<'PY'
import json
from pathlib import Path
import pyarrow.parquet as pq

p = Path("data/analysis/so101_cartesian_action_53")
meta = json.loads((p / "manifest.json").read_text())
table = pq.read_table(p / "cartesian_actions.parquet")
assert meta["source_episodes"] == 53
assert meta["source_frames"] == 30846
assert table.num_rows == 30846
assert meta["trainable_leRobot_dataset"] is False
print("Validated:", meta["source_episodes"], "episodes,", table.num_rows, "rows")
print("First row:", table.slice(0, 1).to_pylist())
PY
```

これは件数・構造の確認。**独立FKとの照合は未実施**であり、
URDFの原点・関節符号・TCP座標系の一致は引き続き検証が必要。

### 12.5 現在どこまで進んだか

**完了**：依存環境の分離、URDF＋STLの準備、5件の単体テスト成功、
53エピソード・30,846フレームのCartesian Actionサイドカー変換。

**未完了**：FK結果と独立解析の一致検証、ACT用7次元Datasetの生成、
Cartesian Action版ACTの学習、IKデコーダ、衝突検証、実機での安全確認。

したがって、今回の成果は**「IKでデータを追加した」ではなく「FKで既存データの別表現を生成した」**段階である。
