# Conda環境構築メモ

調査日: 2026-10-31（Asia/Tokyo）

対象: Apple Silicon Mac（`arm64`）、Miniforge `/Users/ishikawakazuhiko/miniforge3`

この文書は、ローカルに存在する `lerobot`、`so101-ik`、`so101-mujoco` を実際に調査したスナップショットである。Condaの作成履歴、インストール済みパッケージ、主要モジュールのimport、`pip check`を確認した。

> CondaはConda操作の履歴を残すが、過去の `pip install` コマンドは記録しない。そのため、下記の「確認できた履歴」は実記録、「再構築例」のpipコマンドは現在のインストール状態から復元したものである。

## 一覧

| 環境 | 作成日 | Python | 主用途 | 中心となるパッケージ |
|---|---:|---:|---|---|
| `lerobot` | 2026-08-22 | 3.12.13 | 実機操作、データ収集、学習 | LeRobot 0.6.2、PyTorch 2.11.0、TorchVision 0.26.0、OpenCV、Placo/Pinocchio |
| `so101-ik` | 2026-10-26 | 3.12.15 | 逆運動学、可視化、動画処理 | Placo 0.10.2、Pin 3.8.0、Meshcat、SciPy、FFmpeg 8.1.2 |
| `so101-mujoco` | 2026-10-31 | 3.11.18 | MuJoCoシミュレーション | MuJoCo 3.17.0、NumPy 2.4.6、GLFW、PyOpenGL |

## `lerobot`

### 確認できた構築履歴

```sh
conda create -y -n lerobot python=3.12
```

LeRobotはPyPIの通常インストールではなく、次のローカルcheckoutをeditable installした状態になっている。

- checkout: `/Users/ishikawakazuhiko/workspace/lerobot`
- origin: `https://github.com/huggingface/lerobot.git`
- commit: `223a8ad16c52dad961cc1104477ffc3369c5189a`
- package version: `0.6.2`

### 主なインストール内容

| 分類 | パッケージ |
|---|---|
| 学習 | `torch==2.11.0`、`torchvision==0.26.0`、`accelerate==1.15.0`、`safetensors==0.8.0` |
| LeRobot | `lerobot==0.6.2` (editable)、`draccus==0.11.6`、`feetech-servo-sdk==1.0.0`、`pyserial==3.5` |
| データ | `datasets==4.8.5`、`pandas==2.3.3`、`pyarrow==26.0.0`、`av==15.1.0` |
| 画像 | `opencv-python-headless==4.13.0.92`、`Pillow==12.3.0` |
| 数値計算 | `numpy==2.2.6`、`einops==0.8.2`、`sympy==1.14.0` |
| ロボティクス/IK | `placo==0.10.2`、`pin==3.8.0`、`meshcat==0.3.2` |
| 通信/Hub | `huggingface-hub==1.28.0`、`requests==2.34.2`、`aiohttp==3.14.5` |

Python distributionは114件確認された（editable install由来で `lerobot` のメタデータが重複して見える場合がある）。主要モジュールのimportは成功した。PyTorchはMPS対応ビルドだが、調査時のプロセスでは `torch.backends.mps.is_available()` は `False` だった。

### 再構築例

```sh
conda create -y -n lerobot python=3.12
conda activate lerobot
git clone https://github.com/huggingface/lerobot.git ~/workspace/lerobot
git -C ~/workspace/lerobot checkout 223a8ad16c52dad961cc1104477ffc3369c5189a
python -m pip install -e ~/workspace/lerobot
python -m pip install placo==0.10.2
```

このリポジトリ固有のカメラ依存関係などは、ルートの `README.md` にある `.local/camera-deps` と `.local/act-deps` の手順も参照する。

### 注意事項

`python -m pip check` は次の不整合を1件報告した。

```text
cmeel-boost 1.89.0 requires numpy>=2.3,<2.4, but numpy 2.2.6 is installed
```

一方、LeRobot、Torch、TorchVision、OpenCV、Placo、Pinocchioのimport自体は成功した。NumPyだけを安易に更新すると学習側に影響する可能性があるため、用途別環境を維持するか、変更前に学習・IKの両方をテストする。

## `so101-ik`

### 確認できた構築履歴

```sh
conda create -n so101-ik python=3.12 -y
conda install -c conda-forge ffmpeg
```

### 主なインストール内容

| 分類 | パッケージ |
|---|---|
| IK/ロボティクス | `placo==0.10.2`、`pin==3.8.0`、`libpinocchio==3.8.0`、`eigenpy==3.12.0` |
| 可視化 | `meshcat==0.3.2`、`matplotlib==3.11.2`、`pyngrok==9.0.3` |
| 数値計算 | `numpy==2.3.5`、`scipy==1.18.1` |
| データ/補助 | `pyarrow==26.0.0`、`PyYAML==6.0.3`、`ipython==9.17.1` |
| 動画 | `ffmpeg==8.1.2`（Conda/conda-forge系依存を含む） |

Python distributionは56件。`numpy`、`scipy`、`placo`、`pinocchio`、`meshcat`、`matplotlib` のimportは成功し、`pip check` も問題なしだった。

### 再構築例

```sh
conda create -n so101-ik python=3.12 -y
conda activate so101-ik
conda install -c conda-forge ffmpeg
python -m pip install \
  placo==0.10.2 \
  numpy==2.3.5 \
  scipy==1.18.1 \
  matplotlib==3.11.2 \
  meshcat==0.3.2 \
  ipython==9.17.1 \
  pyarrow==26.0.0
```

`placo` がPinocchio/CMeel系の依存パッケージを導入するため、通常は `pin` や `eigenpy` を個別に追加する必要はない。

## `so101-mujoco`

### 確認できた構築履歴

```sh
conda create -n so101-mujoco python=3.11 -y
```

### インストール内容

この環境は小さく、Python distributionは14件だった。

```text
absl-py==2.5.1
etils==1.14.0
fsspec==2026.9.0
glfw==2.11.0
mujoco==3.17.0
numpy==2.4.6
packaging==26.3
PyOpenGL==3.1.10
typing_extensions==4.16.0
websockets==17.4
zipp==4.1.2
```

上記に加えて環境管理用の `pip==26.2.2`、`setuptools==84.2.1`、`wheel==0.48.0` が入っている。`mujoco`、`numpy`、`glfw`、`OpenGL` のimportは成功し、`pip check` も問題なしだった。

### 再構築例

```sh
conda create -n so101-mujoco python=3.11 -y
conda activate so101-mujoco
python -m pip install mujoco==3.17.0 numpy==2.4.6
```

`glfw`、`PyOpenGL`、`absl-py` などは `mujoco` の依存として導入される。

## 検証コマンド

環境の現状を再確認するときは以下を使う。

```sh
conda env list
conda list -n lerobot
conda list -n so101-ik
conda list -n so101-mujoco

conda run -n lerobot python -m pip check
conda run -n so101-ik python -m pip check
conda run -n so101-mujoco python -m pip check
```

完全な再現用ファイルが必要な場合は、用途に応じて次を出力する。

```sh
# 同じOS/CPUでの厳密な再現向け
conda list -n ENV_NAME --explicit > ENV_NAME-explicit.txt

# 人が読みやすく、別マシンへの移植にも使いやすい形式
conda env export -n ENV_NAME --from-history > ENV_NAME-history.yml

# pip側の正確なスナップショット
conda run -n ENV_NAME python -m pip freeze > ENV_NAME-requirements.txt
```

`--from-history` は直接指定したCondaパッケージだけを記録する。pipで導入したものは別のrequirementsファイルで管理する。
