# SO-101 / ArUco Real-to-Sim：Mac実機収集・抽出の完全手順書

> 対象：macOS、SO-101 Leader/Follower、固定iPhone→OBS Sideカメラ、`feature/aruco-calibration-capture-20261107`。
> 最終確認：2026-11-07。コード静的確認による手順。**実機では未検証**。
> **このPRで完成したのは校正「観測データの収集・抽出」まで**。Eye-to-Hand 6DoF解法、MuJoCo一致の確定は後続工程。

## 0. この手順の設計：既存安全テレオペを再利用

- **収集**：既存 `scripts/so101_capture/capture_session.py` がLeader→Follower操作とSide映像＋Follower実測6軸を20Hzで同じLeRobotデータセットに記録する。
- **抽出**：実機とは独立に `scripts/so101_capture/extract_aruco_calibration.py` が静止区間とArUco ID 0〜5の四隅座標を抽出する。
- **実機収集中に `observe_aruco_follower_readonly.py` や他のテレオペプログラムを同時実行しない**。FollowerのUSBシリアルポートを二重に所有しないため。
- **古い53エピソード、既存 `cameras.local.yaml`、Leader/Followerキャリブレーションファイルは変更しない。**
- 収集した映像と状態は同じLeRobotフレームに保存されるが、OBSはカメラ露光時刻を提供しないため、厳密な時刻同期が保証されるわけではない。静止保持で影響を小さくする。
- 各マーカーは別リンクにある場合がある。各マーカーの貼付先リンク特定とFK検証は、6DoF外部校正に必要。

## 1. Mac上の作業環境

### 1-1. リポジトリとブランチ

SO-101リポジトリのルートで（作業中の未コミット変更があれば勝手に破棄しない）：

```bash
pwd
git status --short
git fetch origin
git switch feature/aruco-calibration-capture-20261107
git pull --ff-only
git branch --show-current
test -f scripts/so101_capture/capture_session.py
test -f scripts/so101_capture/extract_aruco_calibration.py
```

想定ブランチ：`feature/aruco-calibration-capture-20261107`。ブランチ切替がローカルで見つからない場合は、未コミット変更がないことを確認してから：

```bash
git switch --track origin/feature/aruco-calibration-capture-20261107
```

既存の作業変更がありswitchに失敗したら、**変更を消さずに停止**して確認する。次のすべてのコマンドはリポジトリルートで実行する。

### 1-2. condaを有効化

```bash
conda env list
conda activate lerobot
which python
python --version
python -c "import lerobot, cv2, av, pyarrow, yaml, numpy; print('LeRobot/cv2/PyAV/Arrow/YAML/NumPy OK'); print('ArUco:', hasattr(cv2, 'aruco'))"
```

- `lerobot` という環境名は**候補**。実際の一覧に存在しなければ、以前SO-101で使用した環境に置き換える（`so101-ik`を自動的に代用しない）。
- `python: command not found` はcondaが未有効な可能性が高い。まず `conda env list` と `conda activate ...` を行う。
- `conda: command not found` ならインストール済みのMiniforge/condaの初期化設定を確認。勝手に環境を新規作成・アップグレードしない。
- `ArUco: False` なら該当環境にOpenCV ArUcoがない。**実機で実行する前に**バージョン・依存関係を確認する。無断で `opencv-python` と `opencv-contrib-python` を混在インストールしない。
- プロジェクトのカメラ依存に分離されたローカル仮想環境がある場合は、これまで通りの実行方式を優先する。最初に上のimportが成立するか確認。

### 1-3. Macのポートと実機安全確認

```bash
ls /dev/tty.usbmodem*
```

既存コード `record_episode.py` のLeader/Follower既定値：
- Follower: `/dev/tty.usbmodem5B7B0098201`
- Leader: `/dev/tty.usbmodem5B7B0185961`

表示される実際のポートと一致するか必ず確認。違っていたら**無理に実行しない**。記録済みの元のキャリブレーションは再実行・上書きしない。

安全のため、障害物・ケーブル干渉・マーカー剥がれ・腕の落下リスクを除き、Followerを支えられる状態にする。初期位置をLeader/Followerで合わせる。既存コードは起動時の同期処理で**Followerを動かす可能性がある**。

## 2. OBS / Sideカメラの準備

### 2-1. OBSを起動する

1. 固定iPhoneのSide映像入力（以前のACT収集と同じOBSソース）を起動する。
2. OBSの**WebSocket v5**が有効であることを確認。接続資格情報は既存の環境変数 `OBS_WEBSOCKET_PASSWORD` またはリポジトリルートのローカル `.env` を使用。**パスワードをYAMLやGitに書かない**。
3. OBSソース名、ソースUUID、`expected_settings`、ズーム、クロップ、回転、表示アスペクト、DroidCamのカメラ設定は以前から変えない。
4. チェッカーボード16枚で求めた内部校正値は、**同じ640×480の画像幾何**と確認できた場合にだけ直接使う。モード変更に伴って実効画角／クロップが変わる場合は再検証が必要。
5. **640×480の要求は、元のiPhone映像を高解像度で再取得できたことを意味しない**。OBS `GetSourceScreenshot` は指定サイズで画像を返す。拡大だけの場合、マーカーの実質的な情報量は増えない。

### 2-2. 既存YAMLの実際の保存場所を探してから、安全に専用YAMLを作成

**重要：** `cameras.local.yaml` がリポジトリ直下にあるとは限らない。
`test -f ...` を独立した行で実行しても、通常の対話シェルでは以降の
`cp` や `python` を止めない。**存在確認に失敗したら、コピー処理に進まない。**

まず、**リポジトリルート**から既存のYAMLを探す（読み取りのみ）：

```bash
pwd
find . -type f \( -name 'cameras*.yaml' -o -name 'cameras*.yml' \) -print
```

見つからない場合、別の作業ディレクトリを使っていた可能性がある。
親ディレクトリも範囲を限定して検索：

```bash
find .. -maxdepth 4 -type f \( -name 'cameras*.yaml' -o -name 'cameras*.yml' \) -print
```

- ファイルがなければ、**そこで停止して旧収集時のカメラ設定を復元・特定する**。
  OBSソースのUUIDや `expected_settings` を推測で書かない。
- 見つかった候補に `cameras` → `side` → `provider: obs_source` があるかを読み取り確認する。
- **そのファイルのパスをSOURCE_YAMLに指定**する。以下の `<発見した既存YAMLのパス>`
  は必ず実在する値に置換し、プレースホルダーのまま実行しない。
- Gitブランチを切り替えたためにGit管理外のYAMLが消えたとは限らない。
  元の作業フォルダが違う可能性もある。

```bash
SOURCE_YAML="<発見した既存YAMLのパス>"
export SOURCE_YAML
```

続いて、次の**ひとつのPythonブロック**で存在確認・検証・コピー・640×480変更を一括実行する。
前段に失敗しても後段が実行される独立コマンド列とは異なり、エラーがあれば
保存処理の前に停止する。既存のYAMLは**読み取り専用**で、新規ファイルのみ作成する。

```bash
python - <<'PY'
import os
from pathlib import Path
import yaml

raw = os.environ.get("SOURCE_YAML", "")
if not raw or raw.startswith("<"):
    raise SystemExit("ERROR: SOURCE_YAML を実在する設定ファイルのパスに変更してください")
source = Path(raw).expanduser()
target = Path("cameras.aruco-640.local.yaml")
if not source.is_file():
    raise SystemExit(f"ERROR: 元のカメラ設定ファイルがありません: {source}")
if target.exists():
    raise SystemExit(f"ERROR: 既にある専用YAMLは上書きしません: {target}")
data = yaml.safe_load(source.read_text(encoding="utf-8"))
if not isinstance(data, dict) or not isinstance(data.get("cameras"), dict):
    raise SystemExit("ERROR: camerasマッピングがないYAMLです")
side = data["cameras"].get("side")
if not isinstance(side, dict) or side.get("provider") != "obs_source":
    raise SystemExit("ERROR: SideがOBS sourceとして設定されていません")
if not side.get("source_uuid") or not side.get("source_name"):
    raise SystemExit("ERROR: SideのOBS source UUID/nameが欠けています")
capture = data.setdefault("capture", {})
if not isinstance(capture, dict):
    raise SystemExit("ERROR: captureがマッピングではありません")
capture["width"], capture["height"] = 640, 480
# 新規作成専用。競合しても上書きせず失敗する。
with target.open("x", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
print("Created:", target)
print("Source retained:", source)
print("Capture: 640x480; Side OBS source:", side["source_name"])
PY
```

作成後は次で**読み取り専用検証**する：

```bash
python - <<'PY'
import sys
sys.path.insert(0, "scripts/so101_capture")
from multi_camera import load_config, capture_image_shape
cfg = load_config("cameras.aruco-640.local.yaml", enabled="side")
print("enabled:", tuple(cfg["active_cameras"]))
print("shape(H,W,C):", capture_image_shape(cfg))
assert tuple(cfg["active_cameras"]) == ("side",)
assert capture_image_shape(cfg) == (480, 640, 3)
PY
git status --short
```

`cameras.aruco-640.local.yaml` は個人のOBS設定・機器情報を含むローカルファイルなので、
**Gitへコミットしない**。必要なら `git check-ignore -v cameras.aruco-640.local.yaml`
で無視設定を確認する。未追跡として表示される場合は、作業に支障のない範囲で
`.git/info/exclude` などのローカル除外設定を使い、リポジトリに公開しない。

### 2-3. 640×480でArUcoが見えるか

Marker辞書：`DICT_4X4_50`、ID **0〜5**。外側の**黒いマーカー正方形14mm**（印刷紙全体18mm・白余白2mm）。コーナー推定に使う長さは14mm。

OBSからの静止画を用いる既存チェック：

```bash
python scripts/so101_capture/check_so101_aruco_image.py \
  --image /path/to/actual-side-still.png \
  --expected-ids 0 1 2 3 4 5 \
  --out-dir outputs/so101-aruco-preview-20261107
```

`/path/to/actual-side-still.png` は**実際に存在する撮影画像に置換**する。Sideカメラの画像だけを渡す（左右黒帯付きのスクリーンショットを使う場合、四隅画素座標は後の校正座標系と異なるので注意）。このスクリプトは見えないIDがあると終了コード2を返す。**必ず全6枚が同時に見える必要はない**。表示された `detected_ids` とannotated PNGで確認する。

## 3. 実機を動かさないpreflight

**先にデータセット名を決める**。同じ `--root` は一度録画したら再利用できない。最初は15〜20秒のスモークテスト専用名を推奨。

```bash
test -d data/so101
test ! -e data/so101/aruco-smoke-20261107
python scripts/so101_capture/capture_session.py \
  --root data/so101/aruco-smoke-20261107 \
  --camera-config cameras.aruco-640.local.yaml \
  --enable-cameras side \
  --seconds 20 --num-episodes 1 \
  --task "ArUco calibration capture smoke test"
```

期待：`File/USB-path preflight passed. No camera or serial connection.`。
ここでは**カメラ・モーターを接続しない**が、実機USBパスと校正ファイルの存在は確認する。エラーならその場で停止。

## 4. スモークテスト：最初の実機録画

`--execute` をつけると実機のモーターを操作する。Leader/Follower初期姿勢・ケーブル・障害物を確認後、対話式のMacターミナルでのみ実行：

```bash
python scripts/so101_capture/capture_session.py \
  --root data/so101/aruco-smoke-20261107 \
  --camera-config cameras.aruco-640.local.yaml \
  --enable-cameras side \
  --seconds 20 --num-episodes 1 \
  --task "ArUco calibration capture smoke test" \
  --execute
```

**実行操作（表示されるモード名を必ず確認）：**
1. 起動時にカメラ接続、Leader/Followerの校正照合、Follower初期保持、Leaderとのゆっくりしたアライメントが実行される。**Followerは動く場合がある**。
2. `SAFE_RESET` で開始。Leader/Followerの姿勢が合っている状態で `r` を押す。保持確認・カウントダウン後に `RECORDING` と表示されたら撮影開始。**`t` は録画なしのテレオペ**なので今回の収集には不向き。
3. `RECORDING` 中、数種類の姿勢へゆっくり動かす。各姿勢で1〜2秒静止。最初の20秒は操作フロー・保存確認だけを目的とし、20〜30姿勢を狙わない。
4. 20秒で自動終了するか、開始から最短1秒を過ぎて `e`。その後 `TELEOP_RESET` になる。
5. `h` で `SAFE_RESET`、状態表示を確認して `q` で終了する。途中で異常があれば安全に中止する。**Ctrl-Cは中断／部分保存になるため通常の終了操作として推奨しない**。
6. **モーターのトルクは自動的にOFFにならない**。終了時も保持力やアーム落下のリスクがある。電源の切断は腕を支えてから行う。異常時は安全を優先。

撮影中に他のプログラムからFollowerのUSBポートを開かない。

### 4-1. スモークテストの保存確認

```bash
test -f data/so101/aruco-smoke-20261107/meta/info.json
test -f data/so101/aruco-smoke-20261107/shutdown.json
python - <<'PY'
from pathlib import Path
import json
root = Path("data/so101/aruco-smoke-20261107")
info = json.loads((root/"meta/info.json").read_text())
shutdown = json.loads((root/"shutdown.json").read_text())
print("episodes:", info.get("total_episodes"))
print("frames:", info.get("total_frames"))
print("side:", "observation.images.side" in info.get("features", {}))
print("shutdown_error:", shutdown.get("error"))
print("cleanup_errors:", shutdown.get("cleanup_errors"))
assert info.get("total_episodes", 0) >= 1
assert info.get("total_frames", 0) > 0
assert "observation.images.side" in info.get("features", {})
assert shutdown.get("error") is None
assert not shutdown.get("cleanup_errors")
PY
```

エラーがあれば**本番撮影に進まない**。録画ファイルが出来ても不完全な場合がある。

### 4-2. スモークテストの抽出・確認

```bash
python scripts/so101_capture/extract_aruco_calibration.py \
  --dataset data/so101/aruco-smoke-20261107 \
  --out-dir outputs/so101-aruco-smoke-20261107 \
  --window 12 --tolerance-deg 0.7 \
  --distinct-deg 3 --min-edge-px 8 --max-poses 10
```

```bash
python - <<'PY'
from pathlib import Path
import json
p = Path("outputs/so101-aruco-smoke-20261107/manifest.json")
m = json.loads(p.read_text())
print("extracted poses:", len(m["samples"]))
for s in m["samples"]:
    print(s["image"], "IDs:", [x["id"] for x in s["markers"]],
          "edges(px):", [x["min_edge_px"] for x in s["markers"]])
PY
```

1件以上の `samples` があり、該当PNGと関節角度が保存されていることを確認。ゼロ件なら静止時間不足・ArUco視認性・縮小や圧縮などを切り分ける。**ゼロ件で120秒撮影へ進まない**。出力先は毎回新しい名前を使い、既存を上書きしない。

## 4-4. 2026-11-07 実機20秒スモーク結果と抽出v2での再試験

実機側の収集は成功：`aruco-smoke-20261107-02` に1エピソード・394フレームを保存した。
旧抽出器は静止代表フレーム6・105のみを検査して両方ともマーカーなし・0件だったが、
別途録画MP4を20フレーム間隔で走査すると、フレーム20/40でID 1、260/280/300でID 0/2/4などが検出できた。
関節状態の12フレーム最大変動幅は20で0.000°、40で0.264°、260〜300は4〜6°前後
（動作中）だった。このため**ArUcoの検出不能と断定してマーカーを張り替える必要はない**。

v2では12フレーム連続で最大変動0.7°以下の窓を探し、その窓に含まれる各フレームを
**すべて走査する**。連続する静止候補を1つの区間とみなし、区間ごとに
`min_edge_px` を満たした**検出マーカー数 → 合計最小辺長 → 最小辺長**
の順に品質を比較し、ベスト1枚と同じフレームの関節状態を出力する。
1つの静止区間から重複姿勢を増やさない。検出失敗時には
`frames_checked`, `frames_with_any_marker`, `ids_seen`,
`max_detected_edge_px` を `rejected` に記録する。

まずリポジトリを更新し、環境は引き続き `lerobot` を使用：

```bash
git status --short
git pull --ff-only
python -m unittest discover -s scripts/so101_capture -p 'test_extract_aruco_calibration.py' -v
```

**既存の出力先は上書き禁止**。新しい出力名 `-v2` を使って、保存済み394フレームを
再抽出する（ロボット、カメラ、OBS、PYTHONPATH不要）。

```bash
python scripts/so101_capture/extract_aruco_calibration.py \
  --dataset data/so101/aruco-smoke-20261107-02 \
  --out-dir outputs/so101-aruco-smoke-20261107-02-v2 \
  --window 12 --tolerance-deg 0.7 \
  --distinct-deg 3 --min-edge-px 8 --max-poses 10
```

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("outputs/so101-aruco-smoke-20261107-02-v2/manifest.json")
m = json.loads(p.read_text(encoding="utf-8"))
print("stationary_regions:", m["stationary_regions"])
print("eligible_frames_examined:", m["eligible_frames_examined"])
print("saved:", len(m["samples"]), "rejected:", len(m["rejected"]))
for item in m["samples"]:
    print("frame", item["frame_index"],
          "IDs", [x["id"] for x in item["markers"]],
          "edges", [x["min_edge_px"] for x in item["markers"]])
for item in m["rejected"]:
    print("rejected:", item["reason"],
          "frames_checked", item["frames_checked"],
          "IDs_seen", item["ids_seen"])
PY
```

**事前に検出数を保証できない**：サンプル保存には静止判定・ArUco実検出・
最低辺長8pxのすべてが必要。0件だった場合も、診断情報を見て
閾値・撮影方法を見直す。今回はデータ再撮影より再抽出を優先する。

## 5. 本番：25前後の多様な姿勢を収集

スモークが成功した場合のみ、新しい `--root` を使用する。まず事前検証：

```bash
test ! -e data/so101/aruco-calibration-20261107
python scripts/so101_capture/capture_session.py \
  --root data/so101/aruco-calibration-20261107 \
  --camera-config cameras.aruco-640.local.yaml \
  --enable-cameras side \
  --seconds 120 --num-episodes 1 \
  --task "Observe stationary ArUco poses for camera calibration"
```

実機実行（明示的に `--execute` を追加）：

```bash
python scripts/so101_capture/capture_session.py \
  --root data/so101/aruco-calibration-20261107 \
  --camera-config cameras.aruco-640.local.yaml \
  --enable-cameras side \
  --seconds 120 --num-episodes 1 \
  --task "Observe stationary ArUco poses for camera calibration" \
  --execute
```

- `r` で録画開始、`RECORDING` を確認。
- 目安：中央5、左右6、前後上下6、手首回転8など**20〜30の重ならない姿勢**。各1〜2秒静止、無理に速度を上げない。
- 120秒の上限を超えた場合は、**別の新しい出力フォルダ**に分けて収集。繰り返し撮影で旧データを上書きしない。
- 終了：`e`（制限時間前）、`h`、`q`。自動終了後は `h`、`q`。
- **Sideの視野から消える姿勢ばかり撮らない**。すべての姿勢で6枚見える必要はないが、主となる同一リンクのArUcoが十分な数の姿勢で検出できることが重要。

終了後は**4-1と同じ確認**で、ディレクトリ名だけ `aruco-calibration-20261107` に置き換える。

## 6. 本番データのオフライン抽出

```bash
python scripts/so101_capture/extract_aruco_calibration.py \
  --dataset data/so101/aruco-calibration-20261107 \
  --out-dir outputs/so101-aruco-calibration-20261107 \
  --window 12 --tolerance-deg 0.7 \
  --distinct-deg 3 --min-edge-px 8 --max-poses 30
```

出力：
- `manifest.json`：各静止姿勢のepisode番号、frame番号、timestamp、実測6関節、検出ID、四隅座標、黒枠一辺14mm、警告。
- `images/pose_*.png`：選ばれたSideフレーム。

**成功の最低条件**：候補姿勢が複数、PNGとJSONの対応が正しく、同じマーカーが異なる位置・方向から十分見えていること。再投影誤差、FK整合性、複数姿勢での再現性を確認するまでは、外部パラメータが求まったとは言えない。

**注：この抽出器は最大30件であり、複数エピソード統合や各リンクごとの校正解決は将来の拡張。** また、すべての録画フレームの画像・関節角度をローカルに残しておけば、別の抽出閾値で再解析できる。

## 7. テスト（ロボット接続不要）

```bash
python -m unittest discover -s scripts/so101_capture -p 'test_extract_aruco_calibration.py' -v
```

このテストは静止姿勢選別の数値ロジックを検証。実LeRobot MP4/Parquet統合テストと実機操作の妥当性は別途必要。

### 4-3. OBS WebSocket依存がない場合（実機接続前に停止したケース）

`ModuleNotFoundError: No module named 'websocket'` /
`Install websocket-client into isolated .local/camera-deps` は、
OBSカメラ接続に必要な `websocket-client` がPythonから見えていないことを示す。
OpenCV/PyAVの `AVFFrameReceiver` 警告は別問題であり、**このエラーを解消するために
既存のFFmpegライブラリを削除・改名しない**。

```bash
python -m pip install --target .local/camera-deps websocket-client
PYTHONPATH="$PWD/.local/camera-deps${PYTHONPATH:+:$PYTHONPATH}" \
  python -c "import websocket; print('websocket-client OK:', websocket.__version__)"
```

`--target` でのインストールは `lerobot` conda環境のsite-packagesに直接追加しない。
**実行時にも同じPYTHONPATHを付ける必要がある**。

カメラ接続前でも、`LeRobotDataset.create()` が出力rootを作った可能性がある。
**出力ディレクトリを消さず、必ず新しいroot名に変更**する。
例：`aruco-smoke-20261107-02`。既存ディレクトリの有無は
`ls -ld data/so101/aruco-smoke-20261107*` で確認できる。
再実行前にカメラ接続・OBS WebSocket v5・パスワードを確認する。

```bash
PYTHONPATH="$PWD/.local/camera-deps${PYTHONPATH:+:$PYTHONPATH}" \
python scripts/so101_capture/capture_session.py \
  --root data/so101/aruco-smoke-20261107-02 \
  --camera-config scripts/so101_capture/cameras.aruco-640.local.yaml \
  --enable-cameras side \
  --seconds 20 --num-episodes 1 \
  --task "ArUco calibration capture smoke test" \
  --execute
```

この実行は**実機モーターが動く可能性がある**ため、十分な初期姿勢・障害物・
ケーブルの確認後に実行する。初回の `--execute` なしpreflightは成功していても、
`--execute` 時のOBS接続・モーター動作を保証するものではない。

## 8. よくあるエラーと対処

| 症状 | 原因の候補・対処 |
|---|---|
| `zsh: command not found: python` | `conda env list` → `conda activate lerobot`（実在する環境名）→ `which python`。 |
| `conda: command not found` | インストール済みcondaの初期化／PATHを確認。プロジェクトに合わせて環境を選び直す。 |
| `No such file: cameras.aruco-640.local.yaml` | 2-2で専用YAMLを作成。Git管理対象の元YAMLを変更しない。 |
| `Unknown or empty enabled camera list` | YAMLの `cameras.side` を確認。 `--enable-cameras side` と完全一致。 |
| `No camera or serial connection` | **正常**：`--execute` なしのpreflight終了。 |
| `No module named websocket` | `websocket-client` が見えていない。4-3を参照。 |
| OBS WebSocket認証失敗 | OBSを起動し既存のWebSocket v5設定とローカル `.env` を確認。パスワードを公開しない。 |
| Camera shape mismatch / `capture width/height` | 専用YAMLの640/480とコードの許可サイズを照合。 |
| `Missing USB port` / calibration missing | 現在のMacポート・旧校正データを確認し、実機接続を停止。再キャリブレーションを自動実行しない。 |
| `Leader/Follower pose difference >3` | 操作を停止。Leader/Followerを安全に同じ姿勢にする。Followerを力任せに押さない。 |
| `Control gap >250 ms` / `Tracking error >8` | スクリプト停止・安全確認。OBS取得遅延、USB通信、ホスト負荷など切り分け。 |
| 録画はあるが抽出サンプル0 | 各姿勢で静止が足りない、マーカーが小さい・隠れる、録画品質が不十分。まず短い再試験。 |
| 一部のIDが見えない | **全6枚同時検出は必須ではない**。1つの基準リンクで多様な姿勢を確保する。 |
| 120秒で足りない | 新しい `--root` で別の収集。以前の録画を上書きしない。 |
| 終了後、アームが保持される | **仕様：トルクは自動OFFにならない**。安全に支えてから電源操作。 |

## 9. 外部校正ソルバーへ渡すときの注意

- 実測14mmとカメラ内部パラメータの画像座標系を一致させる。
- ID 0〜5の**どれがどの剛体リンクにあるか**を明示する。異なる可動リンクの観測を「固定ボード」として足し合わせない。
- マーカー→リンクの取付変換が未知なら、それ自体も推定するEye-to-Hand最適化が必要。
- FKのJ2/J3符号・原点・URDFの正しさ、マーカー取付剛性、姿勢分布が精度を左右する。
- 最後は**未使用の姿勢**でSide画像とMuJoCo投影を比較する。
- 現PRは **観測データ抽出と単体テスト**まで。実機収集データでの結合試験、6DoF解法、MuJoCoへの反映は未完。


## 10. 収集後：J5親子リンクとID 0/1/2の対応（2026-11-07）

実機観測：**J5を単独で回すとリストカメラ本体・ID 0・ID 1は
一緒に回転し、ID 2はJ5では動かない**。ID 2はリストカメラ背面側、
ケーブルと重なる位置にある。取付先がケーブルそのものや
可動するカバーでないことを実物で確認する。

GitHub内の `data/urdf/so101/so101_new_calib.urdf` では
`wrist_roll` (J5) の親は `wrist_link`、子は `gripper_link`、
関節軸は `0 0 1`。

| ID | J5単独回転での動き | URDFリンク（仮説） | 本番22姿勢での採用回数 |
|---|---|---|---:|
| 0 | J5と同時に回転 | `gripper_link` | 10 |
| 1 | J5と同時に回転 | `gripper_link` | 9 |
| 2 | J5単独では回転しない | `wrist_link` | 16 |
| 3 | 未確認 | **未割当** | 5 |
| 4 | 未確認 | **未割当** | 12 |
| 5 | 未確認 | **未割当** | 0 |

**J5の子側リンクで動くことが判明しても、
マーカーからリンク原点への固定6DoF変換は未知**。
なおリストカメラ本体が動くことと、背面側にあるID2が固定であることは
両立する（カメラ本体と別の剛体に貼られている可能性）。どの部品に
貼られているか、ケーブルで浮いていないかを現物確認する。

以下はURDFと本番 `manifest.json` を**読み取り専用**で照合するコマンド。
ロボットとOBSは起動不要。

```bash
conda activate lerobot
python scripts/so101_capture/audit_aruco_links.py \
  --manifest outputs/so101-aruco-calibration-20261108/manifest.json
```

出力にはJ5親/子リンク、候補IDごとの採用数と辺長、各フレーム番号、
未確認事項を記録する。これ自体は外部校正やマーカー取付位置の検証ではない。

**次の工程：**
1. ID 0・1・2の取付面が剛体であることと、J5だけを動かした時の相対運動を実物で確かめる。
2. OBSの最終640×480画像に対応したカメラ内部パラメータを確定する。
3. URDFの角度単位/ゼロ点/符号と `observation.state` のFK整合を検証する。
4. 上記の観測と `marker→link` 未知変換を用いて、Sideカメラの6DoF外部校正を推定する。
5. 使用していない姿勢の再投影で精度を評価する。

**注**：`so101_new_calib.urdf` は一般モデルであり、
今回の実機の追加カメラ固定具やArUco位置を自動的には表現していない。

## 11. URDF FK監査（実機不要・2026-11-07）

`scripts/so101_capture/audit_fk_aruco.py` は既存の22姿勢分の `joints` を
URDFのJ1〜J5へ**度→ラジアン**変換して適用し、`base_link` から
`wrist_link` / `gripper_link` への4×4同次変換を計算する。
URDF `origin xyz/rpy` と `axis` に従い回転関節の変換を連結する。
マーカー自体の取付位置、カメラ内部・外部パラメータ、物理実機との一致は**推定しない**。

```bash
conda activate lerobot
git pull --ff-only
python -m unittest discover -s scripts/so101_capture -p 'test_*aruco.py' -v
python scripts/so101_capture/audit_fk_aruco.py \
  --manifest outputs/so101-aruco-calibration-20261108/manifest.json
```

結果は標準出力のサマリーとJSONに表示する。
`--perturb-j5-deg 10` がデフォルトで、
`wrist_link` の位置・回転差は理論上0、
`gripper_link` の姿勢差は理論上10°となる。
これは**URDFトポロジとFK実装の内部整合性**の確認であり、
実機に対して符号やゼロ点が正しいことの証明ではない。

`joint_limit_checks` は各軸の実測角度レンジとURDF制限を比較する。
J2の -103.21° はURDF定義の下限約-100°を超えるので警告対象。
オフセット／可動制限・calibration.json・実機との差があり得るため、
**角度を自動補正したりURDFを書き換えたりしない**。

`fk_position_ranges_m` は各座標の最大−最小幅であり、
位置推定の誤差や姿勢分布の適否を保証しない。
`per_sample` には各フレームのBase→リンク原点位置とJ5摂動テスト結果を含む。

次の段階では以下を別途実施する必要がある：

1. `observation.state` の角度とURDF角度のゼロ点・符号を外部基準で照合
2. ID 0/1/2のマーカー取付先が剛体であることを確認
3. Sideカメラ最終640×480画像で内部パラメータを校正
4. リンク→マーカーの未知変換とBase→Sideカメラの未知変換を同時推定
5. 未使用姿勢の再投影誤差・幾何の再現性で検証



## 12. 全姿勢ペア監査と追加撮影の判断（2026-11-07）

従来の抽出器では**直前に採用した姿勢との関節角度差**で重複を除外している。
そのため離れた時点の再訪姿勢、特に`wrist_link`上のID2にとって
J5のみ違う姿勢が残り得る。追加撮影の前に
`scripts/so101_capture/audit_aruco_diversity.py` で
全姿勢ペアを監査する（読み取り専用）。

```bash
conda activate lerobot
git pull --ff-only
python -m unittest discover -s scripts/so101_capture -p 'test_*aruco*.py' -v
python scripts/so101_capture/audit_aruco_diversity.py \
  --manifest outputs/so101-aruco-calibration-20261108/manifest.json
```

`pairwise_joint_near_repeats`：
- `all_five_near_duplicate` = J1〜J5最大角度差が3°未満。
- `wrist_fk_near_duplicate_candidate` = J1〜J4最大角度差が3°未満
  （J5が異なっても、ID2のリンク姿勢は同じ可能性）。

`per_marker` にはID0/1（`gripper_link`仮説）、ID2（`wrist_link`仮説）ごとに、
観測枚数、FK上の最大3D位置差・回転差、近接ペア、
位置座標のXYZ広がり・中心化座標の特異値を表示する。
5mm・5°未満の近接ペアは重複候補として報告する。
これらは**仮の閾値**であって、校正精度の保証や真の独立観測数の証明ではない。

**追撮する場合**：現在のデータセットとは別rootに記録し、
同じカメラ位置・ズーム・OBS画像変換を維持する。
ID2（J5上流）を主対象としてJ1〜J4を使い、異なる3D位置と傾きの姿勢を
確保。ID0/1（J5下流）には追加のJ5回転も有効。
各姿勢2秒以上静止し、マーカーの黒枠が十分大きく見える距離・角度を選ぶ。
可動リミットを押し込まないこと。新規撮影は既存22姿勢の再収集ではなく
不足する分布を補完するために行う。

**まだ推定していないもの：** カメラ内部パラメータ、
実機とURDF角度の符号・ゼロ点、
マーカー取付剛体変換、Base-to-Sideカメラ外部変換。
それらの推定には適切な校正ターゲットと別工程が必要。
