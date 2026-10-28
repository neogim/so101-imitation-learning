# SO-101一括診断：120フレームの候補ばらつき・J2/J5・DP2・拡張判定

既存オフラインJSONのみ読み取り、IK再計算なし・Codex不要。120フレームの++ (2/6 PASS)と-- (6/6 PASS)について、位置到達候補・J2/J5の候補分布・DP2の不連続・研究用品質ゲートを同じ出力で比較する。候補解の単なる角度幅はnull-spaceの証明ではない。

```bash
git switch feature/cartesian-action-offline-export
git pull --ff-only origin feature/cartesian-action-offline-export
python -m py_compile scripts/so101_capture/review_dp_candidate_lattice_offline.py
python scripts/so101_capture/review_dp_candidate_lattice_offline.py \
  --candidates outputs/so101-ik-independent-000-119-stage-v1/per_seed.json \
  --paths outputs/so101-ik-dp-000-119-stage-v2/selected_paths.json \
  --summary outputs/so101-ik-dp-000-119-stage-v2/summary.json \
  --start 0 --end 119 \
  --output outputs/so101-ik-dp-000-119-stage-v2/lattice_review_v1.json
```

表示されるコンパクトな結果のみChatGPTへ共有する。詳細JSONはgitignored outputs内に保持。問題の不連続がJ5候補分布の切り替えと一致するか確認する。position-only IKのため、J5自由度があることはあり得るが、確定しない。時系列の角度変化を連続な実機軌道と同一視しない。姿勢拘束、衝突、校正符号・ゼロ位置、トルクは未検証。

## 次の一括段階（別途判断）
120フレーム単独ゲートが全6方向合格の--だけについて、追加区間のオフライン候補生成/DP2を検討できる。しかし**区間ごとのDP2合格は区間境界の連続性を保証しない**。648フレーム全体の滑らかな軌道を主張するには、同じフレーム列で再評価し、境界を跨ぐジャンプ、速度、加速度も検査する必要がある。最終的な648一括計算はこの結果確認後に指示する。

現在のALL-MAPPINGゲートは++の不合格によりSTOPである。--のみの有望性を誤って物理マッピング選択の根拠にしない。
