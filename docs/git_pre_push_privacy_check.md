# プライバシー保護 pre-push フック

**目的**：誤ってMacユーザー名やローカル絶対パス、PNG内テキスト・EXIF、JPEG EXIFをGitHubへ公開することを防ぐ。プッシュするGitコミットのblobを検査する（作業ツリーだけでなく、コミット済みのファイルも対象）。

## 一度だけインストールする（Mac、リポジトリのルート）

~~~bash
git pull --ff-only
python3 scripts/privacy/install_pre_push.py
~~~

インストール先は現在のリポジトリの \`.git/hooks/pre-push\`。既存フックや \`core.hooksPath\` がある場合は**上書きせず中止**する。ほかのリポジトリには影響しない。

このフックは通常の \`git push\` の直前に実行される。例えば、追跡対象に \`/Users/yourname/...\` や \`C:\Users\yourname\...\` が含まれればPushは失敗する。ローカルユーザー名は実行環境から自動取得するため、公開リポジトリのコードには書き込まれない。追加で監視したい識別子は以下を使う。

~~~bash
PRIVACY_EXTRA_IDENTIFIERS="private-nickname,private-hostname" git push
~~~

**運用上の注意**：
- 既にコミットされた不適切なファイルは、作業ツリーから削除するだけでは不十分な場合がある。必要に応じてコミットの修正や履歴からの除去を行う。
- フックは **git push --no-verify** で回避できる。強制力を高めるならGitHub側でも必須のCI検査／プッシュ保護を併用する。
- メタデータなしのPNGでも、**部屋、書類、画面、住所、顔、建物**などの目視可能な情報までは検出できない。GitHub公開前には画像自体も確認する。
- JPEGのEXIFとPNGのeXIf/tEXt/iTXt/zTXtを検出した場合は、**自動削除せずPushを停止**する。EXIFには位置以外の有用な情報もあり得るため、内容を確認してから再エンコードしてコミットし直す。
- すべてのバイナリ形式、ZIP内部、LFSデータ、すでに公開済みの過去履歴を完全に監査するものではない。12MiB超のGit blobも手動確認を求める。
- \`model_comparison.json\` などをGit管理するとき、\`source\` に絶対パスを入れない。相対パスに置換する。
- 初回導入前に漏えいしているコミットは遡って防げない。公開済み情報の削除はGitHub上の履歴・キャッシュも考慮する。

## コミット前に手動確認したい場合

~~~bash
python3 scripts/privacy/check_pre_push.py --check-working-file path/to/report.json
~~~

注：通常のPushはGitがフックに必要なref情報を渡すが、直接スクリプトを引数なしで呼ぶだけでは何も検査されない。**安全確認は通常のPush操作で行う**。

## テスト

~~~bash
python3 -m unittest scripts/so101_sim/test_privacy_pre_push.py -v
~~~
