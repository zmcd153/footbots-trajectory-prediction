# GitHub公開前チェックリスト

- [ ] README内の`<YOUR_ACCOUNT>`をGitHubアカウント名へ変更した
- [ ] 大容量動画、データセット、モデル重みが含まれていない
- [ ] 個人情報、秘密鍵、APIキー、ローカル絶対パスが含まれていない
- [ ] 動画・画像・データセットの再配布権を確認した
- [ ] 公開するモデル重みの学習データ利用条件を確認した
- [ ] `LICENSE`を選択し、著作権者名を記載した
- [ ] `CITATION.cff`の著者情報を更新した
- [ ] `python scripts/verify_repo.py`が成功する
- [ ] 新しい仮想環境でREADMEの主要コマンドを確認した
- [ ] GitHub上でREADMEの日本語とコードブロックが正しく表示される

## GitHubへの基本的な登録例

```bash
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/<YOUR_ACCOUNT>/footbots-trajectory-prediction.git
git push -u origin main
```

GitHub上で先に空のリポジトリを作成し、READMEや`.gitignore`をGitHub側で自動追加しない設定にすると衝突を避けやすくなります。
