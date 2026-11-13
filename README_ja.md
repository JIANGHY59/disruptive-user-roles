# ソーシャルメディアにおける有害なユーザー行動の識別と特徴分析

[English](README.md) | [简体中文](README_zh.md) | **日本語**

リポジトリのルートで `python -m streamlit run app.py --server.port 8502` を実行し、[ローカルの Streamlit デモ](http://127.0.0.1:8502/) を開きます。

Huiyu Jiang による微博（Weibo）の研究プロジェクトです。投稿とプロフィールの収集、中国語の攻撃的表現の検出、アカウント単位の特徴抽出を行います。アカウント分析では、組織的な拡散への関与が疑われるアカウント（中国語で「ネット水軍」）を対象とします。個人への攻撃を繰り返すアカウントのアノテーションと複数の役割の分類は今後の課題です。

## 実装内容

| 機能 | 内容 |
|---|---|
| データ収集 | トピック検索による投稿収集と、UID を指定したプロフィール・投稿の取得 |
| RoBERTa | 中国語の攻撃的表現を投稿単位で検出。学習、CSV 推論、Streamlit デモを提供 |
| 特徴抽出 | 投稿内容、交流ネットワーク、行動、プロフィールに関する特徴量を集計 |
| XGBoost | アカウント単位の学習、ホールドアウト評価、CSV 一括推論、Streamlit でのスコア表示 |

## 事例データ

2024 年 5 月 3 日の「胖猫事件」に関する微博の投稿 1,460 件を、投稿者の UID で既存の検出結果に照合しました。重複を除いた 1,307 アカウントの内訳は以下のとおりです。

| 過去の検出結果 | アカウント数 |
|---|---:|
| 水軍の疑いあり（`bot=1`） | 30 |
| フラグなし（`bot=0`） | 1,276 |
| ラベル欠損 | 1 |

ラベルのある 1,306 アカウントのうち、フラグが付いた割合は **2.30%** です。現行の XGBoost ベースラインは過去の `bot` を教師ラベルに使用し、`bot_score` は入力特徴量から除外しています。ホールドアウト評価は過去のラベルとの一致度を測ります。実際の検出性能を評価するには、独立した人手によるラベル付けが必要です。

## セットアップとデモ

リポジトリのルートディレクトリで実行します。

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

- **XGBoost 水军风险** では、数字の微博 UID を入力し、旧形式の投稿 CSV をアップロードするとアカウントのスコアを表示できます。モデルは既定で `models/water_army.json` から読み込みます。別のモデルを使う場合はローカルの `.env` で `WATER_ARMY_MODEL` を設定してください。
- **RoBERTa 文本检测** では、中国語のテキストを入力してスコアを確認できます。既定の重みの場所は `models/cyberbullying_roberta` です。別の重みを使う場合は `.env` で `ABUSE_MODEL` を設定してください。

リアルタイム取得には、`.env.example` を参考にローカルの `.env` を用意し、自分の微博アカウントの Cookie を `WEIBO_COOKIE` に設定します。以下の数字を対象の UID に置き換えると、プロフィールと投稿 1 ページを取得できるか確認できます。

```bash
python src/crawler/user_profile_crawler.py --check-uid 1234567890
```

微博から再ログイン、追加認証、アクセス制限の応答があった場合は、CSV でスコアを計算できます。リアルタイム取得のサンプルと学習時の事例データでは収集期間が異なります。

## XGBoost の学習と一括推論

旧形式の投稿 CSV を `data/raw/legacy_posts.csv`、既存のラベル CSV を `data/raw/labels.csv` に配置します。投稿 CSV には「楼主ID」「话题内容」などの元の中国語の列名が必要です。ラベル CSV には `uid` と `bot` が必要です。トピック検索クローラーが出力する CSV は列の形式が異なります。

```bash
python src/water_army.py train --posts data/raw/legacy_posts.csv --labels data/raw/labels.csv --model models/water_army.json
python src/water_army.py predict --posts data/raw/legacy_posts.csv --model models/water_army.json --out data/raw/water_army_predictions.csv
```

同じ UID の投稿を集計してから学習用と評価用に分割します。学習時はラベルが空欄のアカウントを除外し、同じ UID に異なるラベルがある場合はエラーを返します。学習結果はモデルと `models/water_army.metrics.json` に保存されます。推論結果には `uid,bot,bot_score` が含まれ、`--threshold` で既定値 0.5 の判定しきい値を変更できます。

## RoBERTa の学習と投稿分析

基盤モデルは `hfl/chinese-roberta-wwm-ext-large` です。ToxiCN、COLD、SWSR の入手方法は [学習用コーパスの案内](data/README.md) を参照してください。

```bash
python src/abuse_detector/train.py --data corpus.csv --out models/cyberbullying_roberta
python src/abuse_detector/predict_csv.py --model models/cyberbullying_roberta --in data/raw/topic_posts.csv --out data/raw/posts_scored.csv
python src/features.py --posts data/raw/posts_scored.csv --profiles data/raw/profiles.csv --out data/features.csv
```

学習スクリプトは `training_log.json` と `test_metrics.json` を出力します。トピック投稿とプロフィールの収集手順、および参考文献は [英語版 README](README.md) を参照してください。

## データの取り扱い

投稿の生データ、アカウント別ラベル、学習用コーパス、モデルの重みはリポジトリに含まれていません。Cookie はローカルの `.env` に保存してください。公開する場合は匿名化した集計結果を使い、モデルのスコアは人手で確認する対象の選定に用いてください。
