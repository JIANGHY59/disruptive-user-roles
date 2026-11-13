"""Screen for repeated abusive posting with RoBERTa and an XGBoost baseline."""

import argparse
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from abuse_detector.detector import AbuseDetector
from water_army import POST_COLUMNS, build_account_features, read_columns, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posts", required=True, type=Path)
    parser.add_argument("--roberta", required=True)
    parser.add_argument("--model", type=Path, help="Optional XGBoost model output")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    posts = read_columns(args.posts, POST_COLUMNS)
    posts["楼主ID"] = posts["楼主ID"].str.strip()
    features = build_account_features(posts)
    unique = posts[["楼主ID", "话题内容"]].dropna().drop_duplicates()
    unique = unique.loc[unique["话题内容"].str.strip().ne("")]
    detector = AbuseDetector(args.roberta)
    unique["high_abuse"] = detector.predict_proba(unique["话题内容"].tolist(), batch_size=8) >= 0.8
    evidence = unique.groupby("楼主ID")["high_abuse"].agg(["size", "sum"])
    repeated = evidence["size"].ge(2)
    positive = repeated & evidence["sum"].ge(2) & evidence["sum"].div(evidence["size"]).ge(0.5)
    negative = repeated & evidence["sum"].eq(0)
    labels = positive[positive | negative].astype(int).rename("troll_proxy")
    print(f"Proxy labels: {int(labels.sum())} positive, {int(labels.eq(0).sum())} negative")
    # Only score accounts with at least two distinct texts in the sample.
    features = features.loc[evidence.index[repeated]]
    result = evidence.loc[features.index].rename(columns={"size": "distinct_posts", "sum": "high_abuse_posts"})
    result["proxy_label"] = labels
    if args.model is not None:
        if labels.value_counts().reindex([0, 1], fill_value=0).min() < 5:
            print("Skipped XGBoost training: need at least 5 accounts in each proxy class.")
        else:
            train(features, labels, args.model)
            model = XGBClassifier()
            model.load_model(args.model)
            result["troll_candidate_score"] = model.predict_proba(features)[:, 1]
    result.index.name = "uid"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
