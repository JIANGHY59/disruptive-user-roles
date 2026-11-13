"""Train and run an account-level XGBoost baseline for the legacy Weibo CSV."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import at_link_ratio


PROFILE_COLUMNS = {
    "发博数量": "statuses_count",
    "关注人数": "friends_count",
    "粉丝数": "followers_count",
    "微博等级": "weibo_level",
}
ENGAGEMENT_COLUMNS = {"转发量": "reposts", "评论量": "comments", "点赞量": "likes"}
CREDIT = {"信用极低": 1, "信用较低": 2, "信用一般": 3, "信用较好": 4, "信用极好": 5}
POST_COLUMNS = ["楼主ID", "话题内容", "楼主昵称", "阳光信用", *PROFILE_COLUMNS, *ENGAGEMENT_COLUMNS]


def read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="string", encoding="utf-8-sig")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} 缺少列：{', '.join(sorted(missing))}")
    return frame[columns].copy()


def parse_count(values: pd.Series) -> pd.Series:
    parts = values.str.replace(",", "", regex=False).str.strip().str.extract(
        r"^(\d+(?:\.\d+)?)\s*([万亿]?)\+?$"
    )
    return pd.to_numeric(parts[0]).astype(float) * parts[1].map({"": 1, "万": 10_000, "亿": 100_000_000})


def account_features(path: Path) -> pd.DataFrame:
    return build_account_features(read_columns(path, POST_COLUMNS))


def build_account_features(posts: pd.DataFrame) -> pd.DataFrame:
    missing = set(POST_COLUMNS) - set(posts.columns)
    if missing:
        raise ValueError(f"博文表缺少列：{', '.join(sorted(missing))}")
    posts = posts[POST_COLUMNS].astype("string").rename(columns={
        "楼主ID": "user_id", "话题内容": "text", **PROFILE_COLUMNS, **ENGAGEMENT_COLUMNS
    })
    posts["user_id"] = posts["user_id"].str.strip().replace("", pd.NA)
    if posts["user_id"].isna().any() or posts.empty:
        raise ValueError("博文表为空或存在空白楼主 ID")

    for column in [*PROFILE_COLUMNS.values(), *ENGAGEMENT_COLUMNS.values()]:
        posts[column] = parse_count(posts[column])
    posts["sunshine_credit"] = posts["阳光信用"].str.strip().map(CREDIT)
    names = posts["楼主昵称"]
    posts["name_digit_ratio"] = names.str.count(r"\d").astype(float) / names.str.len().clip(lower=1)
    posts["text_length"] = posts["text"].str.len().astype(float)

    grouped = posts.groupby("user_id")
    features = grouped[[*PROFILE_COLUMNS.values(), "sunshine_credit", "name_digit_ratio"]].median()
    features["following_follower_ratio"] = features["friends_count"] / features["followers_count"].clip(lower=1)
    features["sampled_post_count"] = grouped.size()
    features["mean_text_length"] = grouped["text_length"].mean()
    features = features.join(grouped[list(ENGAGEMENT_COLUMNS.values())].mean().add_prefix("mean_"))
    return features.join(at_link_ratio(posts)).astype(float)


def account_labels(path: Path) -> pd.Series:
    labels = read_columns(path, ["uid", "bot"])
    labels["uid"] = labels["uid"].str.strip().replace("", pd.NA)
    raw = labels["bot"].str.strip().replace("", pd.NA)
    labels["bot"] = pd.to_numeric(raw, errors="coerce")
    if (raw.notna() & ~labels["bot"].isin([0, 1])).any():
        raise ValueError("bot 标签必须为 0、1 或留空")
    labels = labels.dropna(subset=["bot"])
    if labels["uid"].isna().any():
        raise ValueError("存在有标签但缺少 UID 的记录")
    grouped = labels.groupby("uid")["bot"]
    if grouped.nunique().gt(1).any():
        raise ValueError("同一 UID 存在冲突标签，请先复核")
    return grouped.first().astype(int)


def train(features: pd.DataFrame, labels: pd.Series, model_path: Path) -> None:
    dataset = features.join(labels, how="inner")
    y = dataset.pop(labels.name)
    counts = y.value_counts()
    if len(counts) != 2 or counts.min() < 5:
        raise ValueError("匹配后的训练样本每类至少需要 5 个账号")
    x_train, x_test, y_train, y_test = train_test_split(
        dataset, y, test_size=0.25, stratify=y, random_state=42
    )
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=float(y_train.eq(0).sum() / y_train.eq(1).sum()),
        eval_metric="logloss",
        random_state=42,
        n_jobs=2,
    )
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    predictions = scores >= 0.5
    metrics = {
        "reference": f"input {labels.name} labels",
        "train_accounts": len(x_train),
        "test_accounts": len(x_test),
        "post_accounts_without_matching_label": len(features) - len(dataset),
        "labeled_accounts_without_posts": len(labels) - len(dataset),
        "threshold": 0.5,
        "average_precision": average_precision_score(y_test, scores),
        "confusion_matrix_0_1": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1], output_dict=True, zero_division=0
        ),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    metrics_path = model_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def predict(features: pd.DataFrame, model_path: Path, output: Path, threshold: float) -> None:
    model = XGBClassifier()
    model.load_model(model_path)
    result = score_accounts(model, features, threshold)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, encoding="utf-8-sig")
    print(f"已输出 {len(result)} 个账号的预测")


def score_accounts(model: XGBClassifier, features: pd.DataFrame, threshold: float) -> pd.DataFrame:
    scores = model.predict_proba(features)[:, 1]
    return pd.DataFrame(
        {"bot": (scores >= threshold).astype(int), "bot_score": scores},
        index=features.index.rename("uid"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    inference = commands.add_parser("predict")
    for command in (training, inference):
        command.add_argument("--posts", required=True, type=Path)
        command.add_argument("--model", required=True, type=Path, help="XGBoost .json or .ubj model")
    training.add_argument("--labels", required=True, type=Path)
    inference.add_argument("--out", required=True, type=Path)
    inference.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.model.suffix.lower() not in {".json", ".ubj"}:
        parser.error("--model 必须使用 .json 或 .ubj 后缀")
    if args.command == "predict" and not 0 <= args.threshold <= 1:
        parser.error("--threshold 必须在 0 和 1 之间")

    try:
        features = account_features(args.posts)
        if args.command == "train":
            train(features, account_labels(args.labels), args.model)
        else:
            predict(features, args.model, args.out, args.threshold)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
