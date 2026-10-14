# -*- coding: utf-8 -*-
"""
Build the user feature matrix (four dimensions from the poster).

    Content          : abusive_content_ratio (optional), content_similarity, at_link_ratio
    Network          : degree_centrality, degree_rank_pct
    Behavior         : repost_original_ratio
    Profile          : following_follower_ratio, sunshine_credit, vip_level, name_digit_ratio

Usage:
    python src/features.py --posts data/raw/posts.csv --profiles data/raw/profiles.csv \
                           --out data/features.csv

If posts.csv has a 0/1 column `is_abusive` (output of your abuse detector), the
abusive_content_ratio feature is added.
"""
import argparse
import re

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

AT_LINK_RE = re.compile(r"@\S+|https?://\S+|网页链接")  # 微博去HTML后，链接多显示为“网页链接”


# ----------------------------------------------------------------------------- content
def at_link_ratio(posts: pd.DataFrame) -> pd.Series:
    """Share of a user's posts containing an @mention or a link."""
    flag = posts["text"].fillna("").map(lambda t: bool(AT_LINK_RE.search(t)))
    return flag.groupby(posts["user_id"]).mean().rename("at_link_ratio")


def content_similarity(posts: pd.DataFrame, max_posts: int = 50) -> pd.Series:
    """Mean pairwise cosine similarity (char 2-gram TF-IDF) among a user's own posts.

    Users with < 2 posts get NaN. Align this definition with the one used in your
    original study before reporting numbers.
    """
    texts = posts["text"].fillna("").tolist()
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 2)).fit(texts)
    out = {}
    for uid, grp in posts.groupby("user_id"):
        t = grp["text"].fillna("").head(max_posts).tolist()
        if len(t) < 2:
            out[uid] = np.nan
            continue
        sim = cosine_similarity(vec.transform(t))
        out[uid] = sim[np.triu_indices_from(sim, k=1)].mean()
    return pd.Series(out, name="content_similarity")


def abusive_content_ratio(posts: pd.DataFrame) -> pd.Series | None:
    if "is_abusive" not in posts.columns:
        return None
    return posts.groupby("user_id")["is_abusive"].mean().rename("abusive_content_ratio")


# ----------------------------------------------------------------------------- network
def build_edges(posts: pd.DataFrame) -> pd.DataFrame:
    """Interaction edges (source -> target): reposts and @mentions.

    @mentions only carry a screen name, so they are mapped to user ids through the
    screen_name column; this needs data crawled WITHOUT --anonymize (keep it private).
    """
    edges = []
    for r in posts.itertuples():
        if pd.notna(r.retweeted_user_id) and str(r.retweeted_user_id) != "":
            edges.append((str(r.user_id), str(int(float(r.retweeted_user_id)))))
    name2id = dict(zip(posts["screen_name"].fillna(""), posts["user_id"].astype(str)))
    for r in posts.itertuples():
        for name in str(r.mentions).split("|") if pd.notna(r.mentions) else []:
            if name in name2id:
                edges.append((str(r.user_id), name2id[name]))
    return pd.DataFrame(edges, columns=["source", "target"])


def degree_features(edges: pd.DataFrame) -> pd.DataFrame:
    """Degree centrality + percentile rank (top x%; 10% = among the 10% most connected)."""
    g = nx.Graph()
    g.add_edges_from(e for e in edges.itertuples(index=False) if e.source != e.target)
    dc = pd.Series(nx.degree_centrality(g), name="degree_centrality")
    rank = dc.rank(pct=True, ascending=False).rename("degree_rank_pct")
    return pd.concat([dc, rank], axis=1)


# ----------------------------------------------------------------------------- behavior
def repost_original_ratio(profiles: pd.DataFrame) -> pd.Series:
    """Sampled reposts / max(sampled originals, 1)."""
    ratio = profiles["n_reposts"] / profiles["n_original"].clip(lower=1)
    return pd.Series(ratio.values, index=profiles["user_id"].astype(str), name="repost_original_ratio")


# ----------------------------------------------------------------------------- assemble
def build_feature_matrix(posts: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    posts = posts.copy()
    posts["user_id"] = posts["user_id"].astype(str)
    profiles = profiles.copy()
    profiles["user_id"] = profiles["user_id"].astype(str)

    parts = [at_link_ratio(posts), content_similarity(posts)]
    abusive = abusive_content_ratio(posts)
    if abusive is not None:
        parts.append(abusive)
    feats = pd.concat(parts, axis=1)

    edges = build_edges(posts)
    if len(edges):
        feats = feats.join(degree_features(edges)["degree_centrality"], how="left")
        feats["degree_centrality"] = feats["degree_centrality"].fillna(0)  # 没有任何互动的用户
        feats["degree_rank_pct"] = feats["degree_centrality"].rank(pct=True, ascending=False)

    prof = profiles.set_index("user_id")
    feats = feats.join(repost_original_ratio(profiles), how="left")
    feats["following_follower_ratio"] = (
        prof["friends_count"].astype(float) / prof["followers_count"].astype(float).clip(lower=1)
    )
    for col in ["sunshine_credit", "vip_level", "name_digit_ratio"]:
        feats[col] = prof[col]
    return feats.rename_axis("user_id").reset_index()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--posts", required=True)
    p.add_argument("--profiles", required=True)
    p.add_argument("--out", default="data/features.csv")
    a = p.parse_args()
    posts = pd.read_csv(a.posts, encoding="utf-8-sig")
    profiles = pd.read_csv(a.profiles, encoding="utf-8-sig")
    feats = build_feature_matrix(posts, profiles)
    feats.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"saved {len(feats)} users x {feats.shape[1] - 1} features -> {a.out}")


if __name__ == "__main__":
    main()
