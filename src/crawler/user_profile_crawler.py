# -*- coding: utf-8 -*-
"""
Weibo user-profile crawler (academic research use only).

Input : CSV with a column `user_id` (e.g. output of weibo_crawler.py, run WITHOUT
        --anonymize on your own machine)
Output: one row per user: profile fields + repost/original counts sampled from
        the user's recent timeline + name_digit_ratio.

Usage:
    python src/crawler/user_profile_crawler.py \
        --in data/raw/posts.csv --out data/raw/profiles.csv \
        --timeline-pages 2 --anonymize

NOTE: the endpoints below are Weibo's web-AJAX endpoints. They are undocumented and
change from time to time -- test with a single user first.
"""
import argparse
import csv
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import hash_id  # noqa: E402

load_dotenv()
log = logging.getLogger("user_profile_crawler")

INFO_URL = "https://weibo.com/ajax/profile/info?uid={uid}"
DETAIL_URL = "https://weibo.com/ajax/profile/detail?uid={uid}"
TIMELINE_URL = "https://weibo.com/ajax/statuses/mymblog?uid={uid}&page={page}&feature=0"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

FIELDS = [
    "user_id", "screen_name", "description", "gender", "verified", "verified_reason",
    "created_at", "sunshine_credit", "vip_level",
    "followers_count", "friends_count", "statuses_count",
    "name_digit_ratio", "n_sampled_posts", "n_reposts", "n_original",
]


class RateLimited(Exception):
    pass


def make_session() -> requests.Session:
    cookie = os.getenv("WEIBO_COOKIE")
    if not cookie:
        raise SystemExit("请先在 .env 中设置 WEIBO_COOKIE（参考 .env.example）")
    s = requests.Session()
    # 原代码用 cookies={"cookie": ...} 会把整串当成名为 "cookie" 的单个cookie，应放进 header
    s.headers.update({"cookie": cookie, "user-agent": UA, "referer": "https://weibo.com/"})
    return s


def fetch_json(s: requests.Session, url: str) -> dict:
    resp = s.get(url, timeout=15)
    if resp.status_code in (403, 418, 429):
        raise RateLimited(f"HTTP {resp.status_code}")
    resp.raise_for_status()
    return resp.json()


def name_digit_ratio(name: str) -> float:
    return sum(c.isdigit() for c in name) / len(name) if name else 0.0


def sample_timeline(s: requests.Session, uid, pages: int) -> tuple[int, int]:
    """Return (n_reposts, n_original) over the first `pages` pages of the timeline."""
    reposts = original = 0
    for page in range(1, pages + 1):
        posts = (fetch_json(s, TIMELINE_URL.format(uid=uid, page=page)).get("data") or {}).get("list", [])
        if not posts:
            break
        for p in posts:
            if p.get("retweeted_status"):
                reposts += 1
            else:
                original += 1
        time.sleep(1 + random.random())
    return reposts, original


def get_profile(s: requests.Session, uid, timeline_pages: int) -> dict | None:
    user = (fetch_json(s, INFO_URL.format(uid=uid)).get("data") or {}).get("user")
    if not user:
        return None
    time.sleep(1 + random.random())
    detail = fetch_json(s, DETAIL_URL.format(uid=uid)).get("data") or {}
    reposts, original = sample_timeline(s, uid, timeline_pages) if timeline_pages else (0, 0)
    name = user.get("screen_name", "")
    return {
        "user_id": user.get("id", uid),
        "screen_name": name,
        "description": user.get("description", ""),
        "gender": user.get("gender", ""),
        "verified": user.get("verified", ""),
        "verified_reason": user.get("verified_reason", ""),
        "created_at": detail.get("created_at", ""),
        "sunshine_credit": (detail.get("sunshine_credit") or {}).get("level", ""),
        "vip_level": user.get("mbrank", ""),   # 会员等级字段，请用真实响应确认
        "followers_count": user.get("followers_count", ""),
        "friends_count": user.get("friends_count", ""),
        "statuses_count": user.get("statuses_count", ""),
        "name_digit_ratio": round(name_digit_ratio(name), 4),
        "n_sampled_posts": reposts + original,
        "n_reposts": reposts,
        "n_original": original,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Weibo user profile crawler (academic use only)")
    p.add_argument("--in", dest="inp", required=True, help="含 user_id 列的 CSV")
    p.add_argument("--out", default="data/raw/profiles.csv")
    p.add_argument("--timeline-pages", type=int, default=2, help="抽样多少页历史微博，0=不抓")
    p.add_argument("--anonymize", action="store_true", help="哈希ID，丢弃昵称与简介（数字占比已先算好）")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    uids = pd.read_csv(args.inp, encoding="utf-8-sig")["user_id"].dropna().astype(str).unique().tolist()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 断点续爬：已抓过的 ID 直接跳过（失败的用户不再写空行，下次会重试）
    done: set[str] = set()
    if out.exists():
        done = set(pd.read_csv(out, encoding="utf-8-sig")["user_id"].astype(str))
    new_file = not out.exists()
    s = make_session()

    with open(out, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, uid in enumerate(uids, 1):
            key = hash_id(uid) if args.anonymize else uid
            if key in done:
                continue
            try:
                row = get_profile(s, uid, args.timeline_pages)
            except RateLimited as e:
                log.error("被限流 (%s)，已保存进度，稍后重新运行即可续爬", e)
                break
            except (requests.RequestException, ValueError) as e:
                log.warning("[%d/%d] %s 失败: %s", i, len(uids), uid, e)
                continue
            if row is None:
                log.warning("[%d/%d] %s 无数据（账号不存在/被限制）", i, len(uids), uid)
                continue
            if args.anonymize:
                row["user_id"] = key
                row["screen_name"] = ""
                row["description"] = ""
            writer.writerow(row)
            f.flush()
            log.info("[%d/%d] saved %s", i, len(uids), uid)
            time.sleep(1 + random.random())


if __name__ == "__main__":
    main()
