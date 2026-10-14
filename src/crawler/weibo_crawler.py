# -*- coding: utf-8 -*-
"""
Weibo topic crawler (academic research use only).

Collects posts under a keyword within a time window, plus basic author info.

Usage:
    cp .env.example .env     # fill in WEIBO_COOKIE
    python src/crawler/weibo_crawler.py \
        --keyword "你的话题关键词" \
        --start "2024-05-03 19:00" --end "2024-05-03 20:00" \
        --max-pages 20 --out data/raw/posts.csv --anonymize
"""
import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("weibo_crawler")

SEARCH_URL = (
    "https://s.weibo.com/weibo?q={q}&typeall=1&suball=1"
    "&timescope=custom%3A{start}%3A{end}&Refer=g&page={page}"
)
DETAIL_URL = "https://m.weibo.cn/detail/{mid}"
UA = (
    "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Mobile Safari/537.36"
)

FIELDS = [
    "post_url", "user_id", "text", "screen_name", "gender",
    "verified", "verified_type", "verified_type_ext",
    "statuses_count", "follow_count", "followers_count", "urank",
    "is_repost", "retweeted_user_id", "mentions", "created_at", "reposts_count", "comments_count", "attitudes_count",
]


def make_session() -> requests.Session:
    cookie = os.getenv("WEIBO_COOKIE")
    if not cookie:
        raise SystemExit("请先在 .env 中设置 WEIBO_COOKIE（参考 .env.example）")
    s = requests.Session()
    s.headers.update({"cookie": cookie, "user-agent": UA, "x-requested-with": "XMLHttpRequest"})
    return s


def fmt_scope(dt: datetime) -> str:
    """Weibo search timescope format: YYYY-MM-DD-H (hour has no zero padding)."""
    return f"{dt.year}-{dt.month:02d}-{dt.day:02d}-{dt.hour}"


def polite_sleep(base: float = 1.0) -> None:
    time.sleep(base + random.random())


def get_post_ids(s: requests.Session, keyword: str, start: datetime, end: datetime,
                 max_pages: int) -> list[str]:
    """Search pages -> list of post ids (mid), de-duplicated, order preserved."""
    mids: list[str] = []
    for page in range(1, max_pages + 1):
        url = SEARCH_URL.format(q=quote(keyword), start=fmt_scope(start),
                                end=fmt_scope(end), page=page)
        try:
            resp = s.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning("page %d failed: %s", page, e)
            break
        found = re.findall(r'(?<=mid=")\d{16}', resp.text)
        new = [m for m in dict.fromkeys(found) if m not in mids]
        if not new:
            log.info("page %d: no new ids, stop.", page)
            break
        mids.extend(new)
        log.info("page %d: +%d ids (total %d)", page, len(new), len(mids))
        polite_sleep()
    return mids


def parse_detail(html: str) -> dict | None:
    """Extract the $render_data JSON embedded in m.weibo.cn detail page."""
    m = re.search(r"\$render_data\s*=\s*(\[\s*\{.*?\}\s*\])\s*\[0\]", html, re.S)
    if not m:
        return None
    try:
        status = json.loads(m.group(1))[0]["status"]
    except (json.JSONDecodeError, KeyError, IndexError):
        return None
    user = status.get("user", {})
    text = re.sub(r"<[^>]+>", "", status.get("text", ""))
    retweeted = status.get("retweeted_status") or {}
    created = datetime.strptime(status["created_at"], "%a %b %d %H:%M:%S %z %Y")
    return {
        "user_id": user.get("id"),
        "text": text,
        "retweeted_user_id": (retweeted.get("user") or {}).get("id", ""),
        "mentions": "|".join(re.findall(r"@([\w\-\u4e00-\u9fa5]+)", text)),
        "screen_name": user.get("screen_name"),
        "gender": user.get("gender"),
        "verified": user.get("verified"),
        "verified_type": user.get("verified_type"),
        "verified_type_ext": user.get("verified_type_ext", 0),
        "statuses_count": user.get("statuses_count"),
        "follow_count": user.get("follow_count"),
        "followers_count": user.get("followers_count"),
        "urank": user.get("urank"),
        "is_repost": "retweeted_status" in status,   # 可用于计算 转发/原创 比
        "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
        "reposts_count": status.get("reposts_count"),
        "comments_count": status.get("comments_count"),
        "attitudes_count": status.get("attitudes_count"),
    }


def anonymize(row: dict, salt: str) -> dict:
    """Hash the user id and drop the nickname before anything is saved."""
    uid = str(row["user_id"])
    row["user_id"] = hashlib.sha256((salt + uid).encode()).hexdigest()[:16]
    row["screen_name"] = ""
    return row


def crawl(args) -> None:
    s = make_session()
    start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
    end = datetime.strptime(args.end, "%Y-%m-%d %H:%M")
    mids = get_post_ids(s, args.keyword, start, end, args.max_pages)
    log.info("共找到 %d 条博文，开始抓取详情", len(mids))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    salt = os.getenv("ANON_SALT", "")
    ok = 0
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, mid in enumerate(mids, 1):
            url = DETAIL_URL.format(mid=mid)
            try:
                resp = s.get(url, timeout=15)
                if resp.status_code in (403, 418, 429):
                    log.error("被限流 (HTTP %d)，请稍后再试或更换 Cookie", resp.status_code)
                    break
                row = parse_detail(resp.text)
            except requests.RequestException as e:
                log.warning("[%d/%d] %s 请求失败: %s", i, len(mids), mid, e)
                continue
            if row is None:
                log.warning("[%d/%d] %s 解析失败（微博已删除/无权限/页面结构变化）", i, len(mids), mid)
                continue
            row["post_url"] = url
            if args.anonymize:
                row = anonymize(row, salt)
            writer.writerow(row)
            ok += 1
            log.info("[%d/%d] saved %s", i, len(mids), mid)
            polite_sleep()
    log.info("完成：成功 %d / %d，输出 %s", ok, len(mids), out)


def main() -> None:
    p = argparse.ArgumentParser(description="Weibo topic crawler (academic use only)")
    p.add_argument("--keyword", required=True, help="搜索关键词/话题")
    p.add_argument("--start", required=True, help='起始时间, 如 "2024-05-03 19:00"')
    p.add_argument("--end", required=True, help='结束时间, 如 "2024-05-03 20:00"')
    p.add_argument("--max-pages", type=int, default=20)
    p.add_argument("--out", default="data/raw/posts.csv")
    p.add_argument("--anonymize", action="store_true", help="哈希用户ID并丢弃昵称")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    crawl(p.parse_args())


if __name__ == "__main__":
    main()
