# -*- coding: utf-8 -*-
"""用微调好的模型给爬取的博文打标签，输出 abuse_prob / is_abusive 两列（供 features.py 使用）。

python src/abuse_detector/predict_csv.py --model cyberbullying_roberta2 \
    --in data/raw/posts.csv --out data/raw/posts_scored.csv
旧版数据的文本列叫“话题内容”，可加 --text-col 话题内容
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from abuse_detector.detector import AbuseDetector  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--text-col", default="text")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args()

    df = pd.read_csv(a.inp, encoding="utf-8-sig")
    det = AbuseDetector(a.model)
    df["abuse_prob"] = det.predict_proba(df[a.text_col].tolist(), a.batch_size).round(4)
    df["is_abusive"] = (df["abuse_prob"] > a.threshold).astype(int)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"saved {len(df)} rows -> {a.out}  (abusive: {df['is_abusive'].mean():.1%})")


if __name__ == "__main__":
    main()
