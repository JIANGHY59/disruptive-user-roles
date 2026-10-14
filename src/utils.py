import hashlib
import os


def hash_id(uid, salt: str | None = None) -> str:
    """与 weibo_crawler.anonymize 使用同一公式，保证两个表匿名化后仍可 join。"""
    salt = os.getenv("ANON_SALT", "") if salt is None else salt
    return hashlib.sha256((salt + str(uid)).encode()).hexdigest()[:16]
