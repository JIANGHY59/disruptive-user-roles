# -*- coding: utf-8 -*-
"""训练和推理共用同一个清洗函数，避免两边预处理不一致。"""
import re


def clean_text(text) -> str:
    """去网址、@用户、#话题#，只保留中英文、数字和空白（与训练时的清洗保持一致）。"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"#.*?#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5\s]", "", text)
    return text.strip()
