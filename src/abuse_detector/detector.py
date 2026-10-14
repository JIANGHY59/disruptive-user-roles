# -*- coding: utf-8 -*-
"""辱骂/网络暴力文本检测器：加载微调后的中文 RoBERTa(BERT 结构) 模型做批量推理。"""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess import clean_text  # noqa: E402


class AbuseDetector:
    def __init__(self, model_path: str, device: str | None = None, max_length: int = 128):
        """model_path: 本地目录，或 Hugging Face Hub 上的模型名（如 "用户名/模型名"）。"""
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.max_length = max_length

    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """返回每条文本属于“辱骂”(label=1) 的概率。清洗后为空的文本记为 0。"""
        cleaned = [clean_text(t) for t in texts]
        probs = np.zeros(len(cleaned), dtype=np.float32)
        idx = [i for i, t in enumerate(cleaned) if t]
        for s in range(0, len(idx), batch_size):
            batch_idx = idx[s:s + batch_size]
            enc = self.tokenizer([cleaned[i] for i in batch_idx], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            p = torch.softmax(self.model(**enc).logits, dim=-1)[:, 1].cpu().numpy()
            probs[batch_idx] = p
        return probs

    def predict(self, texts: list[str], threshold: float = 0.5, batch_size: int = 64) -> np.ndarray:
        return (self.predict_proba(texts, batch_size) > threshold).astype(int)
