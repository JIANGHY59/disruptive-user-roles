# -*- coding: utf-8 -*-
"""微调中文 RoBERTa 做辱骂文本二分类。

python src/abuse_detector/train.py --data data/corpus.csv --out models/cyberbullying_roberta
语料 CSV 需包含 text、label(0/1) 两列。默认超参与海报中的“微调后”配置一致。
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess import clean_text  # noqa: E402


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts, self.labels, self.tok, self.max_length = texts, labels, tokenizer, max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], max_length=self.max_length, padding="max_length",
                       truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].flatten(),
                "attention_mask": enc["attention_mask"].flatten(),
                "label": torch.tensor(self.labels[i], dtype=torch.long)}


def split_data(path, text_col, label_col, seed):
    df = pd.read_csv(path, encoding="utf-8-sig")[[text_col, label_col]].dropna()
    texts = [clean_text(t) for t in df[text_col]]
    labels = df[label_col].astype(int).tolist()
    # 8:1:1 分层划分，保证三份里正负样本比例一致
    tr_x, tmp_x, tr_y, tmp_y = train_test_split(texts, labels, test_size=0.2, random_state=seed, stratify=labels)
    va_x, te_x, va_y, te_y = train_test_split(tmp_x, tmp_y, test_size=0.5, random_state=seed, stratify=tmp_y)
    return (tr_x, tr_y), (va_x, va_y), (te_x, te_y)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, true = [], []
    for b in tqdm(loader, desc="eval", leave=False):
        out = model(b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
        preds.extend(out.logits.argmax(-1).cpu().tolist())
        true.extend(b["label"].tolist())
    return true, preds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--model-name", default="hfl/chinese-roberta-wwm-ext-large")
    p.add_argument("--out", default="models/cyberbullying_roberta")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(a.model_name, num_labels=2).to(device)

    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = split_data(a.data, a.text_col, a.label_col, a.seed)
    mk = lambda x, y, shuffle: DataLoader(TextDataset(x, y, tok, a.max_length),
                                          batch_size=a.batch_size, shuffle=shuffle)
    train_loader, val_loader, test_loader = mk(tr_x, tr_y, True), mk(va_x, va_y, False), mk(te_x, te_y, False)

    opt = AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, a.warmup_steps, len(train_loader) * a.epochs)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log, best_f1 = [], -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for b in tqdm(train_loader, desc=f"Epoch {ep}/{a.epochs}"):
            opt.zero_grad()
            loss = model(b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device),
                         labels=b["label"].to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            total += loss.item()
        true, preds = evaluate(model, val_loader, device)
        row = {"epoch": ep, "train_loss": total / len(train_loader),
               "val_acc": accuracy_score(true, preds), "val_f1": f1_score(true, preds)}
        log.append(row)
        print(row)
        if row["val_f1"] > best_f1:  # 只保存验证集 F1 最高的一轮
            best_f1 = row["val_f1"]
            model.save_pretrained(out); tok.save_pretrained(out)
    (out / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    # 测试集只在最后用一次，用来报告最终指标
    best = AutoModelForSequenceClassification.from_pretrained(out).to(device)
    true, preds = evaluate(best, test_loader, device)
    print(classification_report(true, preds, digits=4))
    (out / "test_metrics.json").write_text(json.dumps(
        {"accuracy": accuracy_score(true, preds), "f1": f1_score(true, preds)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
