# -*- coding: utf-8 -*-
"""单句检测 demo:  streamlit run app.py
模型位置通过环境变量 ABUSE_MODEL 指定（本地目录或 Hugging Face Hub 模型名）。"""
import os
import sys

import streamlit as st

sys.path.insert(0, "src")
from abuse_detector.detector import AbuseDetector  # noqa: E402

st.set_page_config(page_title="Chinese Abusive Text Detector", page_icon="🛡️")
st.title("🛡️ 中文网络暴力文本检测")
st.caption("基于微调的中文 RoBERTa。仅用于学术演示，结果不应作为对任何个人的定性依据。")


@st.cache_resource
def load():
    return AbuseDetector(os.getenv("ABUSE_MODEL", "models/cyberbullying_roberta"))


examples = {"（自己输入）": "", "正常": "今天天气真好，适合出去走走。", "含攻击性": "你这种人真是无药可救。"}
choice = st.selectbox("示例", list(examples))
text = st.text_area("输入一段文本", value=examples[choice], height=120)
threshold = st.slider("判定阈值", 0.1, 0.9, 0.5, 0.05)

if st.button("检测") and text.strip():
    prob = float(load().predict_proba([text])[0])
    st.metric("辱骂概率", f"{prob:.1%}")
    (st.error if prob > threshold else st.success)("判定：疑似网络暴力语言" if prob > threshold else "判定：正常内容")
