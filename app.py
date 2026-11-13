# -*- coding: utf-8 -*-
"""微博账号和文本检测演示：streamlit run app.py
模型位置通过环境变量 ABUSE_MODEL 指定（本地目录或 Hugging Face Hub 模型名）。"""
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="微博文本与账号检测", page_icon="🛡️")
st.title("微博文本与账号检测")


@st.cache_resource
def load(model_path: str):
    from abuse_detector.detector import AbuseDetector

    return AbuseDetector(model_path)


@st.cache_resource
def load_water_army(model_path: Path, modified_ns: int):
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(model_path)
    return model


account_tab, text_tab = st.tabs(["XGBoost 水军风险", "RoBERTa 文本检测"])

with account_tab:
    st.header("账号水军风险 · XGBoost")
    st.caption("使用历史检测标签训练的基线。实时采样与训练样本的采集窗口不同，分数仅供复核。")
    with st.form("water_army"):
        uid = st.text_input("微博 UID（纯数字）").strip()
        upload = st.file_uploader("或上传旧版博文 CSV，使用 UID 选择要评分的账号", type=["csv"])
        cutoff = st.slider("疑似水军分数阈值", 0.0, 1.0, 0.5, 0.05)
        submitted = st.form_submit_button("评估账号")

    if submitted:
        import re

        import pandas as pd
        import requests
        from xgboost.core import XGBoostError

        from crawler.user_profile_crawler import RateLimited, make_session, scoring_posts
        from water_army import POST_COLUMNS, build_account_features, score_accounts

        configured = Path(os.getenv("WATER_ARMY_MODEL") or "models/water_army.json")
        model_path = ROOT / configured
        try:
            if not re.fullmatch(r"[0-9]{1,20}", uid):
                raise ValueError("请输入纯数字微博 UID。")
            model = load_water_army(model_path, model_path.stat().st_mtime_ns)
            with st.spinner("正在读取博文并计算分数…"):
                if upload is not None:
                    posts = pd.read_csv(upload, dtype="string", encoding="utf-8-sig", usecols=POST_COLUMNS)
                    posts = posts.loc[posts["楼主ID"].str.strip().eq(uid)]
                    if posts.empty:
                        raise ValueError("CSV 中找不到该 UID。")
                else:
                    with make_session() as session:
                        posts = pd.DataFrame(scoring_posts(session, uid))
                features = build_account_features(posts)
                result = score_accounts(model, features, cutoff).iloc[0]
            st.metric("疑似水军分数", f'{result["bot_score"]:.1%}')
            st.info("达到疑似阈值，建议人工复核。" if result["bot"] else "本次评分低于疑似阈值。")
            st.caption(f"参与评分的博文：{len(posts)} 条")
        except RateLimited as exc:
            st.warning(f"微博限制了访问（{exc}）。可上传 CSV 进行离线评分。")
        except requests.RequestException:
            st.error("微博请求失败。请检查网络，或上传 CSV 进行离线评分。")
        except FileNotFoundError:
            st.error("找不到 XGBoost 模型，请按 README 训练模型，或设置 WATER_ARMY_MODEL。")
        except (ValueError, XGBoostError) as exc:
            st.error(str(exc))

with text_tab:
    st.header("中文攻击性文本 · RoBERTa")
    st.caption("基于微调的中文 RoBERTa。仅用于学术演示，结果不应作为对任何个人的定性依据。")

    examples = {"（自己输入）": "", "正常": "今天天气真好，适合出去走走。", "含攻击性": "你这种人真是无药可救。"}
    choice = st.selectbox("示例", list(examples))
    text = st.text_area("输入一段文本", value=examples[choice], height=120)
    threshold = st.slider("判定阈值", 0.1, 0.9, 0.5, 0.05)

    if st.button("检测") and text.strip():
        model_path = os.getenv("ABUSE_MODEL") or str(ROOT / "models" / "cyberbullying_roberta")
        try:
            with st.spinner("正在加载模型并检测…"):
                prob = float(load(model_path).predict_proba([text])[0])
        except (OSError, ValueError) as exc:
            st.error("无法加载 RoBERTa 模型，请检查 .env 中的 ABUSE_MODEL 是否指向完整的模型目录。")
            st.caption(str(exc))
            st.stop()
        st.metric("辱骂概率", f"{prob:.1%}")
        (st.error if prob > threshold else st.success)("判定：疑似网络暴力语言" if prob > threshold else "判定：正常内容")
