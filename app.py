"""Summarizer tool (demo).  Run with:  streamlit run app.py"""
import streamlit as st

from src.evaluate import rouge_single
from src.textrank import textrank_summarize

st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="wide")
st.title("📝 News & Research Paper Summarizer")
st.caption("Extractive (TextRank) vs Abstractive (BART / T5)")


@st.cache_resource(show_spinner="Loading model...")
def get_model(key: str):
    from src.abstractive import AbstractiveSummarizer
    return AbstractiveSummarizer(key)


with st.sidebar:
    method = st.radio("Method", ["TextRank (extractive)", "BART (abstractive)", "T5 (abstractive)", "Compare all"])
    n_sent = st.slider("TextRank: number of sentences", 1, 10, 3)
    max_len = st.slider("Abstractive: max summary tokens", 40, 250, 128)

text = st.text_area("Paste a news article or paper text", height=300)
reference = st.text_area("Optional: reference summary (to compute ROUGE)", height=100)

if st.button("Summarize", type="primary") and text.strip():
    runs = {}
    if method in ("TextRank (extractive)", "Compare all"):
        runs["TextRank"] = lambda: textrank_summarize(text, n_sent)
    if method in ("BART (abstractive)", "Compare all"):
        runs["BART"] = lambda: get_model("bart").summarize(text, max_length=max_len)
    if method in ("T5 (abstractive)", "Compare all"):
        runs["T5"] = lambda: get_model("t5").summarize(text, max_length=max_len)

    cols = st.columns(len(runs))
    for col, (name, fn) in zip(cols, runs.items()):
        with col:
            st.subheader(name)
            summary = fn()
            st.write(summary)
            st.caption(f"{len(text.split())} → {len(summary.split())} words")
            if reference.strip():
                s = rouge_single(reference, summary)
                st.metric("ROUGE-1 / 2 / L", f"{s['rouge1']} / {s['rouge2']} / {s['rougeL']}")
