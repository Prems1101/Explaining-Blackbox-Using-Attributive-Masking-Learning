import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from explain.aml_explainer import AMLExplainer
from explain.faml_explainer import FAMLExplainer
from evaluation.metrics import (
    sufficiency_score,
    comprehensiveness_score,
    sparsity_score,
)

st.set_page_config(page_title="AML Explainability", page_icon="🔍", layout="centered")

# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    paml = AMLExplainer(model_path="paml_best.pt")
    faml = FAMLExplainer(model_path="paml_best.pt")
    return paml, faml

paml, faml = load_models()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 AML Explainability Demo")
st.caption("Attributive Masking Learning — EMNLP 2024")

mode = st.radio("Select Model:", ["fAML", "pAML"], horizontal=True)
text = st.text_area("Enter text:", "avengers is a best movie", height=80)

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    explainer = paml if mode == "pAML" else faml

    with st.spinner(f"Running {mode}…"):
        result = explainer.explain(text)

    pred  = result["prediction"]
    label = pred["label"]
    conf  = pred["score"]

    # ── Prediction ────────────────────────────────────────────────────────────
    st.subheader("📊 Prediction")
    if label == "POSITIVE":
        st.success(f"{mode} → Label: **{label}**")
    else:
        st.error(f"{mode} → Label: **{label}**")
    st.metric("Confidence", f"{conf:.4f}")

    # ── Word highlight ────────────────────────────────────────────────────────
    st.subheader("🧠 Important Words")
    st.caption("Color intensity reflects attribution score: "
               "🔴 high · 🟠 medium · ⬜ low")

    html = ""
    for word, score in result["word_importance_original"]:
        if score > 0.6:
            bg, fg = "#ff4b4b", "white"
        elif score > 0.35:
            bg, fg = "#ffa500", "white"
        elif score > 0.15:
            bg, fg = "#ffe0a0", "#333"
        else:
            bg, fg = "transparent", "#999"
        html += (
            f"<span style='background:{bg};color:{fg};"
            f"border-radius:4px;padding:2px 4px;margin:2px;display:inline-block'>"
            f"{word}</span> "
        )
    st.markdown(html, unsafe_allow_html=True)

    # ── Top tokens table ──────────────────────────────────────────────────────
    st.subheader("🔥 Top Important Words")
    top = result["word_importance_sorted"][:10]
    cols = st.columns(2)
    for idx, (word, score) in enumerate(top):
        bar = "█" * int(score * 20)
        cols[idx % 2].write(f"**{word}** — {score:.4f}  `{bar}`")

    # ── Evaluation metrics ────────────────────────────────────────────────────
    st.subheader("📈 Faithfulness Metrics")
    with st.spinner("Computing metrics…"):
        suff = sufficiency_score(explainer, text)
        comp = comprehensiveness_score(explainer, text)
        spars = sparsity_score(result)

    c1, c2, c3 = st.columns(3)
    c1.metric("Sufficiency ↓", f"{suff:.4f}",
              help="p(y|x) - p(y|top-k kept). Lower = important tokens preserve prediction.")
    c2.metric("Comprehensiveness ↑", f"{comp:.4f}",
              help="p(y|x) - p(y|top-k removed). Higher = important tokens drive prediction.")
    c3.metric("Sparsity", str(spars),
              help="# of tokens with score > 0.3. Fewer = more focused explanation.")

    # ── All word scores (collapsed) ───────────────────────────────────────────
    with st.expander("All word attribution scores"):
        for word, score in result["word_importance_sorted"]:
            st.write(f"{word}: {score:.4f}")
