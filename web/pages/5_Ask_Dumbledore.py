"""Chat with Dumbledore — Niffler's in-app guide.

Explains predictions, accuracy, and the model's own reasoning, grounded in
Niffler's real data. Free and fully local (no external AI API) — answers
come from pattern matching + retrieval + templates, not a true language
model. Will not give buy/sell trading advice; see
src/chatbot/dumbledore.py for why.
"""

from __future__ import annotations

import sys
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = WEB_DIR.parent
sys.path.insert(0, str(WEB_DIR))
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from data_helpers import ensure_bootstrap, get_ml_model, get_prediction
from theme import inject_css, render_header
from src.chatbot.dumbledore import answer
from src.config import DEFAULT_TICKERS
from src.prediction.universe import get_sp500_tickers
from src.tracking.logger import load_log

st.set_page_config(page_title="Niffler — ask Dumbledore", page_icon=":gem:", layout="wide")
ensure_bootstrap()
inject_css()
render_header("ask the wise old wizard")

st.caption(
    "Dumbledore answers using Niffler's own data — live predictions, the accuracy log, and the AI "
    "model's own stats. Free and fully local (no external AI API, no cost) — and it won't tell you "
    "when to buy or sell, only what the model's signals show. Tip: write tickers in CAPS, e.g. AAPL."
)


@st.cache_data(ttl=3600, show_spinner=False)
def _known_tickers() -> set[str]:
    try:
        universe = set(get_sp500_tickers())
    except Exception:
        universe = set()
    return set(DEFAULT_TICKERS) | universe


if "dumbledore_messages" not in st.session_state:
    st.session_state.dumbledore_messages = [
        {
            "role": "assistant",
            "content": (
                "Hello — I'm Dumbledore, Niffler's resident guide. Ask me why a ticker's prediction "
                "looks the way it does, how the accuracy track record is holding up, or what terms "
                "like RSI mean."
            ),
        }
    ]

for msg in st.session_state.dumbledore_messages:
    avatar = "\U0001F9D9" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

question = st.chat_input('Ask Dumbledore something, e.g. "Why is AAPL predicted to move?"')
if question:
    st.session_state.dumbledore_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Consulting the pensieve..."):
        _, meta = get_ml_model()
        reply = answer(
            question,
            known_tickers=_known_tickers(),
            get_prediction=get_prediction,
            load_log=load_log,
            ml_meta=meta,
        )

    st.session_state.dumbledore_messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant", avatar="\U0001F9D9"):
        st.markdown(reply)

st.markdown("---")
st.caption(
    "Dumbledore explains what Niffler's models see — it is not investment advice, and it will "
    "decline questions about when to buy or sell."
)
