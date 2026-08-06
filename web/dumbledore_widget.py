"""Floating Dumbledore chat widget, shared across every page.

Call render_floating_widget() near the end of each page's script (after
ensure_bootstrap()/inject_css()/render_header()). Renders as a small
button pinned to the bottom-right corner of the viewport that expands into
a chat panel — st.popover natively, with CSS restricted to *positioning*
(fixed corner placement) rather than colors/theming, which is the one
exception the project's Streamlit skill calls out as legitimate: the rest
of Niffler's look comes from theme.py's inject_css() and
.streamlit/config.toml, this file only pins the widget to a fixed screen
position, something no native Streamlit layout primitive does on its own.
"""

from __future__ import annotations

import streamlit as st

from data_helpers import get_ml_model, get_prediction
from src.chatbot.dumbledore import answer
from src.config import DEFAULT_TICKERS
from src.prediction.universe import get_sp500_tickers
from src.tracking.logger import load_log

WIDGET_KEY = "dumbledore_widget"
GREETING = (
    "Hello — I'm Dumbledore. Ask me why a ticker's prediction looks the way it does, how the "
    "accuracy track record is holding up, or what terms like RSI mean."
)


def _inject_widget_css() -> None:
    st.html(
        f"""
        <style>
        div.st-key-{WIDGET_KEY} {{
            position: fixed;
            right: 22px;
            bottom: 68px;
            z-index: 9999;
            width: auto !important;
        }}
        div.st-key-{WIDGET_KEY} [data-testid="stPopoverBody"] {{
            width: 360px;
        }}
        </style>
        """
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _known_tickers() -> set[str]:
    try:
        universe = set(get_sp500_tickers())
    except Exception:
        universe = set()
    return set(DEFAULT_TICKERS) | universe


def render_floating_widget() -> None:
    _inject_widget_css()

    if "dumbledore_messages" not in st.session_state:
        st.session_state.dumbledore_messages = [{"role": "assistant", "content": GREETING}]

    with st.popover("\U0001F9D9 Ask Dumbledore", key=WIDGET_KEY):
        st.caption(
            "Free & local — no external AI API. Explains predictions/accuracy from Niffler's own "
            "data; won't say when to buy or sell. Tip: write tickers in CAPS, e.g. AAPL."
        )

        history_box = st.container(height=320)
        with history_box:
            for msg in st.session_state.dumbledore_messages:
                avatar = "\U0001F9D9" if msg["role"] == "assistant" else None
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

        question = st.chat_input("Ask Dumbledore...", key=f"{WIDGET_KEY}_input")
        if question:
            st.session_state.dumbledore_messages.append({"role": "user", "content": question})
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
            st.rerun()
