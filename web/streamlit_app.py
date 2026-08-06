"""Niffler home page: watchlist snapshot + overall accuracy + nightly whisper.

Run from the niffler/ project root:
    streamlit run web/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
ROOT_DIR = WEB_DIR.parent
sys.path.insert(0, str(WEB_DIR))
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from data_helpers import ensure_bootstrap, get_watchlist_predictions
from dumbledore_widget import render_floating_widget
from theme import inject_css, render_header, render_prediction_card, render_whisper, hoard_gauge
from src.config import DEFAULT_TICKERS
from src.prediction.ml_model import is_available as ml_is_available
from src.tracking.logger import load_log, accuracy_stats

st.set_page_config(page_title="Niffler", page_icon=":gem:", layout="wide")
ensure_bootstrap()
inject_css()
render_header()
render_floating_widget()

if ml_is_available():
    st.caption("The AI correction model is active — predictions below are its output, not just the raw heuristic. See **Model insights** in the sidebar.")
else:
    st.caption("Running on the heuristic model. Run `scripts/train_model.py` to train the AI correction model.")

st.markdown("### Tonight's burrow")
st.caption("Your watchlist, predicted next-session move. Edit `DEFAULT_TICKERS` in `src/config.py` to change it.")

predictions = get_watchlist_predictions(DEFAULT_TICKERS)
any_synthetic = any(not p["is_live"] for p in predictions)

cols = st.columns(len(predictions) if predictions else 1)
for col, pred in zip(cols, predictions):
    with col:
        st.markdown(
            render_prediction_card(
                pred["ticker"], pred["price"], pred["predicted_pct"], pred["confidence"], pred["rationale"],
                source=pred.get("source", "heuristic"),
            ),
            unsafe_allow_html=True,
        )

if any_synthetic:
    st.caption("Some tickers above are showing illustrative data — live fetch wasn't available.")

st.markdown("---")

left, right = st.columns([1, 2])

with left:
    st.markdown("### The hoard")
    stats = accuracy_stats()
    if stats["total"] == 0:
        st.info(
            "No resolved predictions logged yet. Run `scripts/run_predictions.py` before market open and "
            "`scripts/record_actuals.py` after close to start building accuracy history."
        )
        st.plotly_chart(hoard_gauge(0), width="stretch")
    else:
        st.plotly_chart(hoard_gauge(stats["hit_rate"]), width="stretch")
        st.caption(f"{stats['correct']} of {stats['total']} predictions matched actual direction.")

with right:
    st.markdown("### Niffler's nightly whisper")
    up_count = sum(1 for p in predictions if p["direction"] == "up")
    down_count = sum(1 for p in predictions if p["direction"] == "down")
    if up_count > down_count:
        whisper = f"The hoard leans golden tonight — {up_count} of {len(predictions)} tracked creatures sniffed out an upward scent."
    elif down_count > up_count:
        whisper = f"The burrow feels uneasy tonight — {down_count} of {len(predictions)} tracked creatures are pulling back."
    else:
        whisper = "The hoard is quiet tonight — no strong scent in either direction across the watchlist."
    render_whisper(whisper)

    st.markdown("")
    st.markdown(
        "Use the pages in the sidebar to look up any ticker, scan the top 50 predicted movers, "
        "or see how predictions have tracked against reality over time."
    )

st.markdown("---")
st.caption("Niffler is a prototype. Predictions are illustrative, not investment advice.")
