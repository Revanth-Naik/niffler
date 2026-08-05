"""Look up any ticker: price history, a live prediction, and — if this
ticker has logged history — how past predictions compared to reality."""

from __future__ import annotations

import sys
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = WEB_DIR.parent
sys.path.insert(0, str(WEB_DIR))
sys.path.insert(0, str(ROOT_DIR))

import plotly.graph_objects as go
import streamlit as st

from data_helpers import get_prediction
from theme import inject_css, render_header, render_prediction_card, themed_line_layout, GOLD_BRIGHT, EMERALD
from src.tracking.logger import load_log

st.set_page_config(page_title="Niffler — stock lookup", page_icon=":gem:", layout="wide")
inject_css()
render_header("check any creature's scent")

ticker = st.text_input("Ticker", value="AAPL", placeholder="e.g. TSLA").strip().upper()
period = st.select_slider("History window", options=["1mo", "3mo", "6mo", "1y"], value="3mo")

if not ticker:
    st.stop()

with st.spinner(f"Sniffing out {ticker}..."):
    result = get_prediction(ticker, period=period)

if not result["is_live"]:
    st.caption(f"Live data wasn't available for {ticker} — showing illustrative data instead.")

col1, col2 = st.columns([2, 1])

with col1:
    df = result["history"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode="lines", name="Close",
        line=dict(color=GOLD_BRIGHT, width=2),
    ))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title=f"{ticker} — closing price")
    st.plotly_chart(themed_line_layout(fig), width="stretch")

with col2:
    st.markdown(
        render_prediction_card(
            result["ticker"], result["price"], result["predicted_pct"],
            result["confidence"], result["rationale"], source=result.get("source", "heuristic"),
        ),
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown("### Predicted vs. actual — this ticker's track record")

log = load_log()
ticker_log = log[(log["ticker"] == ticker) & (log["actual_close"].notna())].sort_values("date")

if ticker_log.empty:
    st.info(
        f"No resolved predictions logged yet for {ticker}. Run `scripts/run_predictions.py` before open and "
        "`scripts/record_actuals.py` after close on days you track this ticker to build history here."
    )
else:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=ticker_log["date"], y=ticker_log["predicted_pct"], mode="lines+markers",
        name="Predicted %", line=dict(color=GOLD_BRIGHT, width=2, dash="dash"),
    ))
    fig2.add_trace(go.Scatter(
        x=ticker_log["date"], y=ticker_log["actual_pct"], mode="lines+markers",
        name="Actual %", line=dict(color=EMERALD, width=2),
    ))
    fig2.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(themed_line_layout(fig2), width="stretch")

    hits = ticker_log["hit"].sum()
    total = len(ticker_log)
    st.caption(f"{hits} of {total} predictions for {ticker} matched actual direction ({hits/total*100:.0f}%).")
