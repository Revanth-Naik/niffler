"""Scan the S&P 500 and rank by the model's predicted next-session return.
Top 50 highest-predicted movers, bullish and bearish."""

from __future__ import annotations

import sys
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = WEB_DIR.parent
sys.path.insert(0, str(WEB_DIR))
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from data_helpers import ensure_bootstrap, scan_universe
from theme import inject_css, render_header
from src.prediction.universe import get_sp500_tickers

st.set_page_config(page_title="Niffler — top 50", page_icon=":gem:", layout="wide")
ensure_bootstrap()
inject_css()
render_header("where the hoard is growing fastest")

st.caption(
    "Predicted next-session return across the S&P 500, ranked highest to lowest. "
    "This is the model's output, not a recommendation — treat it as a starting point for your own research."
)

if st.button("Rescan the hoard"):
    scan_universe.clear()

with st.spinner("Sniffing out the whole hoard — this can take a minute the first time..."):
    tickers = tuple(get_sp500_tickers())
    scanned = scan_universe(tickers)

if scanned.empty:
    st.warning("Couldn't scan the universe right now — check your network connection and try again.")
    st.stop()

any_synthetic = (~scanned["is_live"]).any()
if any_synthetic:
    st.caption("Some tickers below are showing illustrative data — live fetch wasn't available for them.")

display_columns = {
    "ticker": "Ticker", "last_price": "Last price", "predicted_pct": "Predicted %",
    "direction": "Direction", "confidence": "Confidence %", "source": "Model",
}
source_labels = {"ml": "AI", "heuristic": "Heuristic"}

top_bullish = scanned.head(50).drop(columns=["is_live"]).reset_index(drop=True)
top_bullish["source"] = top_bullish["source"].map(source_labels).fillna(top_bullish["source"])
top_bullish.index += 1

st.markdown("### Top 50 — predicted gainers")
st.dataframe(top_bullish.rename(columns=display_columns), width="stretch", height=560)

with st.expander("Show predicted decliners instead"):
    top_bearish = scanned.tail(50).sort_values("predicted_pct").drop(columns=["is_live"]).reset_index(drop=True)
    top_bearish["source"] = top_bearish["source"].map(source_labels).fillna(top_bearish["source"])
    top_bearish.index += 1
    st.dataframe(top_bearish.rename(columns=display_columns), width="stretch", height=560)

st.caption(f"Universe size: {len(tickers)} tickers. Niffler is a prototype — not investment advice.")
