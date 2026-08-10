"""How Niffler's predictions have actually done: hit rate over time,
predicted vs. actual scatter, error distribution, per-ticker breakdown.

All of this reads from data/processed/predictions_log.csv, which is built
by running scripts/run_predictions.py before market open and
scripts/record_actuals.py after close.
"""

from __future__ import annotations

import sys
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = WEB_DIR.parent
sys.path.insert(0, str(WEB_DIR))
sys.path.insert(0, str(ROOT_DIR))

import plotly.graph_objects as go
import streamlit as st

from data_helpers import ensure_bootstrap
from dumbledore_widget import render_floating_widget
from theme import inject_css, render_header, themed_line_layout, hoard_gauge, GOLD_BRIGHT, EMERALD, MAROON
from src.tracking.logger import load_log, accuracy_stats

st.set_page_config(page_title="Niffler — accuracy", page_icon=":gem:", layout="wide")
ensure_bootstrap()
inject_css()
render_header("how honest has the hoard been")
render_floating_widget()

log = load_log()
resolved = log.dropna(subset=["hit"]).sort_values("date")

if resolved.empty:
    st.info(
        "No resolved predictions yet. Run `scripts/run_predictions.py` each morning before market open, then "
        "`scripts/record_actuals.py` after close, and this page will fill in over time."
    )
    st.stop()

stats = accuracy_stats(resolved)

top1, top2, top3 = st.columns([1, 1, 1])
with top1:
    st.plotly_chart(hoard_gauge(stats["hit_rate"], "Overall hit rate"), width="stretch")
with top2:
    st.metric("Predictions resolved", stats["total"])
    st.metric("Correct calls", stats["correct"])
with top3:
    st.metric("Mean absolute error", f"{stats['mean_abs_error']} pts")
    st.caption("Average gap between predicted % move and actual % move, regardless of direction.")

st.markdown("---")
st.markdown("### Hit rate over time")

daily = resolved.groupby(resolved["date"].dt.date)["hit"].mean().reset_index()
daily["hit_rate"] = (daily["hit"] * 100).round(1)
daily["rolling"] = daily["hit_rate"].rolling(7, min_periods=1).mean().round(1)

fig1 = go.Figure()
fig1.add_trace(go.Bar(x=daily["date"], y=daily["hit_rate"], name="Daily hit rate", marker_color=GOLD_BRIGHT, opacity=0.5))
fig1.add_trace(go.Scatter(x=daily["date"], y=daily["rolling"], name="7-day rolling", line=dict(color=EMERALD, width=3)))
fig1.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Hit rate %")
st.plotly_chart(themed_line_layout(fig1), width="stretch")

st.markdown("### Predicted vs. actual — every resolved prediction")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=resolved["predicted_pct"], y=resolved["actual_pct"], mode="markers",
    marker=dict(color=resolved["hit"].map({True: EMERALD, False: MAROON}), size=8, opacity=0.7),
    text=resolved["ticker"], hovertemplate="%{text}<br>predicted %{x:.2f}%<br>actual %{y:.2f}%",
    name="Predictions",
))
axis_min = min(resolved["predicted_pct"].min(), resolved["actual_pct"].min()) - 0.5
axis_max = max(resolved["predicted_pct"].max(), resolved["actual_pct"].max()) + 0.5
fig2.add_trace(go.Scatter(
    x=[axis_min, axis_max], y=[axis_min, axis_max], mode="lines",
    line=dict(color=GOLD_BRIGHT, dash="dot"), name="Perfect prediction",
))
fig2.update_layout(
    height=380, margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Predicted %", yaxis_title="Actual %",
)
st.plotly_chart(themed_line_layout(fig2), width="stretch")
st.caption("Green = direction was correct. Maroon = direction missed. Dotted gold line = where a perfect prediction would land.")

st.markdown("### Error distribution")
resolved = resolved.copy()
resolved["abs_error"] = (resolved["predicted_pct"] - resolved["actual_pct"]).abs()
fig3 = go.Figure(go.Histogram(x=resolved["abs_error"], nbinsx=20, marker_color=GOLD_BRIGHT))
fig3.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Absolute error (points)", yaxis_title="Count")
st.plotly_chart(themed_line_layout(fig3), width="stretch")

st.markdown("### Per-ticker track record")
# "hit" is a nullable boolean column — groupby().agg() with a custom lambda
# on it silently mis-infers the output dtype for single-row groups (common
# early on, before much history has accumulated), producing True/False
# instead of a percentage. Aggregating a plain float copy sidesteps it.
resolved["hit_numeric"] = resolved["hit"].astype(float)
per_ticker = resolved.groupby("ticker").agg(
    predictions=("hit", "count"),
    hit_rate=("hit_numeric", lambda s: round(s.mean() * 100, 1)),
    mean_abs_error=("abs_error", lambda s: round(s.mean(), 2)),
).sort_values("hit_rate", ascending=False)
st.dataframe(
    per_ticker.rename(columns={
        "predictions": "Predictions", "hit_rate": "Hit rate %", "mean_abs_error": "Mean abs error",
    }),
    width="stretch",
)
