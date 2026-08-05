"""What's actually running under the hood: whether the AI correction model
is trained, how it did on its holdout set versus the plain heuristic, which
features it leans on, and — once there's enough real log history — how the
two have compared on real predictions rather than just the training
holdout."""

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
from theme import inject_css, render_header, themed_line_layout, GOLD_BRIGHT, EMERALD
from src.prediction.ml_model import is_available, load_meta
from src.tracking.logger import load_log

st.set_page_config(page_title="Niffler — model insights", page_icon=":gem:", layout="wide")
ensure_bootstrap()
inject_css()
render_header("what's under the hood")

if not is_available():
    st.info(
        "No trained AI correction model yet. Everything is running on the transparent heuristic "
        "(momentum + RSI). Train one with:\n\n"
        "```\npython scripts/train_model.py --synthetic   # quick offline demo\n"
        "python scripts/train_model.py               # real data, scans the S&P 500\n```"
    )
    st.stop()

meta = load_meta()

st.markdown("### Training run")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trained", meta["trained_at"][:10])
c2.metric("Tickers used", len(meta["tickers_used"]))
c3.metric("Training samples", meta["n_train_samples"])
c4.metric("Holdout samples", meta["n_test_samples"])
if meta.get("synthetic"):
    st.caption("This model was trained on synthetic demo data — retrain without `--synthetic` once you have network access for a model that means something.")

st.markdown("---")
st.markdown("### AI vs. heuristic — training holdout")
st.caption(
    "Measured on data the model never trained on. This is the honest comparison — if the AI model "
    "doesn't clearly beat the heuristic here, it isn't actually helping yet."
)

holdout = meta["holdout_metrics"]
h1, h2 = st.columns(2)

with h1:
    fig_mae = go.Figure(go.Bar(
        x=["Heuristic", "AI model"], y=[holdout["heuristic_mae"], holdout["ml_mae"]],
        marker_color=[EMERALD, GOLD_BRIGHT],
    ))
    fig_mae.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), title="Mean absolute error (lower is better)")
    st.plotly_chart(themed_line_layout(fig_mae), width="stretch")

with h2:
    fig_hit = go.Figure(go.Bar(
        x=["Heuristic", "AI model"], y=[holdout["heuristic_hit_rate"], holdout["ml_hit_rate"]],
        marker_color=[EMERALD, GOLD_BRIGHT],
    ))
    fig_hit.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), title="Direction hit rate % (higher is better)", yaxis_range=[0, 100])
    st.plotly_chart(themed_line_layout(fig_hit), width="stretch")

beat_mae = holdout["ml_mae"] < holdout["heuristic_mae"]
beat_hit = holdout["ml_hit_rate"] > holdout["heuristic_hit_rate"]
if beat_mae and beat_hit:
    st.success("On this holdout, the AI model beat the heuristic on both error and direction — it's doing real correction work.")
elif not beat_mae and not beat_hit:
    st.warning("On this holdout, the AI model didn't beat the heuristic. More history, more tickers, or a longer `--period` usually helps.")
else:
    st.info("Mixed result — the AI model improved one metric but not the other. Worth more training data before trusting it fully.")

st.markdown("---")
st.markdown("### What the model is paying attention to")
importances = meta["feature_importances"]
sorted_features = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
fig_imp = go.Figure(go.Bar(
    x=[v for _, v in sorted_features], y=[k for k, _ in sorted_features],
    orientation="h", marker_color=GOLD_BRIGHT,
))
fig_imp.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Relative importance")
st.plotly_chart(themed_line_layout(fig_imp), width="stretch")
st.caption(
    "`heuristic_pred_pct` / `heuristic_confidence` are the plain heuristic's own output, fed in as "
    "features — high importance here means the AI model is mostly learning *when to trust or override* "
    "the heuristic, rather than ignoring it."
)

st.markdown("---")
st.markdown("### AI vs. heuristic — real logged predictions")

log = load_log()
resolved = log.dropna(subset=["hit"])
if resolved.empty or "source" not in resolved.columns or resolved["source"].nunique() < 1:
    st.info(
        "No resolved real-world predictions logged yet. Run `scripts/run_predictions.py` before open and "
        "`scripts/record_actuals.py` after close — once you have logged predictions from both before and "
        "after this model was trained, this section will compare them head to head."
    )
else:
    by_source = resolved.groupby("source").agg(
        predictions=("hit", "count"),
        hit_rate=("hit", lambda s: round(s.mean() * 100, 1)),
    )
    st.dataframe(
        by_source.rename(columns={"predictions": "Predictions", "hit_rate": "Hit rate %"}),
        width="stretch",
    )
    if by_source["predictions"].min() < 20:
        st.caption("Small sample so far — take this comparison with a grain of salt until more days accumulate.")
