"""Data access helpers shared by the Streamlit pages.

Every function tries live data first (yfinance) and falls back to a
deterministic synthetic series if the fetch fails — no network, ticker
delisted, rate-limited, etc. This keeps the app demoable offline instead of
crashing, while using real data whenever it's available.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

from src.config import DEFAULT_TICKERS
from src.prediction.ml_model import is_available as ml_is_available
from src.prediction.ml_model import load_meta as ml_load_meta
from src.prediction.ml_model import load_model as ml_load_model
from src.prediction.predictor import predict as run_prediction
from src.prediction.synthetic import synthetic_history as _synthetic_history
from src.prediction.train import NotEnoughDataError, train_and_save
from src.tracking.demo_seed import seed_demo_log
from src.tracking.logger import LOG_PATH


@st.cache_resource(show_spinner=False)
def ensure_bootstrap() -> None:
    """Runs once per app process (cached — safe to call from every page).

    On a fresh deploy with an empty filesystem (Streamlit Community Cloud
    resets storage on every restart), there's no predictions log and no
    trained model yet — this generates a demo version of both, the same
    way seed_demo_data.py / train_model.py --synthetic do locally, so the
    app isn't blank on first load. No-ops if real data already exists.
    """
    if not LOG_PATH.exists():
        seed_demo_log()

    if not ml_is_available():
        try:
            train_and_save(DEFAULT_TICKERS, period="1y", synthetic=True)
        except NotEnoughDataError:
            pass  # fine — the app runs perfectly well on the heuristic alone


@st.cache_resource(show_spinner=False)
def get_ml_model():
    """Cached so the trained model is loaded from disk once per session,
    not on every prediction. Streamlit re-runs this if the file changes."""
    if not ml_is_available():
        return None, None
    return ml_load_model(), ml_load_meta()


@st.cache_data(ttl=900, show_spinner=False)
def get_history(ticker: str, period: str = "3mo") -> tuple[pd.DataFrame, bool]:
    """Returns (dataframe, is_live). is_live=False means synthetic fallback."""
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d")
        if df.empty or len(df) < 21:
            raise ValueError("insufficient live data")
        return df, True
    except Exception:
        days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 252}.get(period, 90)
        return _synthetic_history(ticker, days), False


def get_prediction(ticker: str, period: str = "3mo") -> dict:
    df, is_live = get_history(ticker, period=period)
    model, meta = get_ml_model()
    result = run_prediction(df, model=model, meta=meta)
    result["ticker"] = ticker
    result["price"] = float(df["Close"].iloc[-1])
    result["is_live"] = is_live
    result["history"] = df
    return result


def get_watchlist_predictions(tickers: list[str]) -> list[dict]:
    return [get_prediction(t) for t in tickers]


@st.cache_data(ttl=1800, show_spinner=False)
def scan_universe(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Run the prediction model across a whole ticker universe and rank by
    predicted return. Cached for 30 minutes since this can be slow for
    large universes."""
    rows = []
    for ticker in tickers:
        try:
            result = get_prediction(ticker)
            rows.append({
                "ticker": ticker,
                "last_price": round(result["price"], 2),
                "predicted_pct": result["predicted_pct"],
                "direction": result["direction"],
                "confidence": result["confidence"],
                "is_live": result["is_live"],
                "source": result.get("source", "heuristic"),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["ticker", "last_price", "predicted_pct", "direction", "confidence", "is_live", "source"])

    df = pd.DataFrame(rows)
    return df.sort_values("predicted_pct", ascending=False).reset_index(drop=True)
