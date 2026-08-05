"""Generates demo predicted-vs-actual history using the real prediction
model run against synthetic (not real market) price series.

Used two ways:
1. `scripts/seed_demo_data.py` — explicit, one-time CLI setup step.
2. The web app's first-run bootstrap (see web/data_helpers.py) — so a fresh
   deploy with an empty filesystem (e.g. Streamlit Community Cloud, which
   resets storage on every restart) isn't blank on first load.
"""

from __future__ import annotations

import pandas as pd

from src.config import DEFAULT_TICKERS, PROCESSED_DATA_DIR
from src.prediction.model import predict_next_session
from src.prediction.synthetic import synthetic_history
from src.tracking.logger import COLUMNS, LOG_PATH

SESSIONS_TO_SIMULATE = 60
WARMUP_SESSIONS = 21


def build_demo_rows(ticker: str) -> list[dict]:
    history = synthetic_history(ticker, period_days=WARMUP_SESSIONS + SESSIONS_TO_SIMULATE + 1)
    rows = []

    for i in range(WARMUP_SESSIONS, len(history) - 1):
        window = history.iloc[: i + 1]
        result = predict_next_session(window)

        prev_close = float(window["Close"].iloc[-1])
        next_close = float(history["Close"].iloc[i + 1])
        actual_pct = round(((next_close - prev_close) / prev_close) * 100, 2)
        actual_direction = "up" if actual_pct > 0.05 else "down" if actual_pct < -0.05 else "flat"

        rows.append({
            "date": history.index[i + 1].date(),
            "ticker": ticker,
            "prev_close": round(prev_close, 2),
            "predicted_pct": result["predicted_pct"],
            "predicted_direction": result["direction"],
            "confidence": result["confidence"],
            "source": "heuristic",
            "actual_close": round(next_close, 2),
            "actual_pct": actual_pct,
            "actual_direction": actual_direction,
            "hit": result["direction"] == actual_direction,
        })

    return rows


def seed_demo_log(tickers: list[str] | None = None) -> pd.DataFrame:
    """Overwrites the predictions log with fresh demo history. Returns the
    resulting DataFrame."""
    tickers = tickers or DEFAULT_TICKERS
    all_rows = []
    for ticker in tickers:
        all_rows.extend(build_demo_rows(ticker))

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_PATH, index=False)
    return df
