#!/usr/bin/env python3
"""Backfill the predictions log with demo history so the Accuracy page and
home page hoard meter have something to show before you've run the real
daily loop for a while.

This runs the real prediction model against synthetic (not real market)
price series, walking forward day by day so the resulting log is
methodologically consistent — just not based on real prices. Safe to run
once when setting the project up; re-running clears and rebuilds it.

Usage:
    python scripts/seed_demo_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import DEFAULT_TICKERS, PROCESSED_DATA_DIR
from src.prediction.model import predict_next_session
from src.prediction.synthetic import synthetic_history
from src.tracking.logger import LOG_PATH, COLUMNS

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


def main() -> None:
    all_rows = []
    for ticker in DEFAULT_TICKERS:
        all_rows.extend(build_demo_rows(ticker))

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_PATH, index=False)
    print(f"Seeded {len(df)} demo predictions across {len(DEFAULT_TICKERS)} tickers to {LOG_PATH}")
    print(f"Demo hit rate: {df['hit'].mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
