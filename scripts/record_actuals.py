#!/usr/bin/env python3
"""Run after market close: fetch today's closing prices for any tickers
with unresolved predictions, and record how the prediction did.

Usage:
    python scripts/record_actuals.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yfinance as yf

from src.tracking.logger import load_log, record_actuals, accuracy_stats


def main() -> None:
    today = date.today()
    log = load_log()
    pending = log[(log["date"].dt.date == today) & (log["actual_close"].isna())]

    if pending.empty:
        print("Nothing to resolve today.")
        return

    closes = {}
    for ticker in pending["ticker"]:
        try:
            df = yf.Ticker(ticker).history(period="1d", interval="1d")
            if not df.empty:
                closes[ticker] = float(df["Close"].iloc[-1])
        except Exception as exc:
            print(f"  {ticker}: failed ({exc})")

    updated = record_actuals(closes, for_date=today)
    print(f"Resolved {updated} predictions for {today}.")

    stats = accuracy_stats()
    if stats["total"]:
        print(f"Running accuracy: {stats['hit_rate']}% over {stats['total']} predictions.")


if __name__ == "__main__":
    main()
