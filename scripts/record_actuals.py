#!/usr/bin/env python3
"""Run after market close: fetch closing prices for any tickers with
unresolved predictions, and record how each prediction did.

Resolves ANY pending prediction, not just ones dated today — if a run gets
missed (an outage, a bug, the workflow not existing yet), the next run
catches it up automatically instead of leaving it stuck forever.

Usage:
    python scripts/record_actuals.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yfinance as yf

from src.tracking.logger import load_log, record_actuals, accuracy_stats


def main() -> None:
    log = load_log()
    pending = log[log["actual_close"].isna()]

    if pending.empty:
        print("Nothing to resolve.")
        return

    actual_closes: dict[tuple[str, date], float] = {}

    # Group by ticker so each ticker's history is fetched once, covering
    # every pending date for it — not just "today's" close.
    for ticker, group in pending.groupby("ticker"):
        needed_dates = set(group["date"].dt.date)
        oldest = min(needed_dates)
        days_back = max((date.today() - oldest).days + 10, 5)
        try:
            df = yf.Ticker(ticker).history(period=f"{days_back}d", interval="1d")
        except Exception as exc:
            print(f"  {ticker}: failed ({exc})")
            continue
        if df.empty:
            print(f"  {ticker}: no data")
            continue

        for needed_date in needed_dates:
            matches = df[df.index.date == needed_date]
            if not matches.empty:
                actual_closes[(ticker, needed_date)] = float(matches["Close"].iloc[-1])
            else:
                print(f"  {ticker}: no close found for {needed_date} yet (market may not have closed)")

    updated = record_actuals(actual_closes)
    print(f"Resolved {updated} of {len(pending)} pending prediction(s).")

    stats = accuracy_stats()
    if stats["total"]:
        print(f"Running accuracy: {stats['hit_rate']}% over {stats['total']} predictions.")


if __name__ == "__main__":
    main()
