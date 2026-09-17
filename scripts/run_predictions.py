#!/usr/bin/env python3
"""Run before market open: generate today's predictions for the watchlist
(or a custom ticker list) and log them.

Usage:
    python scripts/run_predictions.py
    python scripts/run_predictions.py --tickers AAPL,MSFT,GOOGL
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas_market_calendars as mcal
import yfinance as yf

from src.config import TRACKED_TICKERS
from src.prediction.ml_model import is_available, load_meta, load_model
from src.prediction.predictor import predict
from src.tracking.logger import log_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and log today's predictions.")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers.")
    parser.add_argument(
        "--skip-holiday-check", action="store_true",
        help="Log anyway even if today isn't a NYSE trading day (mainly for manual/offline testing).",
    )
    return parser.parse_args()


def is_market_open_today() -> bool:
    """True if today is a NYSE trading day. The predict.yml schedule only
    excludes weekends — it has no idea about Labor Day, Thanksgiving,
    Christmas, etc. Without this check, a holiday morning still gets
    predictions logged against the previous close, and since the market
    never opens that day, record_actuals.py can never find a closing price
    for that exact date — the prediction is stuck pending forever, not
    just delayed. Checking here means nothing gets logged for a day that
    was never going to resolve."""
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=date.today(), end_date=date.today())
    return not schedule.empty


def main() -> None:
    args = parse_args()

    if not args.skip_holiday_check and not is_market_open_today():
        print(f"{date.today()} is not a NYSE trading day (holiday or weekend) — skipping, nothing logged.")
        return

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else TRACKED_TICKERS
    print(f"Predicting {len(tickers)} ticker(s)...")

    model, meta = (load_model(), load_meta()) if is_available() else (None, None)
    if model is not None:
        print("Using trained ML correction model.")
    else:
        print("No trained model found — using the heuristic. Run scripts/train_model.py to train one.")

    rows = []
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(period="3mo", interval="1d")
            if df.empty:
                print(f"  {ticker}: no data, skipping")
                continue
            result = predict(df, model=model, meta=meta)
            prev_close = float(df["Close"].iloc[-1])
            rows.append({
                "ticker": ticker,
                "prev_close": prev_close,
                "predicted_pct": result["predicted_pct"],
                "predicted_direction": result["direction"],
                "confidence": result["confidence"],
                "source": result["source"],
            })
            print(f"  {ticker}: predicted {result['predicted_pct']:+.2f}% ({result['direction']}, {result['confidence']}% confidence, {result['source']})")
        except Exception as exc:
            print(f"  {ticker}: failed ({exc})")

    if not rows:
        print("No predictions generated.")
        sys.exit(1)

    written = log_predictions(rows)
    if written:
        print(f"Logged {written} new prediction(s).")
    else:
        print("Nothing new to log — today's predictions were already recorded.")


if __name__ == "__main__":
    main()
