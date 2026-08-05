#!/usr/bin/env python3
"""CLI entrypoint: pull daily OHLCV data for a list of tickers and save to data/raw/.

Usage:
    python scripts/fetch_daily.py --tickers AAPL,MSFT,GOOGL
    python scripts/fetch_daily.py                      # uses DEFAULT_TICKERS
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_TICKERS, RAW_DATA_DIR
from src.ingestion.yfinance_client import fetch_many


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch daily stock data for Nifler.")
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers, e.g. AAPL,MSFT,GOOGL. Defaults to DEFAULT_TICKERS.",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="5d",
        help="yfinance period, e.g. 1d, 5d, 1mo (default: 5d)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else DEFAULT_TICKERS

    print(f"Fetching data for: {', '.join(tickers)}")
    df = fetch_many(tickers, period=args.period)

    if df.empty:
        print("No data returned. Check ticker symbols and network access.")
        sys.exit(1)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RAW_DATA_DIR / f"ohlcv_{timestamp}.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
