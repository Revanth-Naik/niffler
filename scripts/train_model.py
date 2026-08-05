#!/usr/bin/env python3
"""Train (or retrain) Niffler's ML correction model.

Pulls historical OHLCV for a ticker universe, builds features (technical
indicators plus the heuristic model's own output), and trains a gradient
boosting regressor to predict next-session % return. Evaluates against the
plain heuristic on a chronological holdout so you can see whether the ML
model is actually adding anything, not just take it on faith — the numbers
print at the end and are saved to models/niffler_gbm_meta.json for the
Model insights page.

This script is the "self-correction" mechanism the project is built
around: re-run it periodically (weekly is reasonable) as more real price
history accumulates, and the model updates to reflect it. Schedule it with
cron, or Niffler's scheduled-task hookup once that's wired up.

Note on the holdout split: examples from every ticker are pooled and split
by date (last ~20% chronologically held out), not per-ticker — with many
tickers this means a handful of examples right at the boundary date can
come from a ticker also present just before it. For a prototype this is an
acceptable approximation; a stricter walk-forward split would isolate this
further.

Usage:
    python scripts/train_model.py                    # default: scan universe, live data
    python scripts/train_model.py --limit 30 --period 1y
    python scripts/train_model.py --synthetic         # offline/demo, no network needed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_TICKERS
from src.prediction.train import PERIOD_DAYS, MODEL_PATH, NotEnoughDataError, train_and_save
from src.prediction.universe import get_sp500_tickers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Niffler's ML correction model.")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers. Defaults to a universe scan.")
    parser.add_argument("--limit", type=int, default=40, help="Max tickers to train on (keeps training time reasonable).")
    parser.add_argument("--period", type=str, default="1y", choices=list(PERIOD_DAYS), help="History window per ticker.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of live yfinance (offline/demo).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    elif args.synthetic:
        tickers = DEFAULT_TICKERS
    else:
        tickers = get_sp500_tickers()[: args.limit]

    print(f"Training on {len(tickers)} tickers, period={args.period}, synthetic={args.synthetic}")

    try:
        meta = train_and_save(tickers, period=args.period, synthetic=args.synthetic)
    except NotEnoughDataError as exc:
        print(str(exc))
        sys.exit(1)

    holdout = meta["holdout_metrics"]
    print(f"Saved model to {MODEL_PATH}")
    print(f"Holdout MAE      — ML: {holdout['ml_mae']:.3f}   heuristic: {holdout['heuristic_mae']:.3f}")
    print(f"Holdout hit rate — ML: {holdout['ml_hit_rate']:.1f}%   heuristic: {holdout['heuristic_hit_rate']:.1f}%")
    if holdout["ml_mae"] >= holdout["heuristic_mae"] and holdout["ml_hit_rate"] <= holdout["heuristic_hit_rate"]:
        print("Note: on this holdout the ML model didn't beat the plain heuristic. "
              "More/longer history usually helps — try a bigger --limit or --period.")


if __name__ == "__main__":
    main()
