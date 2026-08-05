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
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.config import DEFAULT_TICKERS, MODEL_DIR
from src.prediction.features import FEATURE_COLUMNS, build_training_examples
from src.prediction.synthetic import synthetic_history
from src.prediction.universe import get_sp500_tickers

MODEL_PATH = MODEL_DIR / "niffler_gbm.joblib"
META_PATH = MODEL_DIR / "niffler_gbm_meta.json"

PERIOD_DAYS = {"3mo": 90, "6mo": 180, "1y": 252, "2y": 504}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Niffler's ML correction model.")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers. Defaults to a universe scan.")
    parser.add_argument("--limit", type=int, default=40, help="Max tickers to train on (keeps training time reasonable).")
    parser.add_argument("--period", type=str, default="1y", choices=list(PERIOD_DAYS), help="History window per ticker.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of live yfinance (offline/demo).")
    return parser.parse_args()


def gather_training_data(tickers: list[str], period: str, synthetic: bool) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        try:
            if synthetic:
                df = synthetic_history(ticker, period_days=PERIOD_DAYS[period])
            else:
                df = yf.Ticker(ticker).history(period=period, interval="1d")
            if df.empty or len(df) < 60:
                continue
            examples = build_training_examples(df)
            if examples.empty:
                continue
            examples["ticker"] = ticker
            frames.append(examples)
        except Exception as exc:
            print(f"  {ticker}: skipped ({exc})")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def main() -> None:
    args = parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    elif args.synthetic:
        tickers = DEFAULT_TICKERS
    else:
        tickers = get_sp500_tickers()[: args.limit]

    print(f"Training on {len(tickers)} tickers, period={args.period}, synthetic={args.synthetic}")
    data = gather_training_data(tickers, args.period, args.synthetic)

    if len(data) < 200:
        print(f"Only gathered {len(data)} training rows — need at least 200. Try a longer --period, more tickers, or --synthetic.")
        sys.exit(1)

    cutoff = int(len(data) * 0.8)
    train, test = data.iloc[:cutoff], data.iloc[cutoff:]

    X_train, y_train = train[FEATURE_COLUMNS], train["label_next_pct"]
    X_test, y_test = test[FEATURE_COLUMNS], test["label_next_pct"]

    model = GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    model.fit(X_train, y_train)

    ml_pred = model.predict(X_test)
    ml_mae = mean_absolute_error(y_test, ml_pred)
    ml_hit_rate = float((np.sign(ml_pred) == np.sign(y_test.values)).mean() * 100)

    heuristic_pred = X_test["heuristic_pred_pct"].values
    heuristic_mae = mean_absolute_error(y_test, heuristic_pred)
    heuristic_hit_rate = float((np.sign(heuristic_pred) == np.sign(y_test.values)).mean() * 100)

    residual_std = float(np.std(y_test.values - ml_pred))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "tickers_used": tickers,
        "n_train_samples": int(len(train)),
        "n_test_samples": int(len(test)),
        "period": args.period,
        "synthetic": args.synthetic,
        "residual_std": residual_std,
        "feature_importances": dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4).tolist())),
        "holdout_metrics": {
            "ml_mae": round(float(ml_mae), 4),
            "ml_hit_rate": round(ml_hit_rate, 1),
            "heuristic_mae": round(float(heuristic_mae), 4),
            "heuristic_hit_rate": round(heuristic_hit_rate, 1),
        },
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Holdout MAE      — ML: {ml_mae:.3f}   heuristic: {heuristic_mae:.3f}")
    print(f"Holdout hit rate — ML: {ml_hit_rate:.1f}%   heuristic: {heuristic_hit_rate:.1f}%")
    if ml_mae >= heuristic_mae and ml_hit_rate <= heuristic_hit_rate:
        print("Note: on this holdout the ML model didn't beat the plain heuristic. "
              "More/longer history usually helps — try a bigger --limit or --period.")


if __name__ == "__main__":
    main()
