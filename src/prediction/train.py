"""Core training logic for the AI correction model — shared by
scripts/train_model.py (CLI) and the web app's first-run bootstrap (see
web/data_helpers.py).

See scripts/train_model.py's module docstring for the full explanation of
what this model is and why it's trained the way it is. This module holds
the mechanics; the CLI script is a thin wrapper around train_and_save().
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.config import MODEL_DIR
from src.prediction.features import FEATURE_COLUMNS, build_training_examples
from src.prediction.synthetic import synthetic_history

MODEL_PATH = MODEL_DIR / "niffler_gbm.joblib"
META_PATH = MODEL_DIR / "niffler_gbm_meta.json"

PERIOD_DAYS = {"3mo": 90, "6mo": 180, "1y": 252, "2y": 504}


class NotEnoughDataError(ValueError):
    """Raised when too few training rows were gathered to fit a model."""


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
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def train_and_save(tickers: list[str], period: str = "1y", synthetic: bool = False) -> dict:
    """Trains the model, saves it + its metadata to disk, and returns the
    metadata dict. Raises NotEnoughDataError if too few rows were gathered."""
    data = gather_training_data(tickers, period, synthetic)

    if len(data) < 200:
        raise NotEnoughDataError(
            f"Only gathered {len(data)} training rows — need at least 200. "
            "Try a longer period, more tickers, or synthetic=True."
        )

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
        "period": period,
        "synthetic": synthetic,
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

    return meta
