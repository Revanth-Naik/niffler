"""Feature engineering for the ML model.

Two feature groups:
1. Standard technical features (momentum, volatility, RSI, MACD, volume trend).
2. The heuristic model's own output (predicted_pct, confidence) as an input
   feature. This is what lets the ML model learn to *correct* the
   heuristic — e.g. "when the heuristic says +2% but RSI is this extreme,
   it tends to overshoot" — rather than starting from scratch. It's the
   technical implementation of the project's original "predict, then
   correct" idea, just applied at training time instead of only after the
   close.

FEATURE_COLUMNS is the exact column order the model is trained and
queried with — keep training and inference in sync by always going through
build_feature_row / build_feature_frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.prediction.model import add_indicators, score_from_indicators

FEATURE_COLUMNS = [
    "momentum_5_20",
    "momentum_10_50",
    "rsi_14",
    "macd",
    "macd_signal",
    "volatility_10",
    "volume_trend",
    "return_1d",
    "return_5d",
    "heuristic_pred_pct",
    "heuristic_confidence",
]


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def build_feature_frame(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Compute the full feature set for every row of a single ticker's
    history, including what the heuristic model would have predicted at
    each point in time. Everything here uses trailing rolling windows, so
    row i only ever sees data up to and including row i — no lookahead.
    Rows without enough lookback (start of the series) are NaN and should
    be dropped by the caller."""
    ind = add_indicators(df, price_col=price_col)

    out = pd.DataFrame(index=df.index)
    out["momentum_5_20"] = (ind["sma_5"] - ind["sma_20"]) / ind["sma_20"] * 100
    out["momentum_10_50"] = (
        df[price_col].rolling(10).mean() - df[price_col].rolling(50).mean()
    ) / df[price_col].rolling(50).mean() * 100
    out["rsi_14"] = ind["rsi_14"]

    macd_line, signal_line = _macd(df[price_col])
    out["macd"] = macd_line
    out["macd_signal"] = signal_line

    out["volatility_10"] = ind["daily_return"].rolling(10).std() * 100
    out["volume_trend"] = (
        df["Volume"].rolling(5).mean() / df["Volume"].rolling(20).mean() - 1
    ) * 100 if "Volume" in df else 0.0
    out["return_1d"] = ind["daily_return"] * 100
    out["return_5d"] = df[price_col].pct_change(5) * 100

    # Heuristic's own output as a feature, vectorized via the same scoring
    # function predict_next_session uses on a single point — this is what
    # lets the ML model learn to correct the heuristic's biases.
    scored = ind.apply(
        lambda row: score_from_indicators(row["sma_5"], row["sma_20"], row["rsi_14"])
        if pd.notna(row["sma_5"]) and pd.notna(row["sma_20"]) else (np.nan, np.nan),
        axis=1,
        result_type="expand",
    )
    out["heuristic_pred_pct"] = scored[0]
    out["heuristic_confidence"] = scored[1]

    return out


def build_training_examples(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Build a full (features + label) frame for one ticker's history.

    label = next-session % return. Drops rows with incomplete lookback or
    no next-day outcome to learn from.
    """
    features = build_feature_frame(df, price_col=price_col)
    features["label_next_pct"] = df[price_col].pct_change().shift(-1) * 100
    features = features.dropna(subset=FEATURE_COLUMNS + ["label_next_pct"])
    return features


def latest_feature_row(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame | None:
    """Build the single most recent feature row for live inference."""
    features = build_feature_frame(df, price_col=price_col)
    row = features.iloc[[-1]][FEATURE_COLUMNS]
    if row.isna().any(axis=None):
        return None
    return row
