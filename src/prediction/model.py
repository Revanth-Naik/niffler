"""Baseline prediction model for Niffler.

This is intentionally a simple, fully transparent heuristic — not a trained
ML model — so it's easy to audit, explain in the UI, and later replace with
something learned. It combines two classic technical signals:

1. Momentum: short moving average vs. long moving average. Short MA above
   long MA is read as bullish drift; below is bearish drift.
2. Mean reversion: RSI(14). Overbought (>70) pulls the prediction down;
   oversold (<30) pulls it up.

The blend is a starting point for the "predict pre-market, correct
post-market" loop the project is built around. It is not investment advice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Add SMA5, SMA20, RSI14, and daily return columns to an OHLCV frame."""
    out = df.copy()
    out["sma_5"] = out[price_col].rolling(5).mean()
    out["sma_20"] = out[price_col].rolling(20).mean()
    out["daily_return"] = out[price_col].pct_change()

    delta = out[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    out["rsi_14"] = out["rsi_14"].fillna(50)

    return out


def score_from_indicators(sma_5: float, sma_20: float, rsi: float) -> tuple[float, int]:
    """The heuristic's core scoring math, isolated so it can be reused both
    for single-point prediction (predict_next_session) and vectorized
    feature engineering (see src/prediction/features.py) without
    recomputing rolling windows for every row."""
    momentum_pct = ((sma_5 - sma_20) / sma_20) * 100
    momentum_component = float(np.clip(momentum_pct * 0.6, -2.5, 2.5))

    if rsi > 70:
        reversion_component = -0.4 * ((rsi - 70) / 30)
    elif rsi < 30:
        reversion_component = 0.4 * ((30 - rsi) / 30)
    else:
        reversion_component = 0.0

    predicted_pct = round(momentum_component + reversion_component, 2)

    momentum_signal = np.sign(momentum_component)
    reversion_signal = np.sign(reversion_component) if reversion_component != 0 else momentum_signal
    agreement_bonus = 15 if momentum_signal == reversion_signal else -10

    strength = min(abs(momentum_pct) * 20, 45)
    confidence = int(np.clip(45 + strength + agreement_bonus, 5, 95))

    return predicted_pct, confidence


def predict_next_session(df: pd.DataFrame, price_col: str = "Close") -> dict:
    """Predict the next session's % move for a single ticker.

    df must have at least 20 rows of OHLCV history, most recent last.
    Returns a dict with predicted_pct, direction, confidence (0-100), and a
    short plain-language rationale.
    """
    if df is None or len(df) < 21:
        return {
            "predicted_pct": 0.0,
            "direction": "flat",
            "confidence": 0,
            "rationale": "Not enough history to form a view yet.",
        }

    ind = add_indicators(df, price_col=price_col)
    latest = ind.iloc[-1]

    sma_5 = latest["sma_5"]
    sma_20 = latest["sma_20"]
    rsi = latest["rsi_14"]

    if pd.isna(sma_5) or pd.isna(sma_20):
        return {
            "predicted_pct": 0.0,
            "direction": "flat",
            "confidence": 0,
            "rationale": "Not enough history to form a view yet.",
        }

    predicted_pct, confidence = score_from_indicators(sma_5, sma_20, rsi)
    direction = "up" if predicted_pct > 0.05 else "down" if predicted_pct < -0.05 else "flat"

    if direction == "up":
        rationale = f"Short-term average is running above the 20-session trend (RSI {rsi:.0f}) — reads bullish."
    elif direction == "down":
        rationale = f"Short-term average has slipped below the 20-session trend (RSI {rsi:.0f}) — reads bearish."
    else:
        rationale = f"Short and long trend are close together (RSI {rsi:.0f}) — no strong signal either way."

    return {
        "predicted_pct": predicted_pct,
        "direction": direction,
        "confidence": confidence,
        "rationale": rationale,
    }
