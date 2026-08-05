"""Deterministic synthetic OHLCV data — used as an offline fallback when
live data isn't available, and to seed demo history for the Accuracy page.
Not real market data."""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd


def synthetic_history(ticker: str, period_days: int = 90) -> pd.DataFrame:
    # Python's built-in hash() is randomized per-process by default (a
    # security feature, PYTHONHASHSEED) — it is NOT stable across restarts,
    # which would silently break the "deterministic" guarantee this module
    # promises. zlib.crc32 is a real, stable hash for this purpose.
    seed = zlib.crc32(ticker.encode()) % (2**32)
    rng = np.random.default_rng(seed)
    base_price = 50 + (seed % 400)

    returns = rng.normal(loc=0.0004, scale=0.015, size=period_days)
    closes = base_price * np.cumprod(1 + returns)

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=period_days)
    df = pd.DataFrame({
        "Close": closes,
        "Open": closes * (1 - rng.normal(0, 0.003, period_days)),
        "High": closes * (1 + abs(rng.normal(0, 0.006, period_days))),
        "Low": closes * (1 - abs(rng.normal(0, 0.006, period_days))),
        "Volume": rng.integers(1_000_000, 20_000_000, period_days),
    }, index=dates)
    return df
