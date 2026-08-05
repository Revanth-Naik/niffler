"""Free, no-key data source: pulls OHLCV + basic info via yfinance."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


def fetch_daily_ohlcv(ticker: str, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
    """Fetch recent OHLCV data for a single ticker.

    period/interval follow yfinance conventions, e.g. period="5d", interval="1d".
    """
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if data.empty:
        return pd.DataFrame()

    data = data.reset_index()
    data.insert(0, "ticker", ticker)
    data["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
    return data


def fetch_snapshot(ticker: str) -> dict:
    """Fetch a lightweight current-price snapshot for pre/post-market comparison."""
    info = yf.Ticker(ticker).fast_info
    return {
        "ticker": ticker,
        "last_price": getattr(info, "last_price", None),
        "previous_close": getattr(info, "previous_close", None),
        "day_high": getattr(info, "day_high", None),
        "day_low": getattr(info, "day_low", None),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def fetch_many(tickers: list[str], period: str = "5d", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data for multiple tickers and concatenate into one DataFrame."""
    frames = [fetch_daily_ohlcv(t, period=period, interval=interval) for t in tickers]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
