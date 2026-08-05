"""Batch-scan a list of tickers and rank by predicted next-session return."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from src.prediction.model import predict_next_session


def scan_tickers(tickers: list[str], period: str = "3mo", batch_size: int = 50) -> pd.DataFrame:
    """Fetch recent history for each ticker and run the prediction model.

    Downloads in batches via yfinance's multi-ticker support to keep this
    reasonably fast for large universes (e.g. the full S&P 500).
    Returns a DataFrame sorted by predicted_pct descending.
    """
    rows = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            data = yf.download(
                batch, period=period, interval="1d", group_by="ticker",
                progress=False, threads=True, auto_adjust=True,
            )
        except Exception:
            continue

        for ticker in batch:
            try:
                if len(batch) == 1:
                    df = data
                else:
                    df = data[ticker]
                df = df.dropna(how="all")
                if df.empty or "Close" not in df:
                    continue
                result = predict_next_session(df)
                last_price = df["Close"].iloc[-1]
                rows.append({
                    "ticker": ticker,
                    "last_price": round(float(last_price), 2),
                    "predicted_pct": result["predicted_pct"],
                    "direction": result["direction"],
                    "confidence": result["confidence"],
                    "rationale": result["rationale"],
                })
            except Exception:
                continue

    if not rows:
        return pd.DataFrame(columns=["ticker", "last_price", "predicted_pct", "direction", "confidence", "rationale"])

    out = pd.DataFrame(rows)
    return out.sort_values("predicted_pct", ascending=False).reset_index(drop=True)


def top_n(tickers: list[str], n: int = 50, period: str = "3mo") -> pd.DataFrame:
    scanned = scan_tickers(tickers, period=period)
    return scanned.head(n).reset_index(drop=True)
