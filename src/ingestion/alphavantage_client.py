"""Optional second data source: Alpha Vantage (news/sentiment, fundamentals).

Requires ALPHAVANTAGE_API_KEY in .env. Free tier is rate-limited to
25 requests/day, so this is meant to supplement yfinance, not replace it.
"""

from __future__ import annotations

import requests

from src.config import ALPHAVANTAGE_API_KEY

BASE_URL = "https://www.alphavantage.co/query"


def is_enabled() -> bool:
    return bool(ALPHAVANTAGE_API_KEY)


def fetch_news_sentiment(ticker: str, limit: int = 10) -> dict:
    """Fetch recent news + sentiment scores for a ticker."""
    if not is_enabled():
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set — add it to .env to use this source.")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": limit,
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    response = requests.get(BASE_URL, params=params, timeout=15)
    response.raise_for_status()
    return response.json()
