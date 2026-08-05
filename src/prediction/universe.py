"""Ticker universe for the top-50 scanner.

Tries to pull the live S&P 500 constituent list from Wikipedia (free, no
key). If that fails — no network, Wikipedia layout change, etc. — falls
back to a bundled ~100-ticker offline list of well-known large caps so the
app still works. A successful live fetch is cached to disk so subsequent
runs don't need to hit the network every time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ROOT_DIR

CACHE_DIR = ROOT_DIR / "data" / "cache"
LIVE_CACHE_PATH = CACHE_DIR / "sp500_cache.csv"
FALLBACK_PATH = CACHE_DIR / "sp500_fallback.csv"

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _fetch_from_wikipedia() -> list[str]:
    tables = pd.read_html(WIKIPEDIA_URL)
    sp500_table = tables[0]
    tickers = sp500_table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    return sorted(set(tickers))


def get_sp500_tickers(force_refresh: bool = False) -> list[str]:
    """Return the S&P 500 ticker list, preferring a live fetch, falling
    back to cache, then to the bundled offline list."""
    if not force_refresh and LIVE_CACHE_PATH.exists():
        return pd.read_csv(LIVE_CACHE_PATH)["ticker"].tolist()

    try:
        tickers = _fetch_from_wikipedia()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ticker": tickers}).to_csv(LIVE_CACHE_PATH, index=False)
        return tickers
    except Exception:
        return pd.read_csv(FALLBACK_PATH)["ticker"].tolist()
