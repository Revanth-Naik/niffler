"""Project-wide settings for Niffler."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
CACHE_DIR = ROOT_DIR / "data" / "cache"

# The home page's curated watchlist — kept small on purpose so the "Tonight's
# burrow" card row stays readable. Not the same list the daily automation
# predicts/tracks — see TRACKED_TICKERS below.
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


def _load_tracked_tickers() -> list[str]:
    """The universe the daily predict/resolve loop actually runs on. Reads
    the same bundled, curated large-cap list used as the offline S&P 500
    fallback (see prediction/universe.py) — not a live Wikipedia fetch,
    deliberately, since this runs unattended every weekday morning via
    GitHub Actions and shouldn't depend on a scrape succeeding to know what
    to predict. It's also the same list train_model.py trains on, so the
    live predictions and the model's own training universe stay aligned.
    Falls back to DEFAULT_TICKERS if the bundled file is ever missing."""
    path = CACHE_DIR / "sp500_fallback.csv"
    if not path.exists():
        return DEFAULT_TICKERS
    return pd.read_csv(path)["ticker"].tolist()


# Tracked by the daily automation (predict.yml / resolve.yml) and used for
# the accuracy dashboard — deliberately much larger than DEFAULT_TICKERS so
# the tracked track record covers a broad, useful slice of the market
# rather than just five names.
TRACKED_TICKERS = _load_tracked_tickers()

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
