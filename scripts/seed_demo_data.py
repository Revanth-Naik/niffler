#!/usr/bin/env python3
"""Backfill the predictions log with demo history so the Accuracy page and
home page hoard meter have something to show before you've run the real
daily loop for a while.

This runs the real prediction model against synthetic (not real market)
price series, walking forward day by day so the resulting log is
methodologically consistent — just not based on real prices. Safe to run
once when setting the project up; re-running clears and rebuilds it.

Usage:
    python scripts/seed_demo_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_TICKERS
from src.tracking.demo_seed import seed_demo_log
from src.tracking.logger import LOG_PATH


def main() -> None:
    df = seed_demo_log()
    print(f"Seeded {len(df)} demo predictions across {len(DEFAULT_TICKERS)} tickers to {LOG_PATH}")
    print(f"Demo hit rate: {df['hit'].mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
