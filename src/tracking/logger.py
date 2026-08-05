"""Predicted-vs-actual tracking log.

This is the core of the self-correcting loop: every morning we log a
prediction per ticker (predicted % move, direction, confidence). Every
evening after close we fill in what actually happened. Everything the
Accuracy page and the home page's "hoard" meter show is read from this one
CSV.
"""

from __future__ import annotations

from datetime import date as date_cls

import pandas as pd

from src.config import PROCESSED_DATA_DIR

LOG_PATH = PROCESSED_DATA_DIR / "predictions_log.csv"

COLUMNS = [
    "date", "ticker", "prev_close", "predicted_pct", "predicted_direction",
    "confidence", "source", "actual_close", "actual_pct", "actual_direction", "hit",
]


def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    log = pd.read_csv(LOG_PATH, parse_dates=["date"])
    if "source" not in log.columns:
        log["source"] = "heuristic"  # backfill for logs written before the ML model existed
    return log


def _save(df: pd.DataFrame) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_PATH, index=False)


def log_predictions(rows: list[dict], for_date: date_cls | None = None) -> int:
    """Append a batch of pre-market predictions.

    Each row: {ticker, prev_close, predicted_pct, predicted_direction, confidence}
    Skips tickers that already have a row for the given date. Returns the
    number of rows actually written (excluding skipped duplicates).
    """
    target_date = for_date or date_cls.today()
    log = load_log()

    existing_today = set(
        log.loc[log["date"].dt.date == target_date, "ticker"]
    ) if not log.empty else set()

    new_rows = []
    for row in rows:
        if row["ticker"] in existing_today:
            continue
        new_rows.append({
            "date": target_date,
            "ticker": row["ticker"],
            "prev_close": row["prev_close"],
            "predicted_pct": row["predicted_pct"],
            "predicted_direction": row["predicted_direction"],
            "confidence": row["confidence"],
            "source": row.get("source", "heuristic"),
            "actual_close": None,
            "actual_pct": None,
            "actual_direction": None,
            "hit": None,
        })

    if not new_rows:
        return 0

    new_df = pd.DataFrame(new_rows, columns=COLUMNS)
    combined = new_df if log.empty else pd.concat([log, new_df], ignore_index=True)
    _save(combined)
    return len(new_rows)


def record_actuals(actual_closes: dict[str, float], for_date: date_cls | None = None) -> int:
    """Fill in actual outcomes for a date's logged predictions.

    actual_closes: {ticker: closing_price}
    Returns the number of rows updated.
    """
    target_date = for_date or date_cls.today()
    log = load_log()
    if log.empty:
        return 0

    # These columns start out empty (NaN, inferred as float64) until the
    # first real value lands in them — cast to object first so assigning a
    # string/bool doesn't trip a pandas dtype-compatibility warning.
    for col in ("actual_direction", "hit"):
        log[col] = log[col].astype("object")

    mask = (log["date"].dt.date == target_date) & (log["actual_close"].isna())
    updated = 0

    for idx in log[mask].index:
        ticker = log.at[idx, "ticker"]
        if ticker not in actual_closes:
            continue
        prev_close = log.at[idx, "prev_close"]
        actual_close = actual_closes[ticker]
        actual_pct = round(((actual_close - prev_close) / prev_close) * 100, 2) if prev_close else None
        actual_direction = "up" if actual_pct and actual_pct > 0.05 else "down" if actual_pct and actual_pct < -0.05 else "flat"
        predicted_direction = log.at[idx, "predicted_direction"]

        log.at[idx, "actual_close"] = actual_close
        log.at[idx, "actual_pct"] = actual_pct
        log.at[idx, "actual_direction"] = actual_direction
        log.at[idx, "hit"] = bool(predicted_direction == actual_direction)
        updated += 1

    if updated:
        _save(log)
    return updated


def accuracy_stats(df: pd.DataFrame | None = None) -> dict:
    """Summary stats over resolved (actual filled in) predictions."""
    log = df if df is not None else load_log()
    resolved = log.dropna(subset=["hit"])
    if resolved.empty:
        return {"total": 0, "correct": 0, "hit_rate": None, "mean_abs_error": None}

    hit_rate = resolved["hit"].mean() * 100
    mean_abs_error = (resolved["predicted_pct"] - resolved["actual_pct"]).abs().mean()

    return {
        "total": int(len(resolved)),
        "correct": int(resolved["hit"].sum()),
        "hit_rate": round(float(hit_rate), 1),
        "mean_abs_error": round(float(mean_abs_error), 2),
    }
