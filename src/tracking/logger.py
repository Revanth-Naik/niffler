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
    # date_format="ISO8601" tolerates a mix of "YYYY-MM-DD" and
    # "YYYY-MM-DD HH:MM:SS" in the same column — both are valid ISO8601,
    # just different precision. Without this, a single mixed-precision CSV
    # (which _save() used to produce before this fix) makes pandas give up
    # and leave the whole column as strings, silently breaking every
    # .dt accessor call downstream with "Can only use .dt accessor with
    # datetimelike values".
    log = pd.read_csv(LOG_PATH, parse_dates=["date"], date_format="ISO8601")
    if "source" not in log.columns:
        log["source"] = "heuristic"  # backfill for logs written before the ML model existed
    if "hit" in log.columns:
        # A column mixing True/False with blank (unresolved) rows gets read
        # back as dtype "object", not bool — pandas has no way to represent
        # NA in a plain bool column. Object-dtype survives .mean() (returns
        # a real float) but NOT .round() (raises TypeError: Expected
        # numeric dtype, got object instead) on some pandas versions, which
        # is what broke the Accuracy tracker page's "hit rate over time"
        # chart. "boolean" is pandas' nullable bool dtype — it supports NA
        # natively and behaves numerically everywhere downstream.
        log["hit"] = log["hit"].astype("boolean")
    return log


def _save(df: pd.DataFrame) -> None:
    df = df.copy()
    if "date" in df.columns:
        # Always write a single consistent date-only format ("YYYY-MM-DD"),
        # regardless of whether "date" currently holds datetime64 values,
        # raw datetime.date objects, or (after a pd.concat of the two)
        # mixed types — this is what caused the mixed-precision bug above.
        df["date"] = pd.to_datetime(df["date"], format="ISO8601").dt.strftime("%Y-%m-%d")
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


def record_actuals(actual_closes: dict[tuple[str, date_cls], float]) -> int:
    """Fill in actual outcomes for logged predictions.

    actual_closes: {(ticker, prediction_date): closing_price}

    Deliberately resolves ANY pending row that has a matching entry —
    not just ones dated today. Earlier versions only ever looked at
    "today's" pending predictions, which meant a single failed/missed
    resolve run (e.g. a workflow bug, a transient outage) permanently
    orphaned that day's predictions, since the next run would only ever
    check the new "today" again. Keying by (ticker, date) instead lets a
    late or catch-up run backfill anything still pending, regardless of
    how long it's been waiting. Returns the number of rows updated.
    """
    log = load_log()
    if log.empty:
        return 0

    # These columns start out empty (NaN, inferred as float64) until the
    # first real value lands in them — cast to object first so assigning a
    # string/bool doesn't trip a pandas dtype-compatibility warning.
    for col in ("actual_direction", "hit"):
        log[col] = log[col].astype("object")

    updated = 0
    for idx in log[log["actual_close"].isna()].index:
        key = (log.at[idx, "ticker"], log.at[idx, "date"].date())
        if key not in actual_closes:
            continue
        prev_close = log.at[idx, "prev_close"]
        actual_close = actual_closes[key]
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
