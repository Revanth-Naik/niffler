"""Single entry point the rest of the app should call to get a prediction.

Uses the trained ML correction model when one is available and can produce
a result for the given history; otherwise falls back to the transparent
heuristic. Every result is tagged with "source" ("ml" or "heuristic") so
the UI can always show which one actually produced a given number —
no silent substitution.
"""

from __future__ import annotations

import pandas as pd

from src.prediction.ml_model import is_available, load_meta, load_model, predict_next_session_ml
from src.prediction.model import predict_next_session


def predict(df: pd.DataFrame, model=None, meta: dict | None = None) -> dict:
    if is_available():
        result = predict_next_session_ml(df, model=model, meta=meta)
        if result is not None:
            return result

    result = predict_next_session(df)
    result["source"] = "heuristic"
    return result
