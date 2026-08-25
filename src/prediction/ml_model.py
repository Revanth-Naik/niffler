"""Inference for the trained ML correction model.

Framework-agnostic — no Streamlit imports here, so it can be used from
scripts too. Callers should cache load_model()/load_meta() themselves if
calling repeatedly (the Streamlit layer does this with st.cache_resource).
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.prediction.features import latest_feature_row
from src.prediction.train import MODEL_PATH, META_PATH


def is_available() -> bool:
    return MODEL_PATH.exists() and META_PATH.exists()


def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_meta() -> dict | None:
    if not META_PATH.exists():
        return None
    with open(META_PATH) as f:
        return json.load(f)


def predict_next_session_ml(df: pd.DataFrame, model=None, meta: dict | None = None) -> dict | None:
    """Returns None if the model isn't trained yet or there isn't enough
    history to build a feature row — callers should fall back to the
    heuristic in that case."""
    if model is None:
        model = load_model()
    if model is None:
        return None
    if meta is None:
        meta = load_meta() or {}

    row = latest_feature_row(df)
    if row is None:
        return None

    predicted_pct = round(float(model.predict(row)[0]), 2)
    residual_std = meta.get("residual_std", 1.0) or 1.0

    z = abs(predicted_pct) / max(residual_std, 0.01)
    confidence = int(np.clip(40 + z * 20, 10, 95))

    # Sign-based up/down only, no "flat" bucket — matches prediction/model.py
    # and tracking/logger.py. This ML path is where nearly all live
    # predictions actually come from once a trained model exists, so this
    # copy of the old +/-0.05% flat threshold was the real reason the fix
    # committed earlier didn't show up in production: that fix only patched
    # the heuristic's own predict_next_session(), which the app stops using
    # as soon as a trained model is available. See the longer note in
    # prediction/model.py and tracking/logger.py for the full reasoning.
    direction = "up" if predicted_pct >= 0 else "down"

    holdout = meta.get("holdout_metrics", {})
    ml_hit_rate = holdout.get("ml_hit_rate")
    trained_note = f"trained on {meta.get('n_train_samples', '?')} samples" if meta else "trained"
    hit_rate_note = f", {ml_hit_rate}% holdout hit rate" if ml_hit_rate is not None else ""

    rationale = f"ML correction model ({trained_note}{hit_rate_note}) reads {direction}."

    return {
        "predicted_pct": predicted_pct,
        "direction": direction,
        "confidence": confidence,
        "rationale": rationale,
        "source": "ml",
    }
