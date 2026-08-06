"""Dumbledore — Niffler's in-app guide.

Free and fully local: no external LLM API, no cost, no API key. This is
deliberately *not* a real language model — it's pattern matching over the
question plus retrieval from Niffler's own data (live predictions, the
accuracy log, the trained model's own stats), filled into templates. Good
enough to explain what the app already knows; not a substitute for a real
conversational AI if you want one later (swap this module for a Claude API
call and keep the same retrieval — the data-gathering here would still be
useful as the "grounding" step).

Deliberately does NOT give buy/sell trading advice. Two reasons: this app
is deployed at a public URL, so a chatbot dispensing "sell now for profit"
calls would be unlicensed personalized financial advice reaching strangers,
not just you — a real liability problem. And nothing in this project
(heuristic or ML model) is actually good enough to responsibly tell anyone
when to trade. See _advice_guardrail(). This keeps the same "transparent,
not overselling itself" character as the rest of the app (e.g. the Model
insights page, the README disclaimer).

Every public function here is a pure function of its inputs — no
Streamlit or network imports — so it can be tested without spinning up the
app or hitting yfinance.
"""

from __future__ import annotations

import re
from typing import Callable

import pandas as pd

GLOSSARY: dict[str, str] = {
    "rsi": (
        "RSI (Relative Strength Index) measures how sharply a price has moved recently, on a 0-100 "
        "scale. Above 70 usually reads overbought (due for a pullback); below 30 reads oversold (due "
        "for a bounce). Niffler uses RSI(14) — the last 14 sessions."
    ),
    "momentum": (
        "Momentum here compares the 5-day average price to the 20-day average. Short average above "
        "long average reads as bullish drift; below reads as bearish drift."
    ),
    "macd": (
        "MACD compares two moving averages (12-day and 26-day) to spot shifts in momentum. It's one "
        "of the inputs the AI model considers; the plain heuristic doesn't use it directly."
    ),
    "confidence": (
        "Confidence is Niffler's own estimate of how strongly its signals agree — not a probability "
        "of being correct. A 90%-confidence prediction can still miss; it just means the underlying "
        "signals lined up strongly."
    ),
    "volatility": (
        "Volatility measures how much a price has been swinging day to day recently. Higher "
        "volatility means bigger, less predictable moves in either direction."
    ),
    "heuristic": (
        "The heuristic is Niffler's transparent baseline model — pure momentum + RSI math, no "
        "machine learning involved. It's easy to audit and always available as a fallback."
    ),
    "holdout": (
        "A holdout set is data the AI model never saw during training, used to test it honestly. "
        "It's how the Model insights page checks whether the AI model is actually better than the "
        "heuristic, rather than just having memorized its training data."
    ),
}

ADVICE_PATTERN = re.compile(
    r"\b(should i (buy|sell|hold)|when (should|to) sell|sell now|buy now|good time to (buy|sell)|"
    r"will .*(go up|go down|moon|crash|tank)|worth (buying|selling))\b",
    re.IGNORECASE,
)
MODEL_PATTERN = re.compile(
    r"\b(how does .*(model|ai|niffler) work|what is the ai|correction model|how (do|does) (you|it) predict)\b",
    re.IGNORECASE,
)
ACCURACY_PATTERN = re.compile(
    r"\b(accuracy|hit rate|track record|how (good|well|accurate)|been doing|performance)\b",
    re.IGNORECASE,
)
GREETING_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|greetings|good (morning|evening|afternoon))\b[\s!.,]*$",
    re.IGNORECASE,
)


def extract_ticker(text: str, known_tickers: set[str]) -> str | None:
    """Only matches tokens that are already ALL CAPS in the original text.
    This is a deliberate precision tradeoff: several real tickers are also
    common English words (SO, ON, NOW, ALL, IT, KEY, LOW, CAT...) — without
    this restriction, an ordinary sentence would misfire constantly. The
    tradeoff means the user needs to type tickers in caps, which matches
    how the rest of the app already treats ticker input."""
    for token in re.findall(r"\b[A-Z]{1,5}\b", text):
        if token in known_tickers:
            return token
    return None


def _glossary_hits(text: str) -> list[str]:
    lowered = text.lower()
    return [term for term in GLOSSARY if term in lowered]


def answer(
    question: str,
    known_tickers: set[str],
    get_prediction: Callable[[str], dict],
    load_log: Callable[[], pd.DataFrame],
    ml_meta: dict | None,
) -> str:
    """Main entry point. question: raw user text. known_tickers: the set of
    symbols eligible for ticker-matching (e.g. DEFAULT_TICKERS plus the
    S&P 500 universe). get_prediction/load_log: injected data access so
    this module stays testable without Streamlit or network calls."""
    q = question.strip()
    if not q:
        return _fallback()

    if ADVICE_PATTERN.search(q):
        return _advice_guardrail()

    if GREETING_PATTERN.search(q):
        return _greeting()

    if MODEL_PATTERN.search(q):
        return _explain_model(ml_meta)

    ticker = extract_ticker(q, known_tickers)
    glossary_terms = _glossary_hits(q)

    if ACCURACY_PATTERN.search(q):
        return _explain_accuracy(load_log(), ticker)

    if ticker:
        try:
            result = get_prediction(ticker)
        except Exception as exc:
            return f"I went looking for {ticker} and came back empty-handed — ({exc}). Try again in a moment?"
        return _explain_prediction(ticker, result, glossary_terms)

    if glossary_terms:
        return _explain_glossary(glossary_terms)

    return _fallback()


def _advice_guardrail() -> str:
    return (
        "Ah — now that's the one question even the wisest of us should be careful answering. "
        "I can show you what Niffler's model sees for a ticker — its momentum, RSI, and confidence — "
        "but deciding when to buy or sell is a decision only you, or a licensed advisor, should make. "
        "Try asking me *\"why is AAPL predicted to move\"* instead, and I'll show you what's under the hood."
    )


def _greeting() -> str:
    return (
        "Hello — I'm Dumbledore, Niffler's resident guide. I can explain why a ticker's prediction "
        "looks the way it does, walk through the accuracy track record, or explain terms like RSI or "
        "momentum. What would you like to know? (Tip: write tickers in CAPS, e.g. AAPL.)"
    )


def _explain_model(ml_meta: dict | None) -> str:
    if not ml_meta:
        return (
            "Right now Niffler is running purely on its transparent heuristic — momentum plus RSI, "
            "no trained AI model yet. Run `scripts/train_model.py` (or wait for the weekly retrain "
            "workflow) to train the correction layer."
        )
    holdout = ml_meta.get("holdout_metrics", {})
    tickers_used = ml_meta.get("tickers_used", [])
    return (
        f"Niffler's AI model is a gradient-boosted regressor trained on {ml_meta.get('n_train_samples', '?')} "
        f"examples across {len(tickers_used)} tickers, most recently on {str(ml_meta.get('trained_at', '?'))[:10]}. "
        "It learns to correct the plain heuristic rather than starting from scratch — the heuristic's own "
        "output is one of its input features. On its holdout set (data it never trained on), it scored "
        f"{holdout.get('ml_hit_rate', '?')}% direction hit rate versus the heuristic's {holdout.get('heuristic_hit_rate', '?')}%. "
        "See the Model insights page for the full breakdown."
    )


def _explain_accuracy(log: pd.DataFrame, ticker: str | None) -> str:
    resolved = log.dropna(subset=["hit"]) if not log.empty else log
    if resolved.empty:
        return (
            "There's no resolved track record yet — a prediction needs both a morning log entry and "
            "an evening actual-close entry before I can score it. Give the daily loop a few trading "
            "days and ask again."
        )
    if ticker:
        scoped = resolved[resolved["ticker"] == ticker]
        if scoped.empty:
            return (
                f"I don't have any resolved predictions for {ticker} yet — it may not be on the "
                "tracked watchlist, or just hasn't resolved yet."
            )
        hits = int(scoped["hit"].sum())
        total = len(scoped)
        return f"For {ticker}: {hits} of {total} predictions matched the actual direction ({hits / total * 100:.0f}% hit rate)."

    hits = int(resolved["hit"].sum())
    total = len(resolved)
    return (
        f"Across everything Niffler's tracked so far: {hits} of {total} predictions matched the actual "
        f"direction ({hits / total * 100:.0f}% hit rate). See the Accuracy tracker page for the full "
        "picture — hit rate over time, error distribution, and per-ticker breakdown."
    )


def _explain_glossary(terms: list[str]) -> str:
    return "\n\n".join(GLOSSARY[t] for t in terms)


def _explain_prediction(ticker: str, result: dict, glossary_terms: list[str]) -> str:
    direction = result.get("direction", "flat")
    pct = result.get("predicted_pct", 0.0)
    confidence = result.get("confidence", 0)
    source = result.get("source", "heuristic")
    rationale = result.get("rationale", "")
    source_label = "the AI correction model" if source == "ml" else "the heuristic"

    parts = [
        f"{ticker} is currently predicted to move {pct:+.2f}% ({direction}) by {source_label}, at "
        f"{confidence}% confidence."
    ]
    if rationale:
        parts.append(rationale)
    if not result.get("is_live", True):
        parts.append(f"Note: this is illustrative synthetic data for {ticker}, not a live feed.")
    if glossary_terms:
        parts.append("\n".join(GLOSSARY[t] for t in glossary_terms))
    parts.append("Remember — this is what the model's signals show, not a recommendation to act on.")
    return "\n\n".join(parts)


def _fallback() -> str:
    return (
        "I'm not quite sure what you're asking. Try things like:\n\n"
        "- *\"Why is AAPL predicted to move?\"*\n"
        "- *\"What's the accuracy on TSLA?\"*\n"
        "- *\"What is RSI?\"*\n"
        "- *\"How does the AI model work?\"*"
    )
