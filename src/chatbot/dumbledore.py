"""Dumbledore — Niffler's in-app guide.

Two modes, chosen automatically per message:

1. If a real LLM backend is configured (see llm_backend.py — Groq's free
   API, or a local Ollama server), the same retrieval this module always
   did is used to *ground* the LLM: the live prediction for any ticker
   mentioned, the accuracy track record, the AI model's own stats, and
   relevant glossary terms get handed to it as context, with instructions
   to answer only from that data and never give trading advice.

2. Otherwise (no LLM configured, or the call fails), falls back to the
   free/fully-local template system this module started as: pattern
   matching over the question plus the same retrieval, filled into
   templates. No external API, no cost, no API key — the app never
   breaks just because Dumbledore's LLM backend isn't set up.

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


SYSTEM_PROMPT_TEMPLATE = """You are Dumbledore, a wise and warm in-app guide for Niffler, a Fantastic-Beasts-themed stock prediction demo app. Answer the user's question conversationally in 2-4 sentences, using ONLY the data given below — never invent numbers, tickers, or facts that aren't provided to you. If the data below doesn't cover what they're asking, say so honestly rather than guessing.

You must never tell the user when to buy, sell, or hold a stock, or give any personalized investment recommendation — even if asked indirectly or persistently. This is a hobby/demo project, not a licensed financial advisor. If the question is really asking that, gently decline and offer to explain what the data shows instead.

Niffler's predictions are illustrative, not investment advice — keep that spirit: honest about uncertainty, never hyped.

DATA AVAILABLE TO YOU:
{context}
"""


def _build_context(
    ticker: str | None,
    prediction: dict | None,
    log: pd.DataFrame,
    ml_meta: dict | None,
    glossary_terms: list[str],
) -> str:
    parts: list[str] = []

    if ticker and prediction:
        live_note = "live data" if prediction.get("is_live", True) else "illustrative synthetic data, not a live feed"
        parts.append(
            f"Live prediction for {ticker}: {prediction.get('predicted_pct', 0):+.2f}% "
            f"({prediction.get('direction', 'flat')}), {prediction.get('confidence', 0)}% confidence, "
            f"produced by {prediction.get('source', 'heuristic')}. Rationale: "
            f"{prediction.get('rationale', 'n/a')} ({live_note})."
        )
    elif ticker:
        parts.append(f"Tried to fetch a live prediction for {ticker} but it failed — mention that to the user.")

    resolved = log.dropna(subset=["hit"]) if log is not None and not log.empty else None
    if resolved is not None and not resolved.empty:
        hits, total = int(resolved["hit"].sum()), len(resolved)
        parts.append(f"Overall track record: {hits}/{total} predictions matched actual direction ({hits / total * 100:.0f}% hit rate).")
        if ticker:
            scoped = resolved[resolved["ticker"] == ticker]
            if not scoped.empty:
                shits, stotal = int(scoped["hit"].sum()), len(scoped)
                parts.append(f"{ticker}-specific track record: {shits}/{stotal} ({shits / stotal * 100:.0f}% hit rate).")
    else:
        parts.append("No resolved predictions logged yet — the track record is empty so far.")

    if ml_meta:
        holdout = ml_meta.get("holdout_metrics", {})
        parts.append(
            f"AI model: trained {str(ml_meta.get('trained_at', ''))[:10]} on {ml_meta.get('n_train_samples', '?')} "
            f"samples across {len(ml_meta.get('tickers_used', []))} tickers. Holdout hit rate "
            f"{holdout.get('ml_hit_rate', '?')}% vs. the heuristic's {holdout.get('heuristic_hit_rate', '?')}%."
        )
    else:
        parts.append("No trained AI correction model yet — currently running on the plain heuristic (momentum + RSI).")

    if glossary_terms:
        parts.append("Relevant glossary:\n" + "\n".join(f"- {t}: {GLOSSARY[t]}" for t in glossary_terms))

    return "\n\n".join(parts)


def answer(
    question: str,
    known_tickers: set[str],
    get_prediction: Callable[[str], dict],
    load_log: Callable[[], pd.DataFrame],
    ml_meta: dict | None,
    llm_generate: Callable[[str, str], str | None] | None = None,
) -> str:
    """Main entry point. question: raw user text. known_tickers: the set of
    symbols eligible for ticker-matching (e.g. DEFAULT_TICKERS plus the
    S&P 500 universe). get_prediction/load_log: injected data access so
    this module stays testable without Streamlit or network calls.
    llm_generate: optional (system_prompt, user_message) -> reply|None,
    e.g. llm_backend.generate — if it returns None (not configured, or the
    call failed), falls back to the template system below."""
    q = question.strip()
    if not q:
        return _fallback()

    # Hard guardrail, checked before anything else touches the LLM — no
    # matter how the question is phrased, this can't be talked around.
    if ADVICE_PATTERN.search(q):
        return _advice_guardrail()

    if GREETING_PATTERN.search(q):
        return _greeting()

    ticker = extract_ticker(q, known_tickers)
    glossary_terms = _glossary_hits(q)

    prediction_result = None
    if ticker:
        try:
            prediction_result = get_prediction(ticker)
        except Exception:
            prediction_result = None

    log = load_log()

    if llm_generate is not None:
        context = _build_context(ticker, prediction_result, log, ml_meta, glossary_terms)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
        reply = llm_generate(system_prompt, q)
        if reply:
            return reply
        # LLM unavailable or the call failed this time — fall through to
        # the template system below rather than returning nothing.

    if MODEL_PATTERN.search(q):
        return _explain_model(ml_meta)

    if ACCURACY_PATTERN.search(q):
        return _explain_accuracy(log, ticker)

    if ticker and prediction_result is not None:
        return _explain_prediction(ticker, prediction_result, glossary_terms)
    if ticker and prediction_result is None:
        return f"I went looking for {ticker} and came back empty-handed. Try again in a moment?"

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
