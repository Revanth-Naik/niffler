"""Optional real-LLM backend for Dumbledore.

Both of these are entirely optional — dumbledore.py falls back to its
template system if neither is configured, so the app works with zero
setup either way. Two backends, tried in order:

1. Groq (cloud, free tier: https://console.groq.com) — a plain HTTPS API
   call, so it works from anywhere with internet, including the deployed
   Streamlit Cloud app. Enabled by setting GROQ_API_KEY (as a root-level
   entry in Streamlit Cloud's "Secrets" settings, or in a local .env —
   either way it reaches this module as a normal environment variable).

2. Ollama (local only — https://ollama.com) — needs the Ollama app/server
   actually running on the same machine as the Streamlit process. This
   works when you run Niffler on your own Mac with Ollama installed, but
   NOT on Streamlit Community Cloud, which has no way to run a background
   model server alongside your app. Enabled automatically if a local
   Ollama server is reachable at OLLAMA_HOST.

Groq is tried first because it works in both places; Ollama is the
fallback for fully local/offline use when you'd rather not use a cloud
API at all.
"""

from __future__ import annotations

import os

import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

_REQUEST_TIMEOUT = 15


def _call_groq(system_prompt: str, user_message: str) -> str | None:
    if not GROQ_API_KEY:
        return None
    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.4,
                "max_tokens": 350,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def _ollama_reachable() -> bool:
    try:
        requests.get(OLLAMA_HOST, timeout=1)
        return True
    except Exception:
        return False


def _call_ollama(system_prompt: str, user_message: str) -> str | None:
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception:
        return None


def active_backend() -> str | None:
    """Which backend would actually be used right now, or None if neither
    is configured/reachable. Cheap enough to call per-message (Groq check
    is free; the Ollama check is a 1s-timeout local request)."""
    if GROQ_API_KEY:
        return "groq"
    if _ollama_reachable():
        return "ollama"
    return None


def generate(system_prompt: str, user_message: str) -> str | None:
    """Returns None if neither backend produced a reply — caller should
    fall back to the template system in dumbledore.py."""
    backend = active_backend()
    if backend == "groq":
        reply = _call_groq(system_prompt, user_message)
        if reply:
            return reply
        # Groq configured but the call itself failed (bad key, rate
        # limit, network) — try Ollama as a last resort before giving up.
        return _call_ollama(system_prompt, user_message)
    if backend == "ollama":
        return _call_ollama(system_prompt, user_message)
    return None
