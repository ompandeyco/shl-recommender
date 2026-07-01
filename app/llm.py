"""
llm.py
------
Thin I/O adapter around whichever LLM provider is configured via environment
variables.  ALL direct API calls live here so that swapping providers means
changing only this file — agent.py never imports groq/openai/google directly.

Provider detection (checked in order)
--------------------------------------
  GROQ_API_KEY     → Groq   (llama-3.3-70b-versatile by default)
  OPENAI_API_KEY   → OpenAI (gpt-4o-mini by default)
  GOOGLE_API_KEY   → Google Gemini (gemini-2.0-flash by default)

If no key is set the module raises a clear ConfigError at import time.

Environment variables
---------------------
  GROQ_API_KEY        Groq Cloud secret key  (starts with gsk_)
  OPENAI_API_KEY      OpenAI secret key      (starts with sk-)
  GOOGLE_API_KEY      Google AI Studio key
  LLM_MODEL           Override the default model name for the active provider
  LLM_TEMPERATURE     Float, default 0.2
  LLM_MAX_TOKENS      Int,   default 1024
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider detection — resolved once at module import
# ---------------------------------------------------------------------------

class ConfigError(RuntimeError):
    """Raised when no LLM API key is present in the environment."""


def _detect_provider() -> str:
    """Return 'groq', 'openai', or 'google', or raise ConfigError."""
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GOOGLE_API_KEY"):
        return "google"
    raise ConfigError(
        "No LLM API key found. Set GROQ_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY "
        "in your .env file. The agent cannot run without an LLM provider."
    )


_PROVIDER: str = _detect_provider()

# Default model names per provider (overridable via LLM_MODEL)
_DEFAULT_MODELS: dict[str, str] = {
    "groq":   "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "google": "gemini-2.0-flash",
}

_DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
_DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))


def _resolved_model(override: str | None) -> str:
    """Return the model name to use, respecting env + caller override."""
    return override or os.getenv("LLM_MODEL") or _DEFAULT_MODELS[_PROVIDER]


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _with_retry(fn, *, retries: int = 3, backoff: float = 1.5) -> Any:
    """
    Call fn() up to `retries` times with exponential backoff.

    Retries on transient errors (rate limits, server errors).  Re-raises
    the last exception if all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = backoff ** attempt
            log.warning(
                "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, retries, exc, wait,
            )
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

def _chat_openai(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    import openai  # type: ignore[import-untyped]

    client = openai.OpenAI()  # reads OPENAI_API_KEY from env

    def _call() -> str:
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    return _with_retry(_call)


def _embed_openai(text: str) -> list[float]:
    import openai

    client = openai.OpenAI()

    def _call() -> list[float]:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[text],
        )
        return response.data[0].embedding

    return _with_retry(_call)


# ---------------------------------------------------------------------------
# Groq backend
# ---------------------------------------------------------------------------

def _chat_groq(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    from groq import Groq  # type: ignore[import-untyped]

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def _call() -> str:
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    return _with_retry(_call)


def _embed_groq(text: str) -> list[float]:
    # Groq does not provide a native embedding endpoint as of 2025.
    # Fall back to a simple zero vector so retrieval degrades to BM25-only
    # rather than crashing.  A WARNING is emitted so operators are aware.
    log.warning(
        "Groq does not support embeddings — retrieval will use BM25-only. "
        "Add a GOOGLE_API_KEY or OPENAI_API_KEY for hybrid retrieval."
    )
    return []   # empty list → retrieval.py treats as 'no embedding available'


# ---------------------------------------------------------------------------
# Google Gemini backend  (uses google-genai, the current non-deprecated SDK)
# ---------------------------------------------------------------------------

def _chat_google(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    from google import genai                        # type: ignore[import-untyped]
    from google.genai import types as genai_types  # type: ignore[import-untyped]

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    # google.genai uses a flat content list; split system prompt out manually.
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    turns = [m for m in messages if m["role"] != "system"]

    # Build Contents list (alternating user / model).
    contents = []
    for m in turns:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append(genai_types.Content(
            role=role,
            parts=[genai_types.Part(text=m["content"])],
        ))

    system_instruction = system_parts[0] if system_parts else None

    def _call() -> str:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""

    return _with_retry(_call)


def _embed_google(text: str) -> list[float]:
    from google import genai  # type: ignore[import-untyped]

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def _call() -> list[float]:
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=text,
        )
        # The new SDK returns an EmbedContentResponse with .embeddings list
        return result.embeddings[0].values

    return _with_retry(_call)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chat_completion(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> str:
    """
    Call the LLM chat endpoint and return the assistant reply as a plain string.

    Parameters
    ----------
    messages:
        List of ``{"role": ..., "content": ...}`` dicts.  Roles accepted:
        ``system``, ``user``, ``assistant``.
    model:
        Override the default model for this call only.
    temperature:
        Sampling temperature.  Lower = more deterministic.  Default 0.2.
    max_tokens:
        Maximum completion tokens.

    Returns
    -------
    str
        Raw text of the assistant's reply.
    """
    resolved = _resolved_model(model)
    log.debug("chat_completion provider=%s model=%s", _PROVIDER, resolved)

    if _PROVIDER == "groq":
        return _chat_groq(messages, resolved, temperature, max_tokens)
    if _PROVIDER == "openai":
        return _chat_openai(messages, resolved, temperature, max_tokens)
    if _PROVIDER == "google":
        return _chat_google(messages, resolved, temperature, max_tokens)
    raise ConfigError(f"Unknown provider: {_PROVIDER!r}")


def embed(text: str) -> list[float]:
    """
    Return an embedding vector for ``text`` using the configured provider.

    Parameters
    ----------
    text:
        The string to embed.

    Returns
    -------
    list[float]
        Dense float vector.
    """
    if _PROVIDER == "groq":
        return _embed_groq(text)
    if _PROVIDER == "openai":
        return _embed_openai(text)
    if _PROVIDER == "google":
        return _embed_google(text)
    raise ConfigError(f"Unknown provider: {_PROVIDER!r}")


def parse_json_response(raw: str) -> dict:
    """
    Extract and parse the first JSON object from an LLM reply.

    LLMs sometimes wrap JSON in markdown fences (```json ... ```).
    This helper strips the fences before parsing so callers never have to.

    Parameters
    ----------
    raw:
        The raw string returned by chat_completion().

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    ValueError
        If no valid JSON object can be extracted.
    """
    # Strip markdown fences if present.
    text = raw.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = text.split("\n", 1)[-1]
        # Remove closing fence
        if text.endswith("```"):
            text = text[: text.rfind("```")]

    text = text.strip()

    # Find the outermost { ... } block in case the model added extra prose.
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM response:\n{raw[:300]}")

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse JSON from LLM response: {exc}\nRaw: {raw[:300]}"
        ) from exc
