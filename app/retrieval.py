"""
retrieval.py
------------
Hybrid search layer: maps a free-text query to a ranked list of SHL
assessment catalog entries.

WHY HYBRID RETRIEVAL?
---------------------
Catalog entries are short (name + 1-3 sentence description) and often use
exact, domain-specific terminology ("Verify G+", "OPQ32r").  This creates
two failure modes if we pick only one retrieval method:

  * Pure BM25 / keyword search:
      Fails on paraphrases.  A user asking for a "programming assessment"
      gets no hits if the catalog says "coding test" — BM25 requires
      overlapping tokens.

  * Pure semantic / embedding search:
      Fails on exact names.  "OPQ32r" and "OPQ32i" may have very similar
      embeddings (both are personality tools) but mean different things;
      also, rare proper-noun product names may not survive sub-word
      tokenisation well.

Combining both with a weighted average gives us the best of both worlds:
BM25 anchors on exact tokens; embeddings generalise across paraphrases.
This is well-established in production search (e.g. Elasticsearch's
Reciprocal Rank Fusion, Bing's hybrid stack).

Architecture
------------
1. On import — build BM25 index over all catalog items (fast, <1 s for
   hundreds of items).
2. On import (if an embedding provider key exists) — batch-embed all
   catalog descriptions and cache the vectors in memory (called once, reused
   for every query).
3. At query time — score with BM25, optionally score with cosine similarity
   against query embedding, normalise both to [0, 1], weighted-average, sort,
   return top_k.

Providers supported (checked in order)
---------------------------------------
  OPENAI_API_KEY          → OpenAI text-embedding-3-small
  GOOGLE_API_KEY          → Google generativeai text-embedding-004
  (fallback)              → BM25 only, no key required
"""

from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from typing import Optional

from app.catalog import get_all

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1.  BM25 INDEX — built once at module import time
# ---------------------------------------------------------------------------

def _build_corpus(items: list[dict]) -> list[list[str]]:
    """
    Convert each catalog entry into a token list for BM25.

    We concatenate name + description + test_type so that both the product
    name ("Verify Numerical Reasoning") and the category ("Ability & Aptitude")
    participate in lexical matching.  Simple whitespace tokenisation is fine
    here — the catalog is English prose with no exotic characters.
    """
    corpus: list[list[str]] = []
    for item in items:
        text = " ".join([
            item.get("name", ""),
            item.get("description", ""),
            item.get("test_type", ""),
        ])
        tokens = text.lower().split()
        corpus.append(tokens)
    return corpus


def _init_bm25():
    """
    Lazily import rank_bm25 and build the BM25Okapi index.

    We defer the import so that the module can be loaded even if rank_bm25
    is not yet installed (e.g. during CI linting); a clear ImportError is
    raised the first time search() is actually called.
    """
    try:
        from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "rank_bm25 is required for retrieval. "
            "Install it with: pip install rank-bm25"
        ) from exc

    items = get_all()
    if not items:
        log.warning("Catalog is empty — retrieval will return no results.")
        return None, []

    corpus = _build_corpus(items)
    bm25 = BM25Okapi(corpus)
    log.info("BM25 index built over %d catalog items.", len(items))
    return bm25, items


# Module-level singletons — initialised on first call to search() to allow
# the catalog to be loaded (via load_catalog()) before we read it here.
_bm25 = None
_items: list[dict] = []
_bm25_ready = False


def _ensure_bm25() -> None:
    """Build BM25 index once; idempotent afterwards."""
    global _bm25, _items, _bm25_ready
    if not _bm25_ready:
        _bm25, _items = _init_bm25()
        _bm25_ready = True


# ---------------------------------------------------------------------------
# 2.  EMBEDDING CACHE — populated once, reused for every query
# ---------------------------------------------------------------------------

# Stores one float vector per catalog item, keyed by list position.
# Using a plain list (not a dict) for O(1) index access.
_catalog_embeddings: Optional[list[list[float]]] = None
_embedding_provider: Optional[str] = None  # "openai" | "google" | None


def _detect_provider() -> Optional[str]:
    """
    Return which embedding provider to use, based on available env vars.
    Priority: OpenAI > Google.  Returns None if no key found → BM25-only mode.
    """
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GOOGLE_API_KEY"):
        return "google"
    return None


def _embed_openai(texts: list[str]) -> list[list[float]]:
    """
    Batch-embed texts using OpenAI text-embedding-3-small.

    text-embedding-3-small is chosen over ada-002 for its better
    retrieval performance at lower cost (per OpenAI MTEB benchmarks).
    Batch size is capped at 2048 inputs per OpenAI's API limits.
    """
    import openai  # type: ignore[import-untyped]

    client = openai.OpenAI()  # reads OPENAI_API_KEY from env automatically

    # Process in chunks to respect API batch limits.
    BATCH = 2048
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i : i + BATCH]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk,
        )
        # API returns items sorted by index; collect in order.
        all_vectors.extend([item.embedding for item in response.data])

    return all_vectors


def _embed_google(texts: list[str]) -> list[list[float]]:
    """
    Batch-embed texts using Google's text-embedding-004 model via the
    google-generativeai SDK.
    """
    import google.generativeai as genai  # type: ignore[import-untyped]

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    # Google's embed_content accepts a single string or a list.
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=texts,
        task_type="RETRIEVAL_DOCUMENT",
    )
    # result["embedding"] is a list of vectors when content is a list.
    return result["embedding"]


def _embed(texts: list[str], provider: str) -> list[list[float]]:
    """Dispatch to the correct embedding backend."""
    if provider == "openai":
        return _embed_openai(texts)
    if provider == "google":
        return _embed_google(texts)
    raise ValueError(f"Unknown embedding provider: {provider!r}")


def _build_catalog_embeddings() -> None:
    """
    Pre-compute and cache embeddings for every catalog item.

    Called once (lazily) on the first search() invocation.  Subsequent
    calls are no-ops — the vectors live in _catalog_embeddings for the
    lifetime of the process.

    Each item is represented as:  "name. description. test_type."
    The period separators help sentence-aware tokenisers form clean sentences.
    """
    global _catalog_embeddings, _embedding_provider

    provider = _detect_provider()
    if provider is None:
        log.info(
            "No embedding API key found (OPENAI_API_KEY / GOOGLE_API_KEY). "
            "Falling back to BM25-only retrieval."
        )
        return  # _catalog_embeddings stays None → pure BM25 mode

    texts = [
        f"{item.get('name', '')}. "
        f"{item.get('description', '')}. "
        f"{item.get('test_type', '')}."
        for item in _items
    ]

    try:
        log.info(
            "Computing catalog embeddings via %s for %d items…",
            provider, len(texts),
        )
        _catalog_embeddings = _embed(texts, provider)
        _embedding_provider = provider
        log.info("Catalog embeddings cached (%d vectors).", len(_catalog_embeddings))
    except Exception as exc:  # noqa: BLE001
        # Graceful degradation: if embedding fails (quota, network, etc.),
        # we log a warning and fall back to pure BM25 rather than crashing.
        log.warning(
            "Embedding pre-computation failed (%s: %s). "
            "Falling back to BM25-only retrieval.",
            type(exc).__name__, exc,
        )
        _catalog_embeddings = None


# ---------------------------------------------------------------------------
# 3.  SCORING HELPERS
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_sim(a, b) = (a · b) / (|a| * |b|)

    Returns a value in [-1, 1]; for embedding vectors this is effectively
    in [0, 1] because models are trained to produce non-negative similarities
    for related content.
    """
    dot = sum(x * y for x, y in zip(a, b))
    # Guard against zero-vectors (shouldn't happen with well-trained models).
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    norm_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (norm_a * norm_b)


def _minmax_normalise(scores: list[float]) -> list[float]:
    """
    Scale a list of scores to [0, 1] using min-max normalisation.

    This is necessary before combining BM25 and cosine scores because they
    live on completely different scales:
      - BM25 scores are unbounded positive floats (typically 0–20 for short docs).
      - Cosine similarities are in [-1, 1], practically [0, 1].

    Without normalisation a weighted average would be dominated by whichever
    method produces larger raw numbers.

    Edge case: if all scores are identical, return uniform 1.0 to avoid
    division by zero (all items are equally relevant — keep them all).
    """
    lo = min(scores)
    hi = max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    span = hi - lo
    return [(s - lo) / span for s in scores]


# ---------------------------------------------------------------------------
# 4.  PUBLIC API
# ---------------------------------------------------------------------------

def search(query: str, top_k: int = 10) -> list[dict]:
    """
    Return up to ``top_k`` catalog items most relevant to ``query``.

    Scoring pipeline
    ----------------
    Step 1  — BM25 lexical score
        Tokenise the query the same way the corpus was tokenised (lowercase
        whitespace split).  BM25Okapi returns one score per document.

    Step 2  — Embedding similarity score (optional)
        If catalog embeddings were pre-computed, embed the *query* (one API
        call, not one per catalog item!) and compute cosine similarity against
        every cached catalog vector.

    Step 3  — Normalise both score lists to [0, 1]
        Min-max normalisation so the two signals have comparable magnitude
        before combining.

    Step 4  — Weighted average
        hybrid_score = BM25_WEIGHT * bm25_norm + EMBED_WEIGHT * embed_norm
        Default weights are 0.5 / 0.5 (equal contribution).  In practice
        you might tune EMBED_WEIGHT higher for open-ended role descriptions
        and BM25_WEIGHT higher for exact product-name lookups.

    Step 5  — Sort descending, return top_k dicts
        Each returned dict is a direct reference to the in-memory catalog
        entry (no copy), augmented with a "_score" key for observability.

    Parameters
    ----------
    query:
        Raw free-text query (e.g. the user's clarified hiring intent).
    top_k:
        Maximum number of results to return.

    Returns
    -------
    list[dict]
        Catalog items ordered by hybrid score, best match first.
        Each dict includes a ``_score`` float for debugging / eval.
    """
    # ------------------------------------------------------------------
    # Initialise indices on first call (lazy so load_catalog() runs first).
    # ------------------------------------------------------------------
    _ensure_bm25()

    if not _items:
        return []  # Catalog is empty; nothing to search.

    if _catalog_embeddings is None:
        # First-time call or catalog was loaded after module import — try to
        # build embeddings now.  Subsequent calls are no-ops.
        _build_catalog_embeddings()

    # ------------------------------------------------------------------
    # STEP 1: BM25 lexical scoring
    # ------------------------------------------------------------------
    # Tokenise query identically to how corpus documents were tokenised.
    query_tokens = query.lower().split()

    # get_scores() returns a numpy array with one BM25 score per document.
    # Higher is better; 0 means no token overlap.
    bm25_raw: list[float] = _bm25.get_scores(query_tokens).tolist()

    # ------------------------------------------------------------------
    # STEP 2: Embedding similarity scoring (skipped in BM25-only mode)
    # ------------------------------------------------------------------
    embed_raw: list[float] | None = None

    if _catalog_embeddings is not None and _embedding_provider is not None:
        try:
            # Embed the query — ONE API call regardless of catalog size.
            # We use RETRIEVAL_QUERY task type where the API supports it
            # (Google), otherwise the same model handles both sides (OpenAI).
            if _embedding_provider == "openai":
                import openai
                client = openai.OpenAI()
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[query],
                )
                query_vec = response.data[0].embedding

            elif _embedding_provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=query,
                    task_type="RETRIEVAL_QUERY",  # different task type for queries!
                )
                query_vec = result["embedding"]

            else:
                query_vec = None

            if query_vec is not None:
                # Cosine similarity between query and each pre-cached catalog vector.
                embed_raw = [
                    _cosine_similarity(query_vec, doc_vec)
                    for doc_vec in _catalog_embeddings
                ]

        except Exception as exc:  # noqa: BLE001
            # If the query embedding fails at runtime (e.g. API rate limit),
            # degrade gracefully to BM25-only for this particular call.
            log.warning(
                "Query embedding failed (%s: %s); using BM25-only for this request.",
                type(exc).__name__, exc,
            )
            embed_raw = None

    # ------------------------------------------------------------------
    # STEP 3: Normalise both score lists to [0, 1]
    # ------------------------------------------------------------------
    bm25_norm = _minmax_normalise(bm25_raw)

    if embed_raw is not None:
        embed_norm = _minmax_normalise(embed_raw)
    else:
        embed_norm = None

    # ------------------------------------------------------------------
    # STEP 4: Combine into a single hybrid score (weighted average)
    # ------------------------------------------------------------------
    # Equal weighting is a reasonable starting point.  If you A/B test and
    # find that users send exact test names often, increase BM25_WEIGHT.
    # If queries are mostly open-ended job descriptions, increase EMBED_WEIGHT.
    BM25_WEIGHT = 0.5
    EMBED_WEIGHT = 0.5  # only used when embeddings are available

    hybrid: list[float]
    if embed_norm is not None:
        # Full hybrid mode: weighted average of normalised signals.
        hybrid = [
            BM25_WEIGHT * b + EMBED_WEIGHT * e
            for b, e in zip(bm25_norm, embed_norm)
        ]
    else:
        # BM25-only fallback: normalised BM25 score is the only signal.
        hybrid = bm25_norm

    # ------------------------------------------------------------------
    # STEP 5: Sort by score descending, attach score, return top_k
    # ------------------------------------------------------------------
    ranked = sorted(
        enumerate(hybrid),
        key=lambda t: t[1],
        reverse=True,
    )

    results: list[dict] = []
    for idx, score in ranked[:top_k]:
        # Shallow-copy the catalog dict so callers can't mutate the cache,
        # then attach _score for observability / eval harness.
        entry = {**_items[idx], "_score": round(score, 6)}
        results.append(entry)

    return results
