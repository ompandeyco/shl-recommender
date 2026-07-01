"""
catalog.py
----------
Loads the SHL assessment catalog from disk and exposes a lightweight
query interface consumed by retrieval.py.

Responsibilities
----------------
1. Load ``data/catalog.json`` once at startup into memory.
2. Expose ``get_all()`` → list of raw catalog dicts.
3. Expose ``get_by_id(assessment_id: str)`` for O(1) exact lookup.
4. Expose ``filter_by_type(test_type: str)`` for coarse pre-filtering before
   semantic search.

Expected catalog schema (per entry in catalog.json)
----------------------------------------------------
  id                 str   — unique slug/identifier
  name               str   — human-readable assessment name
  url                str   — canonical SHL product-page URL
  test_type          str   — category (e.g. "Ability & Aptitude")
  description        str   — 1-3 sentence blurb
  duration_minutes   int   — approximate test length (0 = unknown)
  remote_proctoring  bool  — whether remote proctoring is available
  adaptive           bool  — whether the test is adaptive/IRT-based

This module intentionally contains *no* ranking or LLM calls — those belong
in retrieval.py and agent.py respectively.
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolved relative to this file so the app works regardless of cwd.
# ---------------------------------------------------------------------------
_CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog.json"

# In-memory store populated by load_catalog().
_catalog: list[dict] = []

# Fast O(1) id → dict mapping built alongside _catalog.
_catalog_by_id: dict[str, dict] = {}


def load_catalog() -> None:
    """
    Read ``data/catalog.json`` into the module-level ``_catalog`` list.

    Called once during application startup (lifespan hook in main.py).
    Raises FileNotFoundError if the catalog file is missing.
    Raises ValueError if the file is not a JSON array.
    """
    global _catalog, _catalog_by_id

    raw = _CATALOG_PATH.read_text(encoding="utf-8")
    data = json.loads(raw)

    if not isinstance(data, list):
        raise ValueError(
            f"catalog.json must contain a JSON array; got {type(data).__name__}"
        )

    _catalog = data
    # Build the id index in a single pass so lookups are O(1) forever.
    _catalog_by_id = {item["id"]: item for item in _catalog if "id" in item}


def get_all() -> list[dict]:
    """Return every assessment in the catalog as a list of dicts."""
    return _catalog


def get_by_id(assessment_id: str) -> dict | None:
    """
    Return the assessment dict with the given id, or None if not found.

    Parameters
    ----------
    assessment_id:
        The unique identifier string stored in catalog.json.
    """
    return _catalog_by_id.get(assessment_id)


def filter_by_type(test_type: str) -> list[dict]:
    """
    Return all assessments whose ``test_type`` matches the given string
    (case-insensitive substring match so partial labels like "ability" work).

    Parameters
    ----------
    test_type:
        E.g. "Ability & Aptitude", "Personality & Behaviour".
    """
    needle = test_type.lower()
    return [
        item for item in _catalog
        if needle in item.get("test_type", "").lower()
    ]
