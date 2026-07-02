"""
replay_harness.py
-----------------
Offline evaluation script that replays reference conversation traces from
eval/traces/ against a running FastAPI server and scores the results.

Usage
-----
    # Server must already be running:
    uvicorn app.main:app --port 8000

    # Run the harness:
    python eval/replay_harness.py
    python eval/replay_harness.py --url http://localhost:8000
    python eval/replay_harness.py --traces ./eval/traces --catalog ./data/catalog.json

Exit codes
----------
    0  — all traces passed (no hard failures)
    1  — one or more hard failures detected (bad URL, schema violation, etc.)

Metrics computed
----------------
    Recall@10
        len(gold_names & retrieved_names) / len(gold_names)
        = fraction of the labeled shortlist that appeared in the agent's
          final recommendations (capped at top-10).
        0.0 for traces with expected_shortlist=[] (e.g. REFUSE traces).

    Mean Recall@10
        Arithmetic mean of Recall@10 across all traces.

Quality flags (hard failures)
------------------------------
    EARLY_REC   — agent returned >= 1 recommendation on turn 1 for a trace
                  whose gold turn 1 assistant reply had 0 recommendations.
                  (Agent should clarify, not jump to recommendations on
                  first contact when the query is vague.)
    BAD_URL     — agent returned a URL not found verbatim in catalog.json.
                  The spec says every URL must come from the scraped catalog.
    BAD_SCHEMA  — /chat returned a non-200 response or a payload missing
                  required fields (reply, recommendations, end_of_conversation).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"reply", "recommendations", "end_of_conversation"}
REQUIRED_REC_FIELDS = {"name", "url", "test_type"}


def validate_response_schema(payload: dict) -> list[str]:
    """
    Return a list of schema violation strings (empty = valid).

    Checks that the top-level response has required fields and that every
    entry in `recommendations` has name, url, and test_type.
    """
    errors: list[str] = []
    missing_top = REQUIRED_FIELDS - payload.keys()
    if missing_top:
        errors.append(f"Missing top-level fields: {missing_top}")
        return errors   # can't check deeper if top-level is broken

    if not isinstance(payload["recommendations"], list):
        errors.append("'recommendations' is not a list")
    else:
        for i, rec in enumerate(payload["recommendations"]):
            missing_rec = REQUIRED_REC_FIELDS - rec.keys()
            if missing_rec:
                errors.append(f"recommendations[{i}] missing fields: {missing_rec}")

    if not isinstance(payload["reply"], str) or not payload["reply"].strip():
        errors.append("'reply' is empty or not a string")

    if not isinstance(payload["end_of_conversation"], bool):
        errors.append("'end_of_conversation' is not a bool")

    return errors


# ---------------------------------------------------------------------------
# Catalog URL index
# ---------------------------------------------------------------------------

def load_catalog_urls(catalog_path: Path) -> set[str]:
    """
    Return the set of all canonical URLs present in catalog.json.

    Used to flag any agent-returned URL that was fabricated rather than
    sourced from the scraped catalog — a hard eval criterion.
    """
    if not catalog_path.exists():
        print(f"  [WARN] Catalog not found at {catalog_path}; URL checks skipped.")
        return set()
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    return {item["url"] for item in data if "url" in item}


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------

def load_traces(traces_dir: Path) -> list[tuple[Path, dict]]:
    """
    Load all *.json trace files from traces_dir, alphabetically sorted.

    Returns list of (path, parsed_dict) so callers can log filenames.
    """
    files = sorted(traces_dir.glob("*.json"))
    traces: list[tuple[Path, dict]] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            traces.append((f, data))
        except json.JSONDecodeError as exc:
            print(f"  [SKIP] {f.name}: invalid JSON — {exc}", file=sys.stderr)
    return traces


# ---------------------------------------------------------------------------
# HTTP replay
# ---------------------------------------------------------------------------

def call_chat(
    client: httpx.Client,
    base_url: str,
    messages: list[dict],
    timeout: float = 35.0,
) -> tuple[int, dict | None, float]:
    """
    POST /chat with the given message history.

    Returns
    -------
    (status_code, parsed_body_or_None, elapsed_seconds)
    """
    url = f"{base_url.rstrip('/')}/chat"
    payload = {"messages": messages}
    time.sleep(5)  # Add a delay of 4-5 seconds between each request
    t0 = time.perf_counter()
    try:
        resp = client.post(url, json=payload, timeout=timeout)
        elapsed = time.perf_counter() - t0
        try:
            body = resp.json()
        except Exception:
            body = None
        return resp.status_code, body, elapsed
    except httpx.TimeoutException:
        elapsed = time.perf_counter() - t0
        return 0, None, elapsed
    except httpx.RequestError as exc:
        elapsed = time.perf_counter() - t0
        print(f"  [ERROR] Request failed: {exc}", file=sys.stderr)
        return 0, None, elapsed


def replay_trace(
    trace: dict,
    base_url: str,
    client: httpx.Client,
    catalog_urls: set[str],
) -> dict:
    """
    Replay a single trace against the server.

    Strategy
    --------
    We extract only the *user* turns from the trace and send them to the
    server one at a time, building up the running message history with each
    turn.  The assistant replies come from the *server* (not from the trace
    gold data) — this is the core of the replay: we test what the live agent
    actually does with the recorded user messages.

    After each server response we append the server's reply to the running
    history so the next turn has the full context (stateless API requires
    the client to maintain history).

    Returns
    -------
    dict with keys:
        trace_name      str
        description     str
        server_turns    list[dict]   — raw server payloads per user turn
        gold_shortlist  list[str]    — expected_shortlist from trace
        flags           list[str]    — human-readable quality failures
        elapsed_total   float        — total wall-clock seconds for the trace
    """
    trace_name = trace.get("description", "?")
    gold_shortlist: list[str] = trace.get("expected_shortlist") or []
    gold_turns = [t for t in trace.get("turns", []) if t.get("role") == "assistant"]
    user_turns = [t for t in trace.get("turns", []) if t.get("role") == "user"]

    # Running history sent to the server.  Starts empty; we append as we go.
    running_history: list[dict] = []

    server_turns: list[dict] = []
    flags: list[str] = []
    elapsed_total = 0.0

    for turn_idx, user_turn in enumerate(user_turns):
        running_history.append({"role": "user", "content": user_turn["content"]})

        status, body, elapsed = call_chat(
            client, base_url, running_history
        )
        elapsed_total += elapsed

        # ── Schema validation ──────────────────────────────────────────────
        if status == 0:
            flags.append(f"BAD_SCHEMA (turn {turn_idx + 1}): request timed out or connection failed")
            # Can't continue replay without a response — break here.
            break

        if status != 200:
            flags.append(f"BAD_SCHEMA (turn {turn_idx + 1}): HTTP {status}")
            break

        if body is None:
            flags.append(f"BAD_SCHEMA (turn {turn_idx + 1}): non-JSON response")
            break

        schema_errors = validate_response_schema(body)
        if schema_errors:
            for err in schema_errors:
                flags.append(f"BAD_SCHEMA (turn {turn_idx + 1}): {err}")
            # Still record the turn so we can inspect it; don't break.
            
        reply_content = body.get("reply", "")
        if "temporarily unable to process, please retry" in reply_content.lower():
            print(f"  [WARN] Turn {turn_idx + 1} hit a rate limit! This data point is invalid.")
            flags.append(f"RATE_LIMIT (turn {turn_idx + 1}): API rate limit reached")

        server_turns.append({"turn": turn_idx + 1, "elapsed": round(elapsed, 2), **body})

        # ── EARLY_REC flag ─────────────────────────────────────────────────
        # On turn 1, if the gold trace expected 0 recommendations (i.e. the
        # agent should have asked a clarifying question), but the server
        # returned >= 1, flag it.  This tests that the agent doesn't jump
        # to conclusions on vague first messages.
        if turn_idx == 0:
            gold_turn_1_recs = (
                gold_turns[0].get("recommendations", []) if gold_turns else []
            )
            server_recs_turn_1 = body.get("recommendations", [])
            if len(gold_turn_1_recs) == 0 and len(server_recs_turn_1) >= 1:
                flags.append(
                    f"EARLY_REC (turn 1): agent returned {len(server_recs_turn_1)} "
                    f"recommendation(s) when gold expected 0 "
                    f"(should have clarified first)"
                )

        # ── BAD_URL flag ───────────────────────────────────────────────────
        # Every URL returned by the agent must exist verbatim in catalog.json.
        # This is a hard eval criterion — hallucinated URLs are a spec failure.
        if catalog_urls:  # skip check if catalog not loaded
            for rec in body.get("recommendations", []):
                url = rec.get("url", "")
                if url and url not in catalog_urls:
                    flags.append(
                        f"BAD_URL (turn {turn_idx + 1}): "
                        f"{rec.get('name', '?')!r} has URL {url!r} "
                        f"not found in catalog.json"
                    )

        # Append server's reply to history so the next turn has context.
        reply_content = body.get("reply", "")
        running_history.append({"role": "assistant", "content": reply_content})

    return {
        "trace_name": trace_name,
        "description": trace.get("description", ""),
        "server_turns": server_turns,
        "gold_shortlist": gold_shortlist,
        "flags": flags,
        "elapsed_total": round(elapsed_total, 2),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def recall_at_k(gold: list[str], retrieved: list[str], k: int = 10) -> float:
    """
    Recall@K = |{relevant} ∩ {retrieved[:K]}| / |{relevant}|

    Uses case-insensitive name matching so minor capitalisation differences
    between the trace labels and the server response don't cause false zeroes.

    Returns 0.0 if gold is empty (nothing to recall).

    Parameters
    ----------
    gold:       labeled expected assessment names
    retrieved:  names the agent actually recommended (in ranked order)
    k:          cutoff (default 10, matching the spec)
    """
    if not gold:
        return 0.0      # REFUSE / CLARIFY traces — no recommendations expected
    gold_lower = {g.lower() for g in gold}
    retrieved_lower = [r.lower() for r in retrieved[:k]]
    hits = sum(1 for r in retrieved_lower if r in gold_lower)
    return hits / len(gold_lower)


def score_result(result: dict) -> dict:
    """
    Compute Recall@10 for a replayed trace.

    Pulls all recommendation names from the *last* server turn that contained
    any recommendations — this matches how the spec evaluates: the final
    shortlist the agent committed to.

    Returns dict with: recall_at_10, retrieved_names, gold_shortlist.
    """
    gold = result["gold_shortlist"]
    server_turns = result["server_turns"]

    # Collect all recommendations from all turns; use the *last non-empty*
    # turn's list as the final committed shortlist.
    final_recs: list[str] = []
    for turn in server_turns:
        recs = turn.get("recommendations", [])
        if recs:
            final_recs = [r.get("name", "") for r in recs]

    return {
        "recall_at_10": recall_at_k(gold, final_recs, k=10),
        "retrieved_names": final_recs,
        "gold_shortlist": gold,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 20) -> str:
    """ASCII progress bar: [####......] 0.75"""
    filled = round(value * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {value:.2f}"


def print_trace_result(
    idx: int,
    path: Path,
    result: dict,
    scores: dict,
) -> None:
    """Pretty-print a single trace result to stdout."""
    recall = scores["recall_at_10"]
    flags = result["flags"]
    gold = scores["gold_shortlist"]
    retrieved = scores["retrieved_names"]

    print()
    print(f"  TRACE {idx:02d}  {path.name}")
    print(f"  {result['description'][:72]}")
    print()

    # Recall bar
    print(f"    Recall@10 : {_bar(recall)}")

    # Gold vs retrieved
    if gold:
        print(f"    Gold      : {', '.join(gold)}")
    else:
        print(f"    Gold      : (none — REFUSE/CLARIFY trace)")
    if retrieved:
        print(f"    Retrieved : {', '.join(retrieved[:10])}")
    else:
        print(f"    Retrieved : (none)")

    # Turn-level summary
    print(f"    Turns     : {len(result['server_turns'])}  "
          f"({result['elapsed_total']:.1f}s total)")

    # Per-turn recommendation counts
    for t in result["server_turns"]:
        n = len(t.get("recommendations", []))
        eoc = "  [EOC]" if t.get("end_of_conversation") else ""
        print(f"      Turn {t['turn']}: {n} rec(s)  ({t['elapsed']:.1f}s){eoc}")

    # Flags
    if flags:
        for flag in flags:
            print(f"    !! {flag}")
    else:
        print(f"    OK  No quality flags")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay evaluation harness for SHL Assessment Recommender."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        metavar="BASE_URL",
        help="Base URL of the running service (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--traces",
        default=str(Path(__file__).parent / "traces"),
        metavar="DIR",
        help="Directory containing trace *.json files",
    )
    parser.add_argument(
        "--catalog",
        default=str(Path(__file__).parent.parent / "data" / "catalog.json"),
        metavar="PATH",
        help="Path to catalog.json for URL validation",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=35.0,
        metavar="SECS",
        help="Per-request HTTP timeout in seconds (default: 35)",
    )
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=0.0,
        dest="fail_threshold",
        metavar="RECALL",
        help="Exit code 1 if mean Recall@10 falls below this (default: 0.0 = never fail on metric)",
    )
    args = parser.parse_args()

    traces_dir = Path(args.traces)
    if not traces_dir.exists():
        print(f"Error: traces directory not found: {traces_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Load catalog URLs for BAD_URL checks ──────────────────────────────
    catalog_urls = load_catalog_urls(Path(args.catalog))
    if catalog_urls:
        print(f"Loaded {len(catalog_urls)} catalog URLs from {args.catalog}")
    else:
        print(f"[WARN] No catalog URLs loaded — BAD_URL checks will be skipped")

    # ── Load traces ────────────────────────────────────────────────────────
    traces = load_traces(traces_dir)
    if not traces:
        print(f"No *.json trace files found in {traces_dir}")
        sys.exit(0)

    print(f"Found {len(traces)} trace(s) — replaying against {args.url}")
    print()

    # ── Replay ─────────────────────────────────────────────────────────────
    all_scores: list[dict] = []
    all_results: list[dict] = []
    all_flags: list[str] = []
    hard_fail = False

    with httpx.Client(timeout=args.timeout) as client:
        for i, (path, trace) in enumerate(traces, 1):
            result = replay_trace(trace, args.url, client, catalog_urls)
            scores = score_result(result)

            all_scores.append(scores)
            all_results.append(result)
            all_flags.extend(result["flags"])

            if result["flags"]:
                hard_fail = True

            print_trace_result(i, path, result, scores)

    # ── Aggregate summary ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  AGGREGATE SUMMARY")
    print("=" * 60)
    print()

    recalls = [s["recall_at_10"] for s in all_scores]
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0

    # Table header
    col = 44
    print(f"  {'Trace':<{col}}  Recall@10  Flags")
    print(f"  {'-' * col}  ---------  -----")
    for (path, _), scores, result in zip(traces, all_scores, all_results):
        flag_str = str(len(result['flags'])) + " flag(s)" if result['flags'] else "OK"
        print(
            f"  {path.name:<{col}}  "
            f"{scores['recall_at_10']:>9.2f}  "
            f"{flag_str}"
        )

    print()
    print(f"  Mean Recall@10 : {mean_recall:.4f}  {_bar(mean_recall)}")
    print()

    # Flag summary
    if all_flags:
        print(f"  Hard failures ({len(all_flags)} total):")
        for flag in all_flags:
            print(f"    !! {flag}")
    else:
        print(f"  No hard failures detected.")

    print()

    # ── Exit code ─────────────────────────────────────────────────────────
    metric_fail = mean_recall < args.fail_threshold
    if hard_fail or metric_fail:
        if hard_fail:
            print(f"Exit 1: hard failures detected (BAD_URL / BAD_SCHEMA / EARLY_REC)")
        if metric_fail:
            print(f"Exit 1: Mean Recall@10 {mean_recall:.4f} < threshold {args.fail_threshold:.4f}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
