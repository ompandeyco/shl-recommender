"""
qa_checklist.py
---------------
Final QA verification script for the SHL Assessment Recommender.
Run against a live server: uvicorn app.main:app --port 8000

Usage:
    python eval/qa_checklist.py [--url http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog.json"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

results: list[tuple[str, str, str]] = []  # (check_name, status, evidence)


def record(name: str, passed: bool, evidence: str) -> None:
    status = PASS if passed else FAIL
    results.append((name, status, evidence))
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}")
    print(f"         {evidence}")
    print()


def post_chat(client: httpx.Client, messages: list[dict], timeout: float = 35.0):
    t0 = time.perf_counter()
    resp = client.post(f"{BASE_URL}/chat", json={"messages": messages}, timeout=timeout)
    elapsed = time.perf_counter() - t0
    return resp, elapsed


# ---------------------------------------------------------------------------
# Check 1: GET /health
# ---------------------------------------------------------------------------
def check_health(client: httpx.Client) -> None:
    resp = client.get(f"{BASE_URL}/health", timeout=5)
    ok = resp.status_code == 200 and resp.json() == {"status": "ok"}
    record(
        "GET /health returns {status:ok} HTTP 200",
        ok,
        f"HTTP {resp.status_code}  body={resp.text[:120]}"
    )


# ---------------------------------------------------------------------------
# Check 2: Response schema
# ---------------------------------------------------------------------------
def check_schema(client: httpx.Client) -> None:
    resp, _ = post_chat(client, [{"role": "user", "content": "I need a personality test for sales managers"}])
    errors = []
    if resp.status_code != 200:
        errors.append(f"HTTP {resp.status_code}")
    else:
        body = resp.json()
        for field in ("reply", "recommendations", "end_of_conversation"):
            if field not in body:
                errors.append(f"missing '{field}'")
        extra = set(body.keys()) - {"reply", "recommendations", "end_of_conversation"}
        if extra:
            errors.append(f"extra fields: {extra}")
        if not isinstance(body.get("reply"), str):
            errors.append("'reply' not a string")
        if not isinstance(body.get("recommendations"), list):
            errors.append("'recommendations' not a list")
        if not isinstance(body.get("end_of_conversation"), bool):
            errors.append("'end_of_conversation' not a bool")
        for i, rec in enumerate(body.get("recommendations", [])):
            for rf in ("name", "url", "test_type"):
                if rf not in rec:
                    errors.append(f"rec[{i}] missing '{rf}'")
            extra_rf = set(rec.keys()) - {"name", "url", "test_type"}
            if extra_rf:
                errors.append(f"rec[{i}] extra fields: {extra_rf}")

    record("Schema: reply/recommendations/end_of_conversation", not errors,
           "Schema valid" if not errors else "; ".join(errors))


# ---------------------------------------------------------------------------
# Check 3: All URLs exist in catalog
# ---------------------------------------------------------------------------
def check_catalog_urls(client: httpx.Client) -> None:
    catalog = json.loads(CATALOG_PATH.read_text())
    valid_urls = {item["url"] for item in catalog if "url" in item}

    resp, _ = post_chat(client, [
        {"role": "user", "content": "I need cognitive and personality tests for a mid-level software engineer role"},
    ])
    bad_urls = []
    if resp.status_code == 200:
        for rec in resp.json().get("recommendations", []):
            url = rec.get("url", "")
            if url and url not in valid_urls:
                bad_urls.append(f"{rec.get('name')!r} -> {url!r}")

    record("All recommendation URLs exist in catalog.json",
           not bad_urls,
           f"Checked {len(valid_urls)} catalog URLs. Bad: {bad_urls or 'none'}")


# ---------------------------------------------------------------------------
# Check 4: CLARIFY/REFUSE return empty recommendations
# ---------------------------------------------------------------------------
def check_clarify_empty_recs(client: httpx.Client) -> None:
    # Vague message that should trigger CLARIFY
    resp_c, _ = post_chat(client, [{"role": "user", "content": "Help me hire someone"}])
    # Out-of-scope message that should trigger REFUSE
    resp_r, _ = post_chat(client, [{"role": "user", "content": "Write me a cover letter for a marketing job"}])

    errors = []
    for label, resp in [("CLARIFY-candidate", resp_c), ("REFUSE-candidate", resp_r)]:
        if resp.status_code == 200:
            body = resp.json()
            recs = body.get("recommendations", [])
            reply = body.get("reply", "")
            # Only flag if it contains recs AND looks like neither a rate-limit nor RETRIEVE
            if recs and "temporarily unable" not in reply.lower():
                errors.append(f"{label}: got {len(recs)} recs, expected 0")
        else:
            errors.append(f"{label}: HTTP {resp.status_code}")

    record("CLARIFY/REFUSE return empty recommendations",
           not errors,
           "OK — both vague+OOT returned 0 recs" if not errors else "; ".join(errors))


# ---------------------------------------------------------------------------
# Check 5: Populated recommendations list is 1–10
# ---------------------------------------------------------------------------
def check_rec_count(client: httpx.Client) -> None:
    resp, _ = post_chat(client, [
        {"role": "user", "content": "I need numerical reasoning and personality tests for a senior finance manager role, on-site proctoring"},
    ])
    errors = []
    if resp.status_code == 200:
        recs = resp.json().get("recommendations", [])
        if recs:
            if not (1 <= len(recs) <= 10):
                errors.append(f"Got {len(recs)} recommendations (must be 1–10)")
            else:
                pass  # good
        # If recs is empty, it may have CLARIFYed — not an error for this check
    else:
        errors.append(f"HTTP {resp.status_code}")

    count = len(resp.json().get("recommendations", [])) if resp.status_code == 200 else "?"
    record("Recommendation count is between 1 and 10 when populated",
           not errors,
           f"Got {count} recs" + ("; " + "; ".join(errors) if errors else ""))


# ---------------------------------------------------------------------------
# Check 6: Empty messages array → clean error, not 500
# ---------------------------------------------------------------------------
def check_empty_messages(client: httpx.Client) -> None:
    resp, _ = post_chat(client, [])
    ok = resp.status_code in (200, 400, 422)
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    crashed = resp.status_code == 500
    record("Empty messages array - clean error, not 500",
           ok and not crashed,
           f"HTTP {resp.status_code}  reply={body.get('reply', body)!r:.100}")


# ---------------------------------------------------------------------------
# Check 7: 8+ turns → end_of_conversation eventually true
# ---------------------------------------------------------------------------
def check_turn_cap(client: httpx.Client) -> None:
    history = []
    eoc_seen = False
    turns_run = 0
    error = None
    rate_limited = False

    # Build a realistic conversation that accumulates context across turns,
    # ensuring the server sees >= 8 user turns and triggers the hard cap.
    exchanges = [
        "I need assessments for a software engineer role.",
        "Looking for something with coding ability.",
        "Also numerical reasoning.",
        "And personality too.",
        "Under 60 minutes total.",
        "Remote proctoring required.",
        "Graduate level preferred.",
        "Also verbal reasoning please.",
        "That's all I need, please recommend now.",
    ]

    for msg in exchanges:
        time.sleep(5)  # avoid rate limits between turns
        history.append({"role": "user", "content": msg})
        try:
            resp, elapsed = post_chat(client, history)
        except Exception as exc:
            error = str(exc)
            break

        turns_run += 1
        if resp.status_code != 200:
            error = f"HTTP {resp.status_code} on turn {turns_run}"
            break
        body = resp.json()
        reply = body.get("reply", "")

        # Don't count rate-limit replies as real assistant turns
        if "temporarily unable to process" in reply.lower():
            rate_limited = True
            history.pop()  # remove the user turn we just added so history stays valid
            turns_run -= 1
            continue

        history.append({"role": "assistant", "content": reply})
        if body.get("end_of_conversation"):
            eoc_seen = True
            break

    evidence = (
        f"EOC reached after {turns_run} turns"
        if eoc_seen
        else f"EOC never seen after {turns_run} turns"
        + ("; rate limit hit during test — result may be unreliable" if rate_limited else "")
        + (f"; error={error}" if error else "")
    )
    record("8+ turns - end_of_conversation becomes True",
           eoc_seen and not error,
           evidence)


# ---------------------------------------------------------------------------
# Check 8: All /chat calls return under 30 s
# ---------------------------------------------------------------------------
def check_latency(client: httpx.Client) -> None:
    calls = [
        [{"role": "user", "content": "Assessments for a sales graduate, personality and verbal reasoning"}],
        [{"role": "user", "content": "I need a cognitive test for a mid-level analyst"}],
    ]
    times = []
    errors = []
    for msgs in calls:
        try:
            resp, elapsed = post_chat(client, msgs, timeout=35.0)
            times.append(elapsed)
            if elapsed >= 30.0:
                errors.append(f"{elapsed:.1f}s >= 30s limit")
        except httpx.TimeoutException:
            errors.append("Request timed out (>35s)")

    avg = sum(times) / len(times) if times else 0
    record("All /chat calls return well under 30 s",
           not errors,
           f"Times: {[f'{t:.2f}s' for t in times]}  avg={avg:.2f}s" + ("; " + "; ".join(errors) if errors else ""))


# ---------------------------------------------------------------------------
# Check 9: NOTE — manual check (fallback test requires .env edit)
# ---------------------------------------------------------------------------
def check_fallback_note() -> None:
    """
    The Groq→Gemini fallback cannot be tested automatically without mutating
    .env. We report WARN with instructions for the manual step.
    """
    # Read .env to check if GOOGLE_API_KEY is present (not prefixed with _)
    env_path = Path(__file__).parent.parent / ".env"
    env_text = env_path.read_text() if env_path.exists() else ""
    
    has_google = "GOOGLE_API_KEY=" in env_text and "_GOOGLE_API_KEY=" not in env_text.replace("GOOGLE_API_KEY=", "")
    groq_active = "GROQ_API_KEY=" in env_text
    
    # Check if _GOOGLE_API_KEY (disabled) is present
    has_disabled_google = "_GOOGLE_API_KEY=" in env_text

    note_parts = []
    if groq_active:
        note_parts.append("GROQ_API_KEY is active (primary)")
    if has_google:
        note_parts.append("GOOGLE_API_KEY is active (fallback)")
    if has_disabled_google and not has_google:
        note_parts.append("⚠ _GOOGLE_API_KEY is DISABLED (prefixed with _) — fallback to Gemini will NOT work")
    
    passed = groq_active and has_google
    record("LLM fallback (Groq→Gemini) — .env key check",
           passed,
           "; ".join(note_parts) if note_parts else "Could not read .env")


# ---------------------------------------------------------------------------
# Check 10: .env not in git history
# ---------------------------------------------------------------------------
def check_env_not_in_git() -> None:
    import subprocess
    result = subprocess.run(
        ["git", "log", "--all", "--full-history", "--", ".env"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent)
    )
    committed = bool(result.stdout.strip())
    in_gitignore = ".env" in (Path(__file__).parent.parent / ".gitignore").read_text()
    record(".env is NOT committed to git",
           not committed and in_gitignore,
           f"git log output: {'(empty — good)' if not committed else result.stdout[:200]}; "
           f".gitignore includes .env: {in_gitignore}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    global BASE_URL
    BASE_URL = args.url.rstrip("/")

    print(f"\n{'='*60}")
    print("  SHL RECOMMENDER — QA CHECKLIST")
    print(f"  Target: {BASE_URL}")
    print(f"{'='*60}\n")

    with httpx.Client(timeout=35.0) as client:
        # Connectivity check first
        try:
            client.get(f"{BASE_URL}/health", timeout=5)
        except Exception as exc:
            print(f"  ERROR: Cannot reach {BASE_URL} — {exc}")
            print("  Make sure uvicorn is running: uvicorn app.main:app --port 8000")
            sys.exit(1)

        check_health(client)
        check_schema(client)
        check_catalog_urls(client)
        check_clarify_empty_recs(client)
        check_rec_count(client)
        check_empty_messages(client)
        check_turn_cap(client)
        check_latency(client)

    check_fallback_note()
    check_env_not_in_git()

    # Summary table
    print(f"\n{'='*60}")
    print("  FINAL QA SUMMARY")
    print(f"{'='*60}\n")
    print(f"  {'#':<4} {'Check':<52} Status")
    print(f"  {'-'*4} {'-'*52} ------")
    passed = 0
    failed = 0
    for i, (name, status, _) in enumerate(results, 1):
        print(f"  {i:<4} {name:<52} {status}")
        if "PASS" in status:
            passed += 1
        else:
            failed += 1

    print(f"\n  Total: {passed} PASS, {failed} FAIL out of {len(results)}\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
