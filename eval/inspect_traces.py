"""
inspect_traces.py
-----------------
Reads all *.json trace files from eval/traces/ and pretty-prints
the key fields of each one: persona, structured facts, turn count,
and the expected/labeled shortlist.

Usage
-----
    python eval/inspect_traces.py
    python eval/inspect_traces.py --traces path/to/other/traces/dir

No scoring, no HTTP calls — read-only inspection only.
"""

from __future__ import annotations

import argparse
import json
import io
import sys

# Force UTF-8 on Windows so ANSI / Unicode output doesn't crash.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

# ANSI colour helpers — degrade gracefully on Windows without colour support
def _cyan(s: str) -> str:    return f"\033[96m{s}\033[0m"
def _yellow(s: str) -> str:  return f"\033[93m{s}\033[0m"
def _green(s: str) -> str:   return f"\033[92m{s}\033[0m"
def _bold(s: str) -> str:    return f"\033[1m{s}\033[0m"
def _dim(s: str) -> str:     return f"\033[2m{s}\033[0m"
def _red(s: str) -> str:     return f"\033[91m{s}\033[0m"


def load_traces(traces_dir: Path) -> list[tuple[Path, dict]]:
    """
    Load every *.json file in traces_dir (alphabetical order).

    Returns
    -------
    list of (path, parsed_dict) tuples — path kept so we can show filenames.
    """
    files = sorted(traces_dir.glob("*.json"))
    if not files:
        return []

    traces: list[tuple[Path, dict]] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            traces.append((f, data))
        except json.JSONDecodeError as exc:
            print(_red(f"  [SKIP] {f.name}: invalid JSON — {exc}"), file=sys.stderr)

    return traces


def _count_turns(trace: dict) -> tuple[int, int]:
    """Return (total_turns, user_turns) for a trace."""
    turns = trace.get("turns", [])
    user_turns = sum(1 for t in turns if t.get("role") == "user")
    return len(turns), user_turns


def _format_facts(facts: dict | None) -> list[str]:
    """
    Pretty-format the structured facts dict into labelled lines.
    Handles None/null values gracefully.
    """
    if not facts:
        return [_dim("  (no facts dict)")]

    lines = []
    label_width = max((len(k) for k in facts), default=0) + 2

    for key, value in facts.items():
        label = (key + ":").ljust(label_width)
        if value is None:
            rendered = _dim("null")
        elif isinstance(value, list):
            rendered = ", ".join(str(v) for v in value) if value else _dim("(empty)")
        elif isinstance(value, bool):
            rendered = _green("yes") if value else _red("no")
        else:
            rendered = str(value)
        lines.append(f"  {_dim(label)} {rendered}")

    return lines


def _format_shortlist(shortlist: list | None, action: str | None) -> list[str]:
    """Render the expected shortlist (or expected action for refusal traces)."""
    lines = []

    if action:
        lines.append(f"  Expected action : {_yellow(action)}")

    if not shortlist:
        lines.append(f"  {_dim('(empty — no recommendations expected)')}")
    else:
        for i, name in enumerate(shortlist, 1):
            lines.append(f"  {_dim(str(i) + '.')} {name}")

    return lines


def print_trace(index: int, path: Path, trace: dict) -> None:
    """Print one trace in a clean, human-readable block."""

    total_turns, user_turns = _count_turns(trace)

    # ── Header ──────────────────────────────────────────────────────────────
    print()
    print(_bold(_cyan(f"=== TRACE {index:02d} --- {path.name} ===")))
    print()

    # ── Description ─────────────────────────────────────────────────────────
    description = trace.get("description", _dim("(no description)"))
    print(_bold("  Description"))
    print(f"  {description}")
    print()

    # ── Persona ─────────────────────────────────────────────────────────────
    persona = trace.get("persona", _dim("(no persona)"))
    print(_bold("  Persona"))
    # Word-wrap at ~80 chars for readability
    words = persona.split()
    line, lines_out = "  ", []
    for word in words:
        if len(line) + len(word) + 1 > 82:
            lines_out.append(line.rstrip())
            line = "  "
        line += word + " "
    if line.strip():
        lines_out.append(line.rstrip())
    print("\n".join(lines_out))
    print()

    # ── Structured facts ────────────────────────────────────────────────────
    print(_bold("  Facts"))
    for line in _format_facts(trace.get("facts")):
        print(line)
    print()

    # ── Turn count ──────────────────────────────────────────────────────────
    print(_bold("  Turns"))
    print(f"  Total : {total_turns}  ({user_turns} user, {total_turns - user_turns} assistant)")
    print()

    # ── Conversation (compact) ──────────────────────────────────────────────
    print(_bold("  Conversation (summary)"))
    for t in trace.get("turns", []):
        role = t.get("role", "?")
        content = t.get("content", "")
        recs = t.get("recommendations", [])

        role_label = _yellow("USER >") if role == "user" else _green("ASST <")
        # Truncate long messages
        snippet = content[:120] + ("..." if len(content) > 120 else "")
        print(f"  {role_label}  {snippet}")

        if recs:
            names = [r.get("name", "?") for r in recs]
            print(f"         {_dim('recs: ' + ', '.join(names))}")
    print()

    # ── Expected shortlist ───────────────────────────────────────────────────
    print(_bold("  Expected shortlist"))
    expected_action = trace.get("expected_action")
    shortlist = trace.get("expected_shortlist")
    for line in _format_shortlist(shortlist, expected_action):
        print(line)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect evaluation trace files in eval/traces/."
    )
    parser.add_argument(
        "--traces",
        default=str(Path(__file__).parent / "traces"),
        metavar="DIR",
        help="Directory containing trace *.json files (default: eval/traces/)",
    )
    args = parser.parse_args()

    traces_dir = Path(args.traces)
    if not traces_dir.exists():
        print(_red(f"Error: traces directory not found: {traces_dir}"), file=sys.stderr)
        sys.exit(1)

    traces = load_traces(traces_dir)

    if not traces:
        print(_yellow(f"No *.json trace files found in {traces_dir}"))
        print(_dim("Drop reference trace files there to get started."))
        sys.exit(0)

    print()
    print(_bold(f"Found {len(traces)} trace(s) in {traces_dir}"))

    for i, (path, trace) in enumerate(traces, 1):
        print_trace(i, path, trace)

    # ── Summary table ────────────────────────────────────────────────────────
    print(_bold(_cyan("=== SUMMARY ===")))
    print()
    col_w = 42
    print(
        f"  {'File':<{col_w}}  {'Turns':>5}  {'Shortlist':>9}  Description"
    )
    print(f"  {'-' * col_w}  {'-----':>5}  {'---------':>9}  -----------")
    for path, trace in traces:
        total_turns, _ = _count_turns(trace)
        shortlist = trace.get("expected_shortlist") or []
        desc = trace.get("description", "")[:55]
        print(
            f"  {path.name:<{col_w}}  {total_turns:>5}  {len(shortlist):>9}  {desc}"
        )
    print()


if __name__ == "__main__":
    main()
