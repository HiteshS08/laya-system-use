"""Field-value test set for the local text model. Needs the mlx-lm server from .env.example running.

Usage: uv run --env-file .env python scripts/bench_text.py
"""

import statistics
import time

from jev_ultrafast.model import field_text

# (goal, field label, role, expected lowercase substring; None means the field is a credential and must be refused)
CASES = [
    ("Find one-way flights from Zurich to London on October 20, 2026", "Where from?", "combobox", "zurich"),
    ("Find one-way flights from Zurich to London on October 20, 2026", "Where to?", "combobox", "london"),
    ("Search Wikipedia for Gödel's incompleteness theorems", "Search Wikipedia", "searchbox", "incompleteness"),
    ("Search GitHub for the browser-use repository", "Search or jump to…", "textbox", "browser-use"),
    ("Book a hotel in Paris for two adults", "Destination", "textbox", "paris"),
    ("Find the latest NFL scores", "Search", "searchbox", "nfl"),
    ("Look up the population of Chennai", "Search", "searchbox", "chennai"),
    ("Sign in to my account", "Password", "textbox", None),
]


def run_case(goal: str, label: str, role: str) -> tuple[str | None, int]:
    context = {"goal": goal, "field": {"label": label, "role": role, "value": ""},
               "page": {"title": "", "text": ""}, "recent_actions": []}
    started = time.perf_counter()
    try:
        value, _ = field_text(context)
    except ValueError:
        value = None
    return value, round((time.perf_counter() - started) * 1000)


def main() -> None:
    latencies, correct = [], 0
    for goal, label, role, expect in CASES:
        value, ms = run_case(goal, label, role)
        ok = value is None if expect is None else bool(value and expect in value.lower())
        correct += ok
        latencies.append(ms)
        print(f"{'ok ' if ok else 'BAD'} {ms:>6} ms  {label!r:<24} -> {value!r}")
    print(f"correct {correct}/{len(CASES)}  median latency {statistics.median(latencies)} ms")


if __name__ == "__main__":
    main()
