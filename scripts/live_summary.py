"""Summarise a live-eval run directory: pass counts by category, routes, and latency medians.

Usage: uv run python scripts/live_summary.py artifacts/live/<run>
"""

import json
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize(records: Sequence[dict]) -> dict:
    by_category: dict[str, list[int]] = {}
    for r in records:
        tally = by_category.setdefault(r.get("category", ""), [0, 0])
        tally[0] += bool(r.get("success"))
        tally[1] += 1
    decisions = [d for r in records for d in r.get("decisions", [])]
    actor_ms = [d["actor_ms"] for d in decisions if d.get("actor_ms")]
    planner_ms = [c["latency_ms"] for r in records for c in r.get("planner_calls", []) if "latency_ms" in c]
    actions = sum(len(r.get("steps", [])) for r in records)
    return {
        "passed": sum(bool(r.get("success")) for r in records), "n": len(records), "by_category": by_category,
        "routes": dict(Counter(d.get("route") or "none" for d in decisions)),
        "median_decision_ms": _median([d["latency_ms"] for d in decisions if d.get("latency_ms") is not None]),
        "median_actor_ms": _median(actor_ms), "median_planner_ms": _median(planner_ms),
        "planner_errors": sum("error" in c for r in records for c in r.get("planner_calls", [])),
        "wall_s_per_action": round(sum(r.get("seconds") or 0 for r in records) / actions, 2) if actions else None,
    }


if __name__ == "__main__":
    run = Path(sys.argv[1])
    print(json.dumps(summarize([json.loads(p.read_text()) for p in sorted(run.glob("*.json"))]), indent=1))
