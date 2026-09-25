"""Paired comparison of live runs on the same tasks: discordant pairs and an exact McNemar test.

With 25 tasks, differences under ~15 points are not distinguishable; this reports how sure a difference is.
Usage: uv run python scripts/compare_runs.py --a artifacts/live/RUN [RUN ...] --b artifacts/live/RUN [RUN ...]
Several runs on one side are combined by per-task majority.
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from math import comb
from pathlib import Path


def outcomes(records: Sequence[dict]) -> dict[str, bool]:
    return {r["task"]: bool(r.get("success")) for r in records}


def load_run(run: Path) -> dict[str, bool]:
    return outcomes([json.loads(p.read_text()) for p in sorted(Path(run).glob("*.json"))])


def majority(runs: Sequence[Mapping[str, bool]]) -> dict[str, bool]:
    tasks = sorted(set().union(*runs)) if runs else []
    return {t: sum(bool(r.get(t)) for r in runs) * 2 > len(runs) for t in tasks}


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value: a binomial test on the discordant pairs."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(a: Mapping[str, bool], b: Mapping[str, bool]) -> dict:
    shared = sorted(set(a) & set(b))
    only_a = [t for t in shared if a[t] and not b[t]]
    only_b = [t for t in shared if b[t] and not a[t]]
    return {"n": len(shared), "a_passed": sum(a[t] for t in shared), "b_passed": sum(b[t] for t in shared),
            "only_a": len(only_a), "only_b": len(only_b), "only_a_tasks": only_a, "only_b_tasks": only_b,
            "p_value": mcnemar_exact(len(only_a), len(only_b))}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", nargs="+", type=Path, required=True)
    parser.add_argument("--b", nargs="+", type=Path, required=True)
    args = parser.parse_args(argv)
    a, b = majority([load_run(r) for r in args.a]), majority([load_run(r) for r in args.b])
    print(json.dumps(compare(a, b), indent=1))


if __name__ == "__main__":
    main()
