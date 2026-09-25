"""Benchmark the goal compiler: valid-program rate, latency and output tokens over suite and held-out goals.

Needs the local model server (see .env.example). No cache: every goal is compiled fresh.
Usage: COMPILER_MODEL=<model> uv run --env-file .env python scripts/bench_compile.py > artifacts/bench_compile.json
Hand-judge the printed programs; the valid rate only says they parse.
"""

import json
import statistics
from collections.abc import Callable, Sequence

from evals.heldout_goals import HELDOUT
from evals.live_tasks import TASKS
from jev_ultrafast.compiler import compile_goal
from jev_ultrafast.program import render_program
from jev_ultrafast.textmodel import complete_text


def bench(goals: Sequence[str], complete: Callable) -> dict:
    programs, latencies, tokens, fallback = {}, [], [], 0
    for goal in goals:
        program, meta = compile_goal(goal, complete=complete)
        programs[goal] = render_program(program)
        fallback += program.source == "fallback"
        latencies.append(meta.get("latency_ms", 0))
        tokens.append(meta.get("completion_tokens", 0))
    return {"n": len(goals), "valid": len(goals) - fallback, "fallback": fallback,
            "median_ms": statistics.median(latencies) if latencies else None,
            "median_completion_tokens": statistics.median(tokens) if tokens else None, "programs": programs}


if __name__ == "__main__":
    goals = list(dict.fromkeys([t.goal for t in TASKS.values()] + [g for _, g in HELDOUT]))
    print(json.dumps(bench(goals, complete_text), indent=1, ensure_ascii=False))
