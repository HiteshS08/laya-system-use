"""Live evaluation: run tasks on real public pages with the local stack and record every step for hand labelling.

Needs: a throwaway Chrome on BU_CDP_URL, the mlx-lm server from .env.example, POLICY_BACKEND=laya.
Read-only public sites only. Usage: uv run --env-file .env python scripts/live_eval.py [task ...]
"""

import json
import sys
import time
from pathlib import Path

from jev_ultrafast import Agent

TASKS = {
    "wiki_featured": ("https://en.wikipedia.org/wiki/Main_Page", "Open today's featured article."),
    "wiki_search": (
        "https://en.wikipedia.org/wiki/Main_Page",
        "Find and open the Wikipedia article about Gödel's incompleteness theorems.",
    ),
    "wiki_long_page": (
        "https://en.wikipedia.org/wiki/Gödel%27s_incompleteness_theorems",
        "Open the Wikipedia article about Kurt Gödel, the logician who proved these theorems.",
    ),
    "hn_comments": ("https://news.ycombinator.com/", "Open the comments page of the top story."),
    "gh_issues": ("https://github.com/browser-use/browser-use", "Open the Issues tab of this repository."),
    "flights": (
        "https://www.google.com/travel/flights?hl=en",
        "Find one-way flights from Zurich to London on October 20, 2026, for one adult in economy. "
        "Stop when matching flight options are visible.",
    ),
}
MAX_STEPS = 12
OUT = Path("artifacts/live")


def run(name: str, url: str, goal: str) -> dict:
    error = None
    started = time.perf_counter()
    with Agent(url, goal) as agent:
        try:
            for _ in range(MAX_STEPS):
                agent.command("tick")
                if agent.state["status"] in {"done", "blocked"}:
                    break
        except Exception as exc:  # noqa: BLE001 - a live run must always leave its record behind
            error = f"{type(exc).__name__}: {exc}"
        state = agent.state
        record = {
            "task": name, "goal": goal, "url": url, "status": state["status"], "error": error,
            "seconds": round(time.perf_counter() - started, 1), "verdicts": state.get("verdicts", []),
            "steps": [
                {"n": i + 1, "operation": h.get("operation"), "action": h.get("action"), "text": h.get("text"),
                 "policy_ms": h.get("latency_ms"), "page_changed": h.get("page_changed"), "url": h.get("url"),
                 "correct": None}
                for i, h in enumerate(state["history"])
            ],
            "decisions": [{"request": d.get("request"), "choice": d.get("choice")} for d in state["decisions"]],
            "success": None,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return record


if __name__ == "__main__":
    for task in sys.argv[1:] or list(TASKS):
        r = run(task, *TASKS[task])
        print(f"{task}: status={r['status']} steps={len(r['steps'])} seconds={r['seconds']} error={r['error']}")
