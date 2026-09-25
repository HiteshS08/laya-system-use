"""Live evaluation: run tasks on real public pages with the local stack and record every step for hand labelling.

Needs: a throwaway Chrome on BU_CDP_URL, the mlx-lm server from .env.example, and POLICY_BACKEND (program, planner
or laya).
Read-only public sites only. Usage: uv run --env-file .env python scripts/live_eval.py [task ...]
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from evals.live_tasks import TASKS, LiveTask, check_outcome
from jev_ultrafast import Agent

MAX_STEPS = 12
OUT = Path("artifacts/live")


def pilot_fields(pilot) -> dict:
    """What the run cost in model calls. A program-backend pilot also leaves its program and decision trace."""
    if pilot is None:
        return {"planner_calls": [], "llm_calls": 0}
    plans = list(pilot.plans)
    fields = {"planner_calls": plans, "llm_calls": sum(p.get("attempts", 1) for p in plans)}
    if hasattr(pilot, "trace"):
        fields.update(program=plans[0].get("program", "") if plans else "", trace=list(pilot.trace),
                      actor_calls=pilot.actor_calls)
    return fields


def run(name: str, task: LiveTask, out: Path) -> dict:
    error = None
    started = time.perf_counter()
    record = {"task": name, "category": task.category, "goal": task.goal, "url": task.url,
              "checks": {"url_regex": task.url_regex, "text_regex": task.text_regex,
                         "via_url_regex": task.via_url_regex, "min_scroll_y": task.min_scroll_y,
                         "flight_date": task.flight_date, "flight_origin_regex": task.flight_origin_regex,
                         "flight_destination_regex": task.flight_destination_regex},
              "backend": os.environ.get("POLICY_BACKEND", ""), "planner_calls": [], "llm_calls": 0,
              "status": "error", "error": None,
              "seconds": None, "verdicts": [], "steps": [], "decisions": [], "success": False}
    try:
        with Agent(task.url, task.goal) as agent:
            expected_link = None
            try:
                expected_link = agent.browser.evaluate(task.expected_link_script) if task.expected_link_script else None
                if task.expected_link_script and not expected_link:
                    raise RuntimeError(f"Could not identify the expected link for {name} on the starting page")
                record["initial_url"] = agent.state["page"]["url"]
                for _ in range(MAX_STEPS):
                    agent.command("tick")
                    if agent.state["status"] in {"done", "blocked"}:
                        break
            finally:
                state = agent.state
                record.update(
                    status=state["status"], expected_link=expected_link, verdicts=state.get("verdicts", []),
                    final_url=state["page"]["url"], final_title=state["page"]["title"],
                    final_text=state["page"]["text"], final_scroll_y=state["page"]["scroll"]["y"],
                    final_actions=state["page"].get("actions", []),
                    success=check_outcome(
                        task, state["page"]["url"], state["page"]["text"], expected_link,
                        visited_urls=(record.get("initial_url", task.url),
                                      *(h["url"] for h in state["history"]), state["page"]["url"]),
                        scroll_y=state["page"]["scroll"]["y"],
                        final_actions=state["page"].get("actions", []),
                    ),
                    steps=[
                        {"n": i + 1, "operation": h.get("operation"), "action": h.get("action"),
                         "text": h.get("text"), "policy_ms": h.get("latency_ms"),
                         "page_changed": h.get("page_changed"),
                         "url_before": (record.get("initial_url", task.url) if i == 0
                                        else state["history"][i - 1]["url"]),
                         "url_after": h.get("url"),
                         "route": h.get("route"), "instruction": h.get("instruction"),
                         "observe_ms": h.get("observe_ms"), "act_ms": h.get("act_ms"),
                         "correct": None, "failure_tag": None}
                        for i, h in enumerate(state["history"])
                    ],
                    decisions=[{"request": d.get("request"), "answers": d.get("raw_answers"),
                                "choice": d.get("choice"), "operation": d.get("operation"),
                                "target": d.get("target"), "confidence": d.get("confidence"),
                                "target_confidence": d.get("target_confidence"),
                                "latency_ms": d.get("latency_ms"), "usage": d.get("usage"),
                                "route": d.get("route"), "instruction": d.get("instruction"),
                                "value": d.get("value"), "tool": d.get("tool"), "actor_ms": d.get("actor_ms")}
                               for d in state["decisions"]],
                    **pilot_fields(getattr(agent, "pilot", None)),
                )
    except Exception as exc:  # noqa: BLE001 - a live run must always leave its record behind
        error = f"{type(exc).__name__}: {exc}"
    record["error"] = error
    record["seconds"] = round(time.perf_counter() - started, 1)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return record


if __name__ == "__main__":
    out = OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for name in sys.argv[1:] or list(TASKS):
        r = run(name, TASKS[name], out)
        print(f"{name}: status={r['status']} success={r['success']} "
              f"steps={len(r['steps'])} seconds={r['seconds']} error={r['error']}")
