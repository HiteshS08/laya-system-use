"""Offline planner benchmark on captured start pages: validity, first-step correctness, latency.

Restart the mlx-lm server with each model, then:
  TEXT_MODEL=<model> uv run --env-file .env python scripts/bench_planner.py <label>
"""

import json
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from jev_ultrafast.model import action_space
from jev_ultrafast.planner import PlanStep, plan
from jev_ultrafast.pruning import prune_actions
from jev_ultrafast.resolver import normalize, resolve

PAGES = Path("artifacts/pages")
COMMENTS = re.compile(r"^(\d+\s+comments?|discuss)$")
SEARCH_TERMS = {"wiki_search": "incompleteness", "wiki_search_turing": "turing", "wiki_search_ada": "lovelace",
                "wiki_search_python": "python", "wiki_search_mallon": "mallon", "turing_to_award": "turing",
                "ada_to_babbage": "lovelace", "python_to_guido": "python", "mallon_to_typhoid": "mallon"}
ORDINAL = {"hn_comments": 0, "hn_second_comments": 1, "hn_third_comments": 2}


def _search_ok(task: str, step: PlanStep) -> bool:
    term = SEARCH_TERMS[task]
    if step.operation == "GOTO":
        return term in step.target_text.casefold()
    if task == "mallon_to_typhoid" and step.operation == "CLICK":
        return "mallon" in normalize(step.target_text)
    return step.operation == "TYPE_TEXT" and term in step.value.casefold()


def _section_ok(step: PlanStep, element: Mapping | None, heading: str) -> bool:
    if step.operation == "SCROLL_TO_TEXT":
        return heading in normalize(step.target_text)
    label = normalize(element["label"]) if element else ""
    return step.operation == "CLICK" and heading in label and not label.startswith("toggle")


def first_step_ok(task: str, step: PlanStep | None, element: Mapping | None, elements: Sequence[Mapping]) -> bool:
    if step is None:
        return False
    if task in SEARCH_TERMS:
        return _search_ok(task, step)
    if task in ORDINAL:
        links = [e for e in elements if COMMENTS.match(normalize(e["label"])) and e.get("landmark") != "nav"]
        return element is not None and len(links) > ORDINAL[task] and element is links[ORDINAL[task]]
    if task.endswith("_references"):
        return _section_ok(step, element, "references")
    if task.endswith("_external_links"):
        return _section_ok(step, element, "external links")
    label = normalize(element["label"]) if element else ""
    expected = {"wiki_featured": lambda: "featured" in normalize(element.get("section", "")) if element else False,
                "wiki_long_page": lambda: "godel" in label,
                "gh_issues": lambda: label.startswith("issues"),
                }
    if task in expected:
        return step.operation == "CLICK" and expected[task]()
    if task.startswith("flights"):
        return step.operation in {"TYPE_TEXT", "CLICK"} and bool(
            re.search(r"where (from|to)", normalize(step.target_text)))
    raise KeyError(f"No first-step check for task {task!r}")


def bench_one(path: Path) -> dict:
    record = json.loads(path.read_text())
    page = record["page"]
    elements, _, _ = action_space(prune_actions(page["actions"], record["goal"]))
    try:
        result = plan(record["goal"], page, elements, [], [])
    except ValueError as exc:
        return {"task": record["task"], "valid": False, "ok": False, "ms": None, "error": str(exc)}
    step = result.steps[0] if result.steps else None
    pool = [e for e in elements if step and step.operation in e["operations"]]
    element = resolve(step.target_text, pool)[0] if step else None
    return {"task": record["task"], "valid": True, "ok": bool(first_step_ok(record["task"], step, element, elements)),
            "ms": result.latency_ms, "chars": result.request_chars, "step": step.__dict__ if step else None}


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "model"
    # Exclude this script's own bench_*.json output: a prior run's summary otherwise reappears here as a
    # "captured page" on the next run (same directory, same *.json glob) and crashes bench_one on record["page"].
    rows = [bench_one(p) for p in sorted(PAGES.glob("*.json")) if not p.name.startswith("bench_")]
    for r in rows:
        print(f"{r['task']:24} valid={r['valid']!s:5} ok={r['ok']!s:5} ms={r['ms']} {r.get('step') or r.get('error')}")
    times = [r["ms"] for r in rows if r["ms"] is not None]
    print(f"[{label}] n={len(rows)} valid={sum(r['valid'] for r in rows)} first_step_ok={sum(r['ok'] for r in rows)} "
          f"median_ms={statistics.median(times) if times else None} "
          f"p90_ms={sorted(times)[int(0.9 * (len(times) - 1))] if times else None}")
    Path(f"artifacts/pages/bench_{label}.json").write_text(json.dumps(rows, indent=1))
