# Part 3 — Phase A+B diagnosis, benchmarks, evaluation, Gate A+B (Tasks 11–15)

Read `../2026-09-24-planner-actor.md` (Global Constraints, Review Focus) and the spec first. All tasks here need
the throwaway Chrome (`BU_CDP_URL`) and, except Task 11, the mlx-lm server.

---

### Task 11: Diagnose the GitHub Issues click

In the baseline, the actor chose "Issues 146" twice and the URL did not change; the third click navigated.
Two explanations fit: (A) GitHub navigates client-side (Turbo) and our 50 ms post-click settle observes before the
URL changes; (B) the hit-test point lands on an element outside the link. This task measures which, then fixes only
that one.

**Files:**
- Create: `scripts/diagnose_click.py`
- Modify (only if A is confirmed): `jev_ultrafast/browser.py` (`Browser.observe` post-input wait)
- Test (only if A): `tests/test_agent.py` (new test)

**Interfaces:**
- Consumes: `Browser`, `action_space`.
- Produces (only if A): `Browser.observe` waits, after a click on a link whose `href` differs from the current URL
  (ignoring fragments), until `location.href` changes or 3 s pass.

- [ ] **Step 1: Write the diagnosis script**

```python
# scripts/diagnose_click.py
"""Throwaway-Chrome diagnosis: why does clicking GitHub's Issues tab sometimes not navigate?

Usage: uv run --env-file .env python scripts/diagnose_click.py [runs]
"""

import json
import sys
import time

from jev_ultrafast.browser import Browser

REPO = "https://github.com/browser-use/browser-use"
HIT = """(node => {
  const e=window.__jevFast.nodes.get(node); const r=e.getBoundingClientRect();
  const x=r.x+r.width/2, y=r.y+r.height/2, hit=document.elementFromPoint(x,y);
  return {x, y, contains: e.contains(hit), hit: hit ? hit.outerHTML.slice(0,160) : null,
          href: e.href, target: e.outerHTML.slice(0,160)};
})"""


def one_run() -> dict:
    browser = Browser(REPO)
    try:
        page = browser.observe(screenshot=False)
        issues = next(a for a in page["actions"] if a["kind"] == "click" and a["label"].startswith("Issues"))
        hit = browser.evaluate(f"{HIT}({issues['node']})")
        browser.act(issues, page)
        started, timeline = time.perf_counter(), []
        for delay in (0.05, 0.25, 0.5, 1.0, 2.0, 3.0):
            time.sleep(max(0.0, delay - (time.perf_counter() - started)))
            timeline.append((delay, browser.evaluate("location.href")))
        return {"hit_test": hit, "timeline": timeline}
    finally:
        browser.close()


if __name__ == "__main__":
    for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 5):
        print(json.dumps({"run": i + 1, **one_run()}, indent=1))
```

- [ ] **Step 2: Run it five times and classify**

Run: `uv run --env-file .env python scripts/diagnose_click.py 5`

Classify from the output and record the numbers in the task report:
- **A (late navigation):** `hit_test.contains` is `true` and the timeline URL changes to `/issues` only after
  0.05 s in at least one run.
- **B (hit-test miss):** `hit_test.contains` is `false`, or the URL never changes within 3 s while `contains` is
  `true` in no run.
- **Neither:** the URL changes at 0.05 s in all five runs. Then the baseline failure was transient; write that in the
  report, commit the script, and skip Steps 3–6.

If **B**: stop after committing the script (Step 6) and report the `hit` and `target` HTML to the user. Fixing a
covered target needs a design decision (which element to click instead) that this plan does not make.

- [ ] **Step 3 (A only): Write the failing test**

```python
def test_link_click_waits_for_client_side_navigation(monkeypatch):
    from jev_ultrafast import browser as b

    urls = iter(["https://github.com/r", "https://github.com/r", "https://github.com/r/issues"])
    br = b.Browser.__new__(b.Browser)
    br.session = "s"
    br.after_input = {"kind": "click", "node": 1, "href": "https://github.com/r/issues",
                      "page_url": "https://github.com/r"}
    br.call = Mock()
    br.evaluate = Mock(side_effect=lambda expr: next(urls) if expr == "location.href" else None)
    monkeypatch.setattr(b, "browser_operation", Mock(return_value={"url": "https://github.com/r/issues"}))
    monkeypatch.setattr(b.time, "sleep", Mock())
    br.observe(screenshot=False)
    assert br.evaluate.call_count == 3
```

- [ ] **Step 4 (A only): Run it to verify it fails**

Run: `uv run pytest tests/test_agent.py::test_link_click_waits_for_client_side_navigation -v`
Expected: FAIL (`assert 0 == 3` or similar — no URL polling yet).

- [ ] **Step 5 (A only): Implement the wait**

In `Browser.act`, record the page URL with the action:

```python
        self.after_input = {**action, "page_url": page["url"]} if action["kind"] != "wait" else None
```

In `Browser.observe`, at the start of the `if getattr(self, "after_input", None):` block, before the existing
`Runtime.evaluate` call, add:

```python
            if action["kind"] == "click" and _navigates(action):
                deadline = time.monotonic() + NAVIGATION_WAIT_S
                while time.monotonic() < deadline and self.evaluate("location.href") == action["page_url"]:
                    time.sleep(0.05)
```

and module-level:

```python
NAVIGATION_WAIT_S = 3.0


def _navigates(action) -> bool:
    """A link to another document: client-side routers (GitHub Turbo) change the URL after the click returns."""
    href, current = action.get("href") or "", action.get("page_url") or ""
    return href.startswith(("http://", "https://")) and href.split("#")[0] != current.split("#")[0]
```

Note `action` inside `observe` is the local unpacked from `self.after_input` (existing code:
`action, self.after_input = self.after_input, None`).

- [ ] **Step 6: Verify and commit**

Run (A): `uv run pytest -q && uv run ruff check . && uv run --env-file .env python scripts/diagnose_click.py 3`
Expected (A): tests pass; in all 3 runs the first observation after the click (the agent's view) is on `/issues`.

```bash
scripts/commit.sh "fix: wait for client-side navigation after link clicks"   # A
scripts/commit.sh "chore: add click diagnosis script"                         # B or neither
```

---

### Task 12: Planner model benchmark (4B vs 1.7B)

The planner is now on the critical path. Before any live run, measure plan validity, first-step correctness and
latency for Qwen3-4B-Instruct-2507 and Qwen3-1.7B (thinking off) on the suite's real start pages.

**Files:**
- Create: `scripts/capture_pages.py`, `scripts/bench_planner.py`
- Test: `tests/test_bench_planner.py`

**Interfaces:**
- Consumes: `evals.live_tasks.TASKS`; `Browser`; `planner.plan`; `pruning.prune_actions`; `model.action_space`;
  `resolver.resolve`.
- Produces: `artifacts/pages/<task>.json` (captured start pages); `first_step_ok(task: str, step: PlanStep | None,
  element: Mapping | None, elements: Sequence[Mapping]) -> bool`; a printed table per model.

- [ ] **Step 1: Capture the start pages**

```python
# scripts/capture_pages.py
"""Save each live task's start page (as the agent observes it) for offline planner benchmarks."""

import json
import sys
from pathlib import Path

from evals.live_tasks import TASKS
from jev_ultrafast.browser import Browser

OUT = Path("artifacts/pages")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or list(TASKS):
        browser = Browser(TASKS[name].url)
        try:
            page = browser.observe(screenshot=False)
        finally:
            browser.close()
        keep = {k: page[k] for k in ("url", "title", "text", "outline", "actions")}
        (OUT / f"{name}.json").write_text(json.dumps({"task": name, "goal": TASKS[name].goal, "page": keep}))
        print(name, len(page["actions"]), "actions")
```

Run: `uv run --env-file .env python scripts/capture_pages.py`
Expected: 25 lines, each with a non-zero action count.

- [ ] **Step 2: Write the failing tests for the first-step checks**

```python
# tests/test_bench_planner.py
from scripts.bench_planner import first_step_ok
from jev_ultrafast.planner import PlanStep


def el(index, label, **kw):
    return {"index": str(index), "label": label, "operations": ["CLICK"], **kw}


HN = [el(1, "comments", landmark="nav"), el(2, "Story A"), el(3, "48 comments"), el(4, "Story B"),
      el(5, "17 comments"), el(6, "Story C"), el(7, "discuss")]


def test_search_tasks_accept_goto_or_typing_the_name():
    ok = PlanStep("GOTO", "https://en.wikipedia.org/w/index.php?search=Ada+Lovelace&title=Special%3ASearch&go=Go",
                  "", "Go to the search results.")
    assert first_step_ok("wiki_search_ada", ok, None, [])
    typed = PlanStep("TYPE_TEXT", "Search Wikipedia", "Ada Lovelace", "Type it.")
    assert first_step_ok("wiki_search_ada", typed, None, [])
    assert not first_step_ok("wiki_search_ada", PlanStep("CLICK", "View history", "", "x"), None, [])


def test_hn_ordinal_comments_link():
    assert first_step_ok("hn_comments", PlanStep("CLICK", "48 comments", "", "x"), HN[2], HN)
    assert first_step_ok("hn_second_comments", PlanStep("CLICK", "17 comments", "", "x"), HN[4], HN)
    assert first_step_ok("hn_third_comments", PlanStep("CLICK", "discuss", "", "x"), HN[6], HN)
    assert not first_step_ok("hn_second_comments", PlanStep("CLICK", "comments", "", "x"), HN[0], HN)


def test_section_tasks_accept_toc_link_or_scroll_but_not_toggle():
    assert first_step_ok("ada_references", PlanStep("CLICK", "8 References", "", "x"), el(1, "8 References"), [])
    assert first_step_ok("ada_references", PlanStep("SCROLL_TO_TEXT", "References", "", "x"), None, [])
    toggle = el(2, "Toggle References subsection")
    assert not first_step_ok("ada_references", PlanStep("CLICK", toggle["label"], "", "x"), toggle, [])


def test_no_step_is_never_ok():
    assert not first_step_ok("gh_issues", None, None, [])
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_bench_planner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.bench_planner'` (if `scripts` is not importable,
add an empty `scripts/__init__.py` in Step 4).

- [ ] **Step 4: Implement the benchmark**

```python
# scripts/bench_planner.py
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
        return step.operation in {"TYPE_TEXT", "CLICK"} and bool(re.search(r"where (from|to)", normalize(step.target_text)))
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
    rows = [bench_one(p) for p in sorted(PAGES.glob("*.json"))]
    for r in rows:
        print(f"{r['task']:24} valid={r['valid']!s:5} ok={r['ok']!s:5} ms={r['ms']} {r.get('step') or r.get('error')}")
    times = [r["ms"] for r in rows if r["ms"] is not None]
    print(f"[{label}] n={len(rows)} valid={sum(r['valid'] for r in rows)} first_step_ok={sum(r['ok'] for r in rows)} "
          f"median_ms={statistics.median(times) if times else None} "
          f"p90_ms={sorted(times)[int(0.9 * (len(times) - 1))] if times else None}")
    Path(f"artifacts/pages/bench_{label}.json").write_text(json.dumps(rows, indent=1))
```

Create `scripts/__init__.py` (empty) if the test import needs it.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bench_planner.py -v && uv run ruff check scripts tests`
Expected: 4 passed; `All checks passed!`

- [ ] **Step 6: Run the benchmark for both models**

```bash
# 4B (default server)
uv run --env-file .env python scripts/bench_planner.py qwen3-4b
# 1.7B: stop the server, start it with the other model, then
pkill -f mlx_lm.server; uv run mlx_lm.server --model mlx-community/Qwen3-1.7B-4bit --port 8080 &
sleep 20
TEXT_MODEL=mlx-community/Qwen3-1.7B-4bit uv run --env-file .env python scripts/bench_planner.py qwen3-1.7b
```

Expected: two summary lines `[qwen3-4b] n=25 ...` and `[qwen3-1.7b] n=25 ...`.

**Decision rule** (write it and both summary lines into the task report): choose 1.7B only if its `first_step_ok`
is within 1 of 4B's **and** its median latency is ≤ 60% of 4B's; otherwise keep 4B. If the chosen model's median
exceeds 3,000 ms, report that the speed target is at risk before continuing. Restore the server to the chosen
model, and if it is 1.7B, set `TEXT_MODEL` in `.env.example` accordingly.

- [ ] **Step 7: Commit**

```bash
scripts/commit.sh "feat: add planner benchmark on captured start pages"
```

---

### Task 13: Actor replay regression script

Keeps the replay probe as a repeatable check: the six recorded failures, re-asked in the new request shape (one
target question, step as goal). Run after every actor change.

**Files:**
- Create: `scripts/replay_probe.py`
- Test: `tests/test_replay_probe.py`

**Interfaces:**
- Consumes: `actor.actor_request`; `planner.PlanStep`; `candidates.Candidate`; `policy.laya_predict`; recorded
  baseline files in `artifacts/live/20260923T175703Z/` (local, gitignored).
- Produces: `parse_option(text: str) -> tuple[str, str, str]` (label, role, value); `CASES`; a printed pass/fail
  table; exit code 0.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_replay_probe.py
from scripts.replay_probe import parse_option


def test_parse_option_round_trips_render_option():
    assert parse_option("Mary Mallon (link)") == ("Mary Mallon", "link", "")
    assert parse_option("Where from? (combobox, =Chennai)") == ("Where from?", "combobox", "Chennai")
    assert parse_option("Change ticket type. Round trip (combobox, =Round trip)") == (
        "Change ticket type. Round trip", "combobox", "Round trip")
    assert parse_option("plain label") == ("plain label", "", "")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_replay_probe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.replay_probe'`

- [ ] **Step 3: Implement**

```python
# scripts/replay_probe.py
"""Re-ask six recorded live failures in the actor's request shape; report which element the actor picks.

Usage: uv run --env-file .env python scripts/replay_probe.py [run_dir]
"""

import json
import re
import sys
from pathlib import Path

from jev_ultrafast.actor import actor_request
from jev_ultrafast.candidates import Candidate
from jev_ultrafast.planner import PlanStep
from jev_ultrafast.policy import laya_predict

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/live/20260923T175703Z")
_OPTION = re.compile(r"^(.*) \(([^,()]+)(?:, =(.*))?\)$")
# (task, decision index, question id in the recording, step, expected label prefix)
CASES = [
    ("wiki_search_ada", 0, "type_text_target", PlanStep("TYPE_TEXT", "Search Wikipedia", "Ada Lovelace",
     'Type "Ada Lovelace" into the Search Wikipedia box.'), "Search Wikipedia"),
    ("wiki_featured", 0, "click_target", PlanStep("CLICK", "Mary Mallon", "",
     "Click the Mary Mallon link in today's featured article."), "Mary Mallon"),
    ("mallon_to_typhoid", 0, "click_target", PlanStep("CLICK", "Mary Mallon", "",
     "Click the Mary Mallon link in today's featured article."), "Mary Mallon"),
    ("hn_comments", 0, "click_target", PlanStep("CLICK", "48 comments", "",
     "Click the 48 comments link under the top story."), "48 comments"),
    ("ada_references", 0, "click_target", PlanStep("CLICK", "8 References", "",
     "Click the 8 References link in the contents."), "8 References"),
    ("flights", 1, "type_text_target", PlanStep("TYPE_TEXT", "Where to?", "London",
     'Type "London" into the Where to? field.'), "Where to?"),
]


def parse_option(text: str) -> tuple[str, str, str]:
    match = _OPTION.match(text)
    if not match:
        return text, "", ""
    return match.group(1), match.group(2), match.group(3) or ""


def _candidates(record: dict, index: int, question: str) -> list[Candidate]:
    questions = record["decisions"][index]["request"]["questions"]
    criteria = questions.get(question) or questions["click_target"]
    return [Candidate(i, *parse_option(text)[:2], parse_option(text)[2]) for i, text in criteria.items()]


def main() -> None:
    passed = 0
    for task, index, question, step, expected in CASES:
        record = json.loads((RUN / f"{task}.json").read_text())
        candidates = _candidates(record, index, question)
        if len(candidates) == 1:
            label, prob = candidates[0].label, 1.0
        else:
            state, questions = actor_request(step, candidates, [])
            answer = laya_predict(state, questions)["answers"][next(iter(questions))]
            label = next(c.label for c in candidates if c.id == answer["choice"])
            prob = answer["probabilities"][answer["choice"]]
        ok = label.startswith(expected)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} {task:18} picked={label[:40]!r} p={prob:.2f} expected={expected!r}")
    print(f"{passed}/{len(CASES)} probe decisions correct")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test and the probe**

Run: `uv run pytest tests/test_replay_probe.py -v && LAYA_CHECKPOINT=checkpoints/laya_browser_mind2web uv run python scripts/replay_probe.py`
Expected: test passes; probe prints `>= 4/6 probe decisions correct` with the published checkpoint (the manual
probe found 5 of 6 target choices right with the goal replaced by the step; the flights typing case is the known
weak one). Record the line in the task report. A result below 4/6 means the request shape differs from the probe:
compare `actor_request` output with `artifacts/live/.../<task>.json` decisions before continuing.

- [ ] **Step 5: Commit**

```bash
scripts/commit.sh "test: add actor replay probe for recorded live failures"
```

---

### Task 14: Live eval records routes, planner calls and timing

**Files:**
- Modify: `scripts/live_eval.py` (decision/step records, backend label, planner calls)
- Create: `scripts/live_summary.py`
- Test: `tests/test_live_summary.py`

**Interfaces:**
- Consumes: `Agent.pilot.plans`; decision dict keys from Task 9 (`route`, `instruction`, `value`, `tool`,
  `actor_ms`, `latency_ms`); history keys from Task 10.
- Produces: records with `backend`, per-step `route` and `instruction`, per-decision `route`, `instruction`,
  `value`, `tool`, `actor_ms`, and top-level `planner_calls`; `summarize(records: Sequence[dict]) -> dict` with
  `passed`, `n`, `by_category`, `median_decision_ms`, `median_actor_ms`, `median_planner_ms`, `wall_s_per_action`,
  `routes` (counter).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_live_summary.py
from scripts.live_summary import summarize


def rec(task, category, success, steps, seconds, decisions, planner_calls=()):
    return {"task": task, "category": category, "success": success, "steps": [{}] * steps, "seconds": seconds,
            "decisions": decisions, "planner_calls": list(planner_calls)}


def test_summary_counts_categories_routes_and_medians():
    records = [
        rec("a", "site_search", True, 2, 10.0,
            [{"route": "resolver", "latency_ms": 900, "actor_ms": 0},
             {"route": "actor", "latency_ms": 1200, "actor_ms": 800}],
            [{"latency_ms": 2000}, {"latency_ms": 2500}]),
        rec("b", "site_search", False, 4, 20.0, [{"route": "planner_pick", "latency_ms": 3000, "actor_ms": 700}],
            [{"error": "no valid plan"}]),
    ]
    s = summarize(records)
    assert (s["passed"], s["n"]) == (1, 2)
    assert s["by_category"] == {"site_search": [1, 2]}
    assert s["routes"] == {"resolver": 1, "actor": 1, "planner_pick": 1}
    assert s["median_decision_ms"] == 1200 and s["median_actor_ms"] == 750
    assert s["median_planner_ms"] == 2250 and s["wall_s_per_action"] == 5.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_live_summary.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.live_summary'`

- [ ] **Step 3: Implement the summary**

```python
# scripts/live_summary.py
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
```

- [ ] **Step 4: Record routes and planner calls in `live_eval.py`**

In `run()`, set the backend label instead of the fixed string: in the initial `record` dict replace
`"route": "actor_only",` with

```python
              "backend": os.environ.get("POLICY_BACKEND", ""), "planner_calls": [],
```

(add `import os`). In the `record.update(...)` call inside `finally`, add:

```python
                    planner_calls=list(agent.pilot.plans) if agent.pilot else [],
```

extend each step dict with

```python
                         "route": h.get("route"), "instruction": h.get("instruction"),
```

and extend each decision dict with

```python
                                "route": d.get("route"), "instruction": d.get("instruction"),
                                "value": d.get("value"), "tool": d.get("tool"), "actor_ms": d.get("actor_ms"),
```

- [ ] **Step 5: Run the tests and a one-task smoke run**

Run: `uv run pytest -q && uv run ruff check . && POLICY_BACKEND=planner uv run --env-file .env python scripts/live_eval.py gh_issues`
then `uv run python scripts/live_summary.py "$(ls -d artifacts/live/2* | tail -1)"`
Expected: tests pass; the summary JSON shows `n: 1`, a non-empty `routes`, and `median_planner_ms` not null.

- [ ] **Step 6: Commit**

```bash
scripts/commit.sh "feat: record routes, planner calls and timing in live eval"
```

---

### Task 15: Gate A+B — live suite run and report

**Files:**
- Create: `docs/superpowers/reports/2026-09-24-gate-ab.md` (the report; contains no secrets, no page dumps)

- [ ] **Step 1: Prepare**

- Throwaway Chrome running on `BU_CDP_URL`; mlx-lm server running the model chosen in Task 12.
- `uv run pytest -q` passes; `git status --short` is clean.
- Environment for the run: `POLICY_BACKEND=planner`, `ACTOR_TAU=0.5`, `ACTOR_CONTEXT=0`,
  `LAYA_CHECKPOINT=checkpoints/laya_browser_mind2web`.

- [ ] **Step 2: Run the full suite once**

Run: `POLICY_BACKEND=planner ACTOR_TAU=0.5 uv run --env-file .env python scripts/live_eval.py`
Expected: 25 lines. Do not re-run individual tasks to improve the score; one run per task is the protocol.

- [ ] **Step 3: Summarise**

Run: `uv run python scripts/live_summary.py artifacts/live/<run>`

- [ ] **Step 4: Tag every failed task**

For each failed task, read its record and set each wrong step's `failure_tag` (in the JSON file) to one of
`planner | resolver | actor | router | options | actuation | site`, and `correct` to `true/false` for every step,
using the same judgement rule as the baseline (`manual_review`: "Progress judged from goal, action, and observed
URL/page change"). Tag definitions: `planner` — wrong or missing step/operation/value, false done/blocked;
`resolver` — resolver matched the wrong element; `actor` — actor chose wrong with confidence ≥ τ; `router` — the
planner's top-5 pick was wrong; `options` — the needed element was not among candidates; `actuation` — right
element, action had no effect; `site` — the site changed, errored or timed out.

- [ ] **Step 5: Write the report**

`docs/superpowers/reports/2026-09-24-gate-ab.md` with: configuration (backend, planner model, τ, checkpoint,
date); overall pass count with n and the baseline 7/25 beside it; the category table beside the baseline's;
per-task table (result, one-line observation); route counts; median actor ms, median planner ms, median decision
ms, wall s/action versus the targets (1.0 s / 3.0 s / 4.0 s); failure-tag counts; the gate verdict against
**≥ 15/25**; and the note that differences under ~15 points (4 tasks) are not distinguishable.

- [ ] **Step 6: Commit and stop**

```bash
scripts/commit.sh "docs: add Gate A+B live evaluation report"
```

**STOP.** Report the result to the user (pass count, category table, timing versus targets, top failure tags) and
wait for their decision before starting Phase C. If the gate is missed, do not tune anything further.
