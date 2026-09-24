# Part 4 — Phase C: retrained actor (Tasks 16–22)

Starts only after the user has seen the Gate A+B report (Task 15) and said to continue.
Read `../2026-09-24-planner-actor.md` (Global Constraints) and the spec (§6) first.

Data paths used below: Mind2Web train shards `data/mind2web/data/train/train_*.json`; test splits
`data/mind2web/test/test_{task,website,domain}/*.json` (evaluation only). Mind2Web's `cleaned_html` keeps landmark
tags, headings and `li`/`tr` rows (checked: `<nav>` in 200/299 sampled steps, `<h2>` 207/299, `<li>` 299/299) but
not `href`, so link targets are shown to the planner only, never to the actor.

---

### Task 16: Context fields in the Mind2Web pipeline with train/serve parity

**Files:**
- Modify: `training/mind2web.py` (`sections_by_id`, `landmark_of`, `row_text_of`, `context_for`; `to_candidate`,
  `reclaim_gold`, `parse_step` pass sections)
- Test: `tests/test_mind2web.py` (new tests), `tests/test_snapshot_live.py` (live parity test)

**Interfaces:**
- Consumes: `formatter.context_text` (Task 7); the rules in Task 2's Interfaces block.
- Produces: `Candidate.context` filled for Mind2Web candidates; `sections_by_id(index: dict) -> dict[str, str]`;
  `context_for(node, section: str) -> str`; `to_candidate(raw, index, sections=None)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mind2web.py`:

```python
from pathlib import Path

from lxml import etree

from training.mind2web import context_for, node_index, sections_by_id

FIXTURE = Path(__file__).parent / "fixtures" / "context_page.html"


def _indexed_fixture():
    root = etree.HTML(FIXTURE.read_text())
    for i, node in enumerate(root.iter(tag=etree.Element)):
        node.set("backend_node_id", str(i))
    index = node_index(etree.tostring(root, encoding="unicode"))
    return index, sections_by_id(index)


def _context(label):
    index, sections = _indexed_fixture()
    node = next(n for n in index.values() if n.tag == "a" and "".join(n.itertext()).strip() == label)
    return context_for(node, sections[node.get("backend_node_id")])


def test_context_matches_the_live_snapshot_rules():
    assert _context("Home") == "header"
    assert _context("comments") == "nav"
    assert _context("Mary Mallon") == "main > From today's featured article"
    assert _context("48 comments") == "main > From today's featured article · row: Story one48 comments"
    assert _context("Official archive") == "main > External links · row: Archive Official archive"
    assert _context("About") == "footer"


def test_role_landmark_wins_over_default():
    html = '<div role="navigation"><a backend_node_id="1">Next</a></div>'
    index = node_index(html)
    assert context_for(index["1"], "") == "nav"
```

Add to `tests/test_snapshot_live.py`:

```python
def test_live_context_equals_training_context(observed):
    from jev_ultrafast.formatter import context_text

    _, page = observed
    for label in ("Home", "comments", "Mary Mallon", "48 comments", "Official archive", "About"):
        a = by_label(page, label)
        live = context_text(a["landmark"], a["section"], a["row_text"])
        assert live == _training_context(label), label


def _training_context(label):
    from test_mind2web import _context

    return _context(label)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --env-file .env pytest tests/test_mind2web.py tests/test_snapshot_live.py -v`
Expected: FAIL with `ImportError: cannot import name 'context_for'`

- [ ] **Step 3: Implement**

In `training/mind2web.py`, add the import `from jev_ultrafast.formatter import context_text` and, after
`MAX_LABEL`:

```python
# Mirrors jev_ultrafast/snapshot.js exactly (landmark, section, row_text); parity is tested.
LANDMARK_ROLES = {"navigation": "nav", "banner": "header", "contentinfo": "footer", "complementary": "aside",
                  "main": "main", "form": "form", "search": "form", "dialog": "dialog"}
LANDMARK_TAGS = {"nav": "nav", "header": "header", "footer": "footer", "aside": "aside", "main": "main",
                 "form": "form", "dialog": "dialog"}
HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
UNSECTIONED = frozenset({"nav", "header", "footer"})
CONTEXT_CHARS = 80


def _squash(text: str) -> str:
    return " ".join(text.split())[:CONTEXT_CHARS]


def landmark_of(node) -> str:
    for ancestor in node.iterancestors():
        role = LANDMARK_ROLES.get(ancestor.get("role") or "")
        if role:
            return role
        tag = LANDMARK_TAGS.get(ancestor.tag) if isinstance(ancestor.tag, str) else None
        if tag:
            return tag
    return "main"


def row_text_of(node) -> str:
    row = next((a for a in node.iterancestors() if a.tag in ("li", "tr")), None)
    return _squash("".join(row.itertext())) if row is not None else ""


def sections_by_id(index: dict) -> dict[str, str]:
    """backend_node_id -> text of the last heading at or before it in document order."""
    heading, sections = "", {}
    for bid, node in index.items():
        if node.tag in HEADING_TAGS:
            heading = _squash("".join(node.itertext()))
        sections[bid] = heading
    return sections


def context_for(node, section: str) -> str:
    mark = landmark_of(node)
    return context_text(mark, "" if mark in UNSECTIONED else section, row_text_of(node))
```

Change `to_candidate` to accept and use sections:

```python
def to_candidate(raw: dict, index: dict, sections: dict | None = None) -> tuple[Candidate, bool]:
    attrs = {k: str(v) for k, v in json.loads(raw["attributes"]).items()}
    tag, bid = raw["tag"], raw["backend_node_id"]
    role = role_for(tag, attrs)
    ops = ops_for(tag, role)
    value = attrs.get("input_value", "")[:MAX_LABEL] if "TYPE_TEXT" in ops else ""
    # The live snapshot names an unnamed element by its role (`name(e) || rname`), so training must as well.
    label = label_for(index.get(bid), attrs) or role
    node = index.get(bid)
    context = context_for(node, (sections or {}).get(bid, "")) if node is not None and sections is not None else ""
    return Candidate(bid, label, role, value, ops, context), is_interactive(tag, attrs)
```

In `reclaim_gold`, add a `sections: dict | None = None` parameter and pass it: `return to_candidate(raw, index,
sections)[0]`. In `parse_step`, after `order = ...`:

```python
    sections = sections_by_id(index)
```

and pass `sections` to every `to_candidate(raw, index, sections)` call and to `reclaim_gold(..., sections)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --env-file .env pytest tests/test_mind2web.py tests/test_snapshot_live.py -v && uv run pytest -q`
Expected: all pass; the live parity test runs (not skipped).

- [ ] **Step 5: Coverage check on real data**

```bash
uv run python - <<'EOF'
import json, glob, collections
from training.mind2web import parse_step
steps = [a for t in json.load(open(sorted(glob.glob("data/mind2web/data/train/train_*.json"))[0]))[:40]
         for a in t["actions"]]
c = collections.Counter()
for s in steps:
    p = parse_step(s)
    if p.gold:
        c["gold"] += 1
        c["gold_has_section"] += " > " in p.gold.context
        c["gold_landmark_" + p.gold.context.split(" ")[0].split(">")[0]] += 1
print(len(steps), dict(c))
EOF
```

Expected: a dict where `gold_has_section` is more than a third of `gold`. Record the line in the task report. If it
is under 10%, headings in `cleaned_html` lack `backend_node_id`: stop and report (the section rule would then need
the full parse, which is a design change).

- [ ] **Step 6: Commit**

```bash
scripts/commit.sh "feat: derive live-identical element context from Mind2Web DOM"
```

---

### Task 17: Step conditioning and hard negatives

**Files:**
- Modify: `jev_ultrafast/formatter.py` (`build_request(..., with_context=False)`)
- Modify: `training/mind2web.py` (`step_text`, `is_hard`, `_row`, `task_rows`)
- Modify: `training/build_cases.py` (`--step-rate`, `--context`, `--seed`)
- Modify: `training/prepare_items.py` (duplicate items from hard rows)
- Test: `tests/test_mind2web.py`, `tests/test_build_cases.py`, `tests/test_prepare_items.py`

**Interfaces:**
- Consumes: `resolver.normalize`; `Candidate.context`.
- Produces: `step_text(op, label, value, role, rng) -> str`; `is_hard(gold: Candidate, shortlisted:
  Sequence[Candidate]) -> bool`; `task_rows(task, k=DEFAULT_K, *, step_rate=0.0, context=False, rng=None)`;
  rows gain `mode` (`"goal"|"step"`) and `hard` (bool); `build(..., step_rate=0.0, context=False, seed=0)`;
  `HARD_REPEAT = 1` in `prepare_items` (hard rows appear twice).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mind2web.py` (it already imports `make_step`, `make_task` from `m2w_fixtures` and
`training.mind2web as m2w`):

```python
import random

from jev_ultrafast.candidates import Candidate
from training.mind2web import is_hard, step_text


def test_step_text_uses_templates_and_may_drop_one_label_word():
    rng = random.Random(0)
    texts = {step_text("TYPE_TEXT", "Where to", "London", "combobox", rng) for _ in range(40)}
    assert all("London" in t for t in texts)
    assert any("Where to" in t for t in texts) and len(texts) >= 3
    click = step_text("CLICK", "Issues", "", "link", random.Random(1))
    assert "Issues" in click


def test_hard_rows_have_a_confusable_distractor():
    gold = Candidate("1", "48 comments", "link", context="main > Stories · row: Story one48 comments")
    assert is_hard(gold, [gold, Candidate("2", "48 comments", "link")])
    assert is_hard(gold, [gold, Candidate("3", "comments", "link", context="nav")])
    same_section = Candidate("4", "hide", "link", context="main > Stories")
    assert is_hard(Candidate("1", "Story one", "link", context="main > Stories"), [same_section])
    assert not is_hard(gold, [gold, Candidate("5", "Donate", "link", context="header")])
```

Add a `task_rows` step-mode test next to the existing `task_rows` tests:

```python
def test_step_mode_replaces_the_goal_and_keeps_the_gold_shortlisted():
    task = make_task(make_step("CLICK", "10"), make_step("TYPE", "20", "abc"))
    rows = list(m2w.task_rows(task, step_rate=1.0, context=True, rng=random.Random(0)))
    valid = [r for r in rows if r["valid_gold"]]
    assert valid and all(r["mode"] == "step" for r in valid)
    assert all(r["state"]["goal"] != task["confirmed_task"] for r in valid)
    assert all(r["gold_in_shortlist"] for r in valid)
    goal_rows = list(m2w.task_rows(task))
    assert all(r["mode"] == "goal" and r["state"]["goal"] == task["confirmed_task"] for r in goal_rows)
```

Add to `tests/test_prepare_items.py`:

```python
def test_hard_rows_are_repeated(monkeypatch):
    from training import prepare_items as pi

    monkeypatch.setattr(pi, "build_item", lambda tok, cfg, state, q, g: {"ids": [1], "markers": [0]})
    row = {"drop_reason": None, "gold": {"click_target": {}}, "questions": {"click_target": {}}, "state": {}}
    items, stats = pi.build_items(None, {}, [row, {**row, "hard": True}])
    assert len(items) == 1 + (1 + pi.HARD_REPEAT) and stats["hard_rows"] == 1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_mind2web.py tests/test_prepare_items.py -v`
Expected: FAIL with `ImportError: cannot import name 'is_hard'` and the prepare_items assertion.

- [ ] **Step 3: `build_request` takes `with_context`**

In `jev_ultrafast/formatter.py`:

```python
def build_request(
    goal: str, history: Sequence[str], candidates_by_op: Mapping[str, Sequence[Candidate]], *,
    with_context: bool = False,
) -> tuple[dict, dict]:
```

and in its target-question loop use `render_option(c, with_context=with_context)`.

- [ ] **Step 4: Step templates, hard rows, and `_row`/`task_rows`**

In `training/mind2web.py`, add `import random` and `from jev_ultrafast.resolver import normalize`, then:

```python
STEP_TEMPLATES = {
    "CLICK": ('Click "{label}".', "Click the {label} {role}.", 'Open "{label}".'),
    "TYPE_TEXT": ('Type "{value}" into "{label}".', 'Enter "{value}" in the {label} field.',
                  'Fill {label} with "{value}".'),
    "SELECT": ('Select "{value}" in "{label}".', 'Choose "{value}" from the {label} dropdown.',
               'Set {label} to "{value}".'),
}
DROP_WORD_P = 0.3


def step_text(op: str, label: str, value: str, role: str, rng: random.Random) -> str:
    """A planner-like instruction for the gold action; one word may be missing so exact matching is not enough."""
    words = label.split()
    if len(words) > 1 and rng.random() < DROP_WORD_P:
        cut = rng.randrange(len(words))
        words = words[:cut] + words[cut + 1:]
    return rng.choice(STEP_TEMPLATES[op]).format(label=" ".join(words)[:70], value=value[:40], role=role or "element")


def _section(context: str) -> str:
    return context.split(" · row: ")[0].partition(" > ")[2]


def is_hard(gold: Candidate, shortlisted: Sequence[Candidate]) -> bool:
    gold_label, gold_words = normalize(gold.label), set(normalize(gold.label).split())
    for c in shortlisted:
        if c.id == gold.id:
            continue
        if normalize(c.label) == gold_label:
            return True
        if _section(c.context) and _section(c.context) == _section(gold.context):
            return True
        if c.context.split(" ")[0] in {"nav", "header"} and gold_words & set(normalize(c.label).split()):
            return True
    return False
```

Replace `_row` and `task_rows`:

```python
def _row(task: dict, index: int, parsed: ParsedStep, history: Sequence[str], k: int, *,
         step_rate: float = 0.0, context: bool = False, rng: random.Random | None = None) -> dict:
    goal = task["confirmed_task"]
    reason = _drop_reason(parsed)
    valid = reason is None
    query, mode = goal, "goal"
    if valid and rng is not None and rng.random() < step_rate:
        query, mode = step_text(parsed.op, parsed.gold.label, parsed.value, parsed.gold.role, rng), "step"
    per_op = {op: [c for c in parsed.pool if op in c.ops] for op in OPERATIONS}
    # The live actor's shortlist query is the step alone (jev_ultrafast/actor.py), so step rows match it.
    rank_history = [] if mode == "step" else history
    by_op = {op: shortlist(query, rank_history, cands, k) for op, cands in per_op.items()}
    state, questions = build_request(query, history, by_op, with_context=context)
    gold_id = parsed.gold.id if valid else None
    in_shortlist = valid and gold_id in {c.id for c in by_op[parsed.op]}
    if valid and not in_shortlist:
        reason = "gold_not_shortlisted"
    gold = _gold(parsed.op, gold_id, questions) if reason is None else {}
    if reason is None and not gold:
        reason = "trivial"
    return {
        "task_id": task["annotation_id"], "website": task["website"], "domain": task["domain"], "goal": goal,
        "step": index, "state": state, "questions": questions, "gold": gold, "gold_op": parsed.op,
        "gold_id": gold_id, "valid_gold": valid, "gold_in_shortlist": in_shortlist, "drop_reason": reason,
        "ops_available": [op for op in OPERATIONS if per_op[op]],
        "sole": {op: cands[0].id for op, cands in by_op.items() if len(cands) == 1},
        "top1_by_op": {op: rank_candidates(query, rank_history, cands)[0].id
                       for op, cands in per_op.items() if cands},
        "pool_size": len(parsed.pool), "mode": mode,
        "hard": bool(in_shortlist and is_hard(parsed.gold, by_op[parsed.op])),
    }


def task_rows(task: dict, k: int = DEFAULT_K, *, step_rate: float = 0.0, context: bool = False,
              rng: random.Random | None = None) -> Iterator[dict]:
    history: list[str] = []
    for i, step in enumerate(task["actions"]):
        parsed = parse_step(step)
        yield _row(task, i, parsed, history, k, step_rate=step_rate, context=context, rng=rng)
        history.append(render_history_item(parsed.op, parsed.gold.label if parsed.gold else "", parsed.value))
```

- [ ] **Step 5: CLI flags in `build_cases.py`**

Change `build` to take and pass the new options:

```python
def build(paths: Sequence[Path], out_dir: Path, name: str, k: int, dev_mod: int, limit_tasks: int, *,
          step_rate: float = 0.0, context: bool = False, seed: int = 0) -> dict:
    rng = random.Random(seed)
```

and inside the loop: `task_rows(task, k, step_rate=step_rate, context=context, rng=rng)`. Add to `main`:

```python
    parser.add_argument("--step-rate", type=float, default=0.0, help="share of valid steps whose goal is a step")
    parser.add_argument("--context", action="store_true", help="render element context in options")
    parser.add_argument("--seed", type=int, default=0)
```

and pass `step_rate=args.step_rate, context=args.context, seed=args.seed` to `build`. Add `import random`.
Add to `tests/test_build_cases.py` (same shard setup as `test_build_writes_jsonl_and_summary`):

```python
def test_build_step_mode_marks_every_valid_row(tmp_path):
    shard = tmp_path / "train_0.json"
    shard.write_text(json.dumps([make_task(make_step("CLICK", "10"), make_step("TYPE", "20", "abc"))]))
    bc.build([shard], tmp_path / "out", "train", k=20, dev_mod=0, limit_tasks=0, step_rate=1.0, context=True, seed=0)
    rows = [json.loads(line) for line in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    assert all(r["mode"] == "step" for r in rows if r["valid_gold"])
```

(Use the imports already at the top of that file; add `import json` and
`from m2w_fixtures import make_step, make_task` if missing.)

- [ ] **Step 6: Oversample hard rows in `prepare_items.py`**

```python
HARD_REPEAT = 1  # hard rows appear 1 + HARD_REPEAT = 2 times (spec §6: oversampled ×2)


def build_items(tok, cfg: dict, rows: Sequence[dict]) -> tuple[list[dict], Counter]:
    items: list[dict] = []
    stats: Counter = Counter()
    for row in rows:
        if row["drop_reason"] is not None:
            stats["rows_skipped"] += 1
            continue
        copies = 1 + (HARD_REPEAT if row.get("hard") else 0)
        stats["hard_rows"] += bool(row.get("hard"))
        for qid, gold_q in row["gold"].items():
            item = build_item(tok, cfg, row["state"], row["questions"][qid], gold_q)
            if item is None:
                stats["items_dropped_overflow"] += 1
            else:
                items.extend([item] * copies)
    stats["items"] = len(items)
    return items, stats
```


- [ ] **Step 7: Run everything**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
scripts/commit.sh "feat: add step-conditioned rows and hard-negative oversampling"
```

---

### Task 18: Step-mode evaluation and τ fitting

**Files:**
- Modify: `training/evaluate.py` (`laya_predictor` reports target confidence; `evaluate_rows(..., picks=None)`;
  `--dump-picks`)
- Create: `training/fit_tau.py`
- Test: `tests/test_evaluate.py`, `tests/test_fit_tau.py`

**Interfaces:**
- Produces: predictor dict gains `confidence: {op: float}`; `evaluate_rows(rows, predict, picks: list | None =
  None)` appends `{"confidence", "correct", "gold_op"}` per scored row with a confidence for the gold op;
  `fit_tau(picks, target=0.90) -> dict` with `tau`, `acceptance`, `accepted_accuracy`, `n`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fit_tau.py
from training.fit_tau import fit_tau


def test_lowest_threshold_meeting_the_accuracy_target():
    picks = [{"confidence": c, "correct": ok} for c, ok in
             [(0.95, True), (0.9, True), (0.8, True), (0.7, False), (0.6, True), (0.3, False)]]
    result = fit_tau(picks, target=0.75)
    assert result == {"tau": 0.6, "acceptance": 5 / 6, "accepted_accuracy": 0.8, "n": 6}


def test_unreachable_target_accepts_nothing():
    result = fit_tau([{"confidence": 0.9, "correct": False}], target=0.9)
    assert result["tau"] > 1.0 and result["acceptance"] == 0.0
```

Add to `tests/test_evaluate.py`:

```python
def test_picks_record_confidence_and_correctness():
    row = {"task_id": "t", "step": 0, "valid_gold": True, "gold_in_shortlist": True, "gold_op": "CLICK",
           "gold_id": "2", "drop_reason": None}
    picks = []
    evaluate_rows([row], lambda r: {"operation": "CLICK", "targets": {"CLICK": "2"},
                                    "confidence": {"CLICK": 0.7}}, picks=picks)
    assert picks == [{"confidence": 0.7, "correct": True, "gold_op": "CLICK"}]
```

(Adjust the row dict to whatever minimal fields the existing `evaluate_rows` tests use for `_forced_outcome`.)

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_fit_tau.py tests/test_evaluate.py -v`
Expected: FAIL (`ModuleNotFoundError: training.fit_tau`; `unexpected keyword argument 'picks'`).

- [ ] **Step 3: Implement**

```python
# training/fit_tau.py
"""Fit the router threshold τ on dev picks: the lowest confidence at which accepted picks are ≥ target accurate.

Usage: uv run python -m training.fit_tau training/out/picks_<actor>_dev_step.jsonl [--target 0.9]
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

UNREACHABLE = 1.01


def fit_tau(picks: Sequence[dict], target: float = 0.90) -> dict:
    for tau in sorted({p["confidence"] for p in picks}):
        accepted = [p for p in picks if p["confidence"] >= tau]
        accuracy = sum(p["correct"] for p in accepted) / len(accepted)
        if accuracy >= target:
            return {"tau": tau, "acceptance": len(accepted) / len(picks), "accepted_accuracy": accuracy,
                    "n": len(picks)}
    return {"tau": UNREACHABLE, "acceptance": 0.0, "accepted_accuracy": None, "n": len(picks)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("picks", type=Path)
    parser.add_argument("--target", type=float, default=0.90)
    args = parser.parse_args(argv)
    picks = [json.loads(line) for line in args.picks.read_text().splitlines() if line.strip()]
    print(json.dumps(fit_tau(picks, args.target)))


if __name__ == "__main__":
    main()
```

In `training/evaluate.py`:
- `evaluate_rows(rows, predict, picks: list | None = None)`; after computing `element_ok` for a scored row:

```python
        confidence = pred.get("confidence", {}).get(row["gold_op"])
        if picks is not None and confidence is not None:
            picks.append({"confidence": confidence, "correct": element_ok, "gold_op": row["gold_op"]})
```

- in `laya_predictor`, collect confidences:

```python
        confidence = {}
        for op in ops:
            head = answers.get(f"{op.lower()}_target")
            if head:
                targets[op] = head["choice"]
                confidence[op] = head["probabilities"][head["choice"]]
        return {"operation": operation, "targets": targets, "confidence": confidence}
```

- CLI: `parser.add_argument("--dump-picks", type=Path)`; when given, pass a list to `evaluate_rows` and write it
  as JSONL to that path after evaluation.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
scripts/commit.sh "feat: report target confidence in offline eval and fit router threshold"
```

---

### Task 19: Kev — licence, checkout, export, smoke training

Kev runs from its own checkout and virtualenv (`~/kev`), not as a dependency of this repo, so its Torch/MLX stack
cannot conflict with ours. This repo only exports data for it and talks to its local server (Task 20).

**Files:**
- Create: `training/kev_export.py`
- Test: `tests/test_kev_export.py`
- Modify: `NOTICE.md` (Kev and its base model licences)

**Interfaces:**
- Produces: `kev_state(state: Mapping) -> str`; `to_kev_rows(row: Mapping) -> list[dict]` (one example per gold
  question: `{"state": str, "questions": {qid: question}, "label": {qid: choice}}`; hard rows repeated like
  `prepare_items`); CLI `python -m training.kev_export --cases in.jsonl --out out.jsonl`.

- [ ] **Step 1: Check licences before downloading anything**

```bash
for m in jaredpalmer/kev-0.8b Qwen/Qwen3.5-0.8B-Base; do
  curl -s "https://huggingface.co/api/models/$m" | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print('$m', d.get('cardData',{}).get('license'), d.get('gated'))"
done
curl -s https://api.github.com/repos/jaredpalmer/kev | python3 -c "import sys,json; print(json.load(sys.stdin)['license'])"
```

Expected: Apache-2.0 (or MIT) for all three and `gated` False. If any is missing, gated, or non-permissive,
**stop and report to the user**; skip Tasks 19–20 and run Task 21 for Laya only. Otherwise add to `NOTICE.md`:

```
- Kev (https://github.com/jaredpalmer/kev), Apache-2.0: decision-model architecture and training code, used from a
  separate checkout. Base weights Qwen/Qwen3.5-0.8B-Base (<licence printed above>).
```

- [ ] **Step 2: Check out Kev and read its data format**

```bash
git clone https://github.com/jaredpalmer/kev.git ~/kev && cd ~/kev && uv sync --extra serve
uv run python -m kev.train --help
grep -rn "\"label\"\|'label'\|\[\"questions\"\]\|\[\"state\"\]" kev/ | head -30
```

Write down in the task report: the exact CLI flags for data path, init checkpoint, step limit, output directory and
precision; and the JSONL field names and label shape. The code below assumes the README's format (`state`,
`questions`, `label` mapping question id → choice). If the loader differs, change only `to_kev_rows` and its test to
match, and say so in the report.

- [ ] **Step 3: Write the failing test**

```python
# tests/test_kev_export.py
from training.kev_export import kev_state, to_kev_rows
from training.prepare_items import HARD_REPEAT

ROW = {"drop_reason": None, "hard": False,
       "state": {"goal": 'Click "Issues".', "recent_actions": ["CLICK Code"]},
       "questions": {"operation": {"type": "choice", "criteria": {"CLICK": "c", "TYPE_TEXT": "t"}},
                     "click_target": {"type": "choice", "instructions": "Which element?",
                                      "criteria": {"1": "Code (link)", "2": "Issues 146 (link)"}}},
       "gold": {"operation": {"probabilities": {"CLICK": 1.0, "TYPE_TEXT": 0.0}},
                "click_target": {"probabilities": {"1": 0.0, "2": 1.0}}}}


def test_state_is_text_with_goal_and_recent_actions():
    assert kev_state(ROW["state"]) == 'Goal: Click "Issues".\nRecent actions: CLICK Code'
    assert kev_state({"goal": "g", "recent_actions": []}) == "Goal: g"


def test_one_example_per_gold_question_with_the_argmax_label():
    rows = to_kev_rows(ROW)
    assert [list(r["questions"]) for r in rows] == [["operation"], ["click_target"]]
    assert rows[1]["label"] == {"click_target": "2"} and rows[1]["state"].startswith("Goal:")


def test_dropped_rows_export_nothing_and_hard_rows_repeat():
    assert to_kev_rows({**ROW, "drop_reason": "trivial"}) == []
    assert len(to_kev_rows({**ROW, "hard": True})) == 2 * (1 + HARD_REPEAT)
```

- [ ] **Step 4: Run it to verify it fails**

Run: `uv run pytest tests/test_kev_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.kev_export'`

- [ ] **Step 5: Implement**

```python
# training/kev_export.py
"""Export Mind2Web case rows to Kev's JSONL training format (one question per example).

Usage: uv run python -m training.kev_export --cases training/out/train_v2.jsonl --out training/out/kev_train.jsonl
"""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from training.prepare_items import HARD_REPEAT


def kev_state(state: Mapping) -> str:
    """Kev reads text state; serving (jev_ultrafast/kev_backend.py) uses this same rendering."""
    lines = [f"Goal: {state['goal']}"]
    if state.get("recent_actions"):
        lines.append("Recent actions: " + "; ".join(state["recent_actions"]))
    return "\n".join(lines)


def to_kev_rows(row: Mapping) -> list[dict]:
    if row["drop_reason"] is not None:
        return []
    examples = []
    for qid, gold in row["gold"].items():
        probabilities = gold["probabilities"]
        label = max(probabilities, key=probabilities.get)
        examples.append({"state": kev_state(row["state"]), "questions": {qid: row["questions"][qid]},
                         "label": {qid: label}})
    return examples * (1 + (HARD_REPEAT if row.get("hard") else 0))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    count = 0
    with args.out.open("w") as out:
        for line in args.cases.read_text().splitlines():
            for example in to_kev_rows(json.loads(line)):
                out.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
    print(f"{count} Kev examples -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the tests, then a local 2-step dry run**

```bash
uv run pytest tests/test_kev_export.py -v
uv run python -m training.kev_export --cases training/out/smoke.jsonl --out training/out/kev_smoke.jsonl
cd ~/kev && uv run python -m kev.train --data ~/laya-browser/training/out/kev_smoke.jsonl \
  --init_from jaredpalmer/kev-0.8b <step-limit flag from Step 2> 2 <output flag from Step 2> /tmp/kev_dry
```

Expected: tests pass; the dry run loads the data and completes 2 steps without a data-format error. A format
error means `to_kev_rows` does not match Kev's loader: fix it (and the test), rerun.

- [ ] **Step 7: Commit**

```bash
scripts/commit.sh "feat: export Mind2Web cases to Kev's training format"
```

---

### Task 20: Kev serving backend

**Files:**
- Create: `jev_ultrafast/kev_backend.py`
- Modify: `jev_ultrafast/actor.py` (`default_predict()`), `jev_ultrafast/pilot.py` (`predict=None` default),
  `training/evaluate.py` (`--predictor kev`), `.env.example` (`ACTOR_BACKEND`, `KEV_BASE_URL`, `KEV_MODEL`)
- Test: `tests/test_kev_backend.py`

**Interfaces:**
- Consumes: `training.kev_export.kev_state`; `model.post_json(url, key, body, *, timeout)`.
- Produces: `kev_predict(state: dict, questions: dict) -> dict` (Kev's System One response, containing
  `answers`); `actor.default_predict() -> Predict` (`kev_predict` when `ACTOR_BACKEND=kev`, else
  `policy.laya_predict`); `Pilot(goal, predict=None, ...)` resolves `default_predict()` at construction.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_kev_backend.py
from unittest.mock import Mock

from jev_ultrafast import actor, kev_backend, model, policy


def test_kev_predict_posts_text_state_to_the_local_server(monkeypatch):
    post = Mock(return_value={"answers": {"click_target": {"choice": "2", "confidence": 0.8,
                                                           "probabilities": {"1": 0.2, "2": 0.8}}}})
    monkeypatch.setattr(model, "post_json", post)
    monkeypatch.setenv("KEV_BASE_URL", "http://127.0.0.1:8090")
    out = kev_backend.kev_predict({"goal": "Click Issues.", "recent_actions": []},
                                  {"click_target": {"type": "choice", "criteria": {"1": "a", "2": "b"}}})
    url, _key, body = post.call_args.args
    assert url == "http://127.0.0.1:8090/v1/systemone" and body["state"] == "Goal: Click Issues."
    assert out["answers"]["click_target"]["choice"] == "2"


def test_backend_switch(monkeypatch):
    monkeypatch.delenv("ACTOR_BACKEND", raising=False)
    assert actor.default_predict() is policy.laya_predict
    monkeypatch.setenv("ACTOR_BACKEND", "kev")
    assert actor.default_predict() is kev_backend.kev_predict
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_kev_backend.py -v`
Expected: FAIL with `ImportError: cannot import name 'kev_backend'`

- [ ] **Step 3: Implement**

```python
# jev_ultrafast/kev_backend.py
"""Kev actor served by Kev's own local System One server (separate checkout, see training/kev_export.py)."""

import os

from . import model
from training.kev_export import kev_state

DEFAULT_KEV_URL = "http://127.0.0.1:8090"
DEFAULT_KEV_MODEL = "kev-m2w"


def kev_predict(state: dict, questions: dict) -> dict:
    base = os.environ.get("KEV_BASE_URL", DEFAULT_KEV_URL).rstrip("/")
    body = {"model": os.environ.get("KEV_MODEL", DEFAULT_KEV_MODEL), "state": kev_state(state),
            "questions": questions}
    return model.post_json(base + "/v1/systemone", os.environ.get("KEV_API_KEY", "local"), body, timeout=10)
```

In `jev_ultrafast/actor.py` add:

```python
def default_predict() -> Predict:
    if os.environ.get("ACTOR_BACKEND") == "kev":
        from .kev_backend import kev_predict

        return kev_predict
    from .policy import laya_predict

    return laya_predict
```

In `jev_ultrafast/pilot.py`, change the constructor parameter to `predict: Callable | None = None` and set
`self._predict = predict or default_predict()` (import `default_predict` from `.actor`).

In `training/evaluate.py`, add `"kev"` to `--predictor` choices and a predictor mirroring `laya_predictor` that
calls `kev_predict(row["state"], questions)` instead of `agent.system_one`.

Append to `.env.example`:

```
# Actor model: "laya" (in-process checkpoint) or "kev" (Kev's local server from its own checkout).
ACTOR_BACKEND=laya
KEV_BASE_URL=http://127.0.0.1:8090
KEV_MODEL=kev-m2w
```

- [ ] **Step 4: Run everything**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
scripts/commit.sh "feat: add Kev actor backend behind ACTOR_BACKEND"
```

---

### Task 21: Laya-v2 and Kev-0.8B training, offline evaluation, τ fit

Operational task. Every command's key output line goes in the task report.

**Files:**
- Create: `training/kaggle/laya_kernel.py`, `training/kaggle/kev_kernel.py`,
  `training/kaggle/laya-kernel-metadata.json`, `training/kaggle/kev-kernel-metadata.json`
- Create: `docs/superpowers/reports/2026-09-xx-gate-c-offline.md` (date of the run)

- [ ] **Step 1: Build case files**

```bash
T="data/mind2web/data/train/train_*.json"
uv run python -m training.build_cases --input $T --name train_v2 --dev-mod 20 --step-rate 0.7 --context --seed 0
uv run python -m training.build_cases --input $T --name devstep --dev-mod 20 --step-rate 1.0 --context --seed 1
uv run python -m training.build_cases --input $T --name devgoal --dev-mod 20 --step-rate 0.0 --context --seed 1
for s in task website domain; do
  uv run python -m training.build_cases --input data/mind2web/test/test_$s/*.json --name test_${s}_step \
    --step-rate 1.0 --context --seed 2
  uv run python -m training.build_cases --input data/mind2web/test/test_$s/*.json --name test_${s}_goal \
    --step-rate 0.0 --context --seed 2
done
```

Dev sets used below: `training/out/devstep_dev.jsonl`, `training/out/devgoal_dev.jsonl` (the `devstep.jsonl` /
`devgoal.jsonl` main files are unused duplicates of train and may be deleted).

- [ ] **Step 2: Tokenise and check the budget**

```bash
uv run python -m training.prepare_items --cases training/out/train_v2.jsonl --out training/out/train_items.pt
uv run python -m training.prepare_items --cases training/out/train_v2_dev.jsonl --out training/out/dev_items.pt
```

Expected log: `items=... stats={... 'items_dropped_overflow': N ...}`. If `N / (items + N) > 0.01`, set
`SECTION_CHARS = 24` and `ROW_CHARS = 0` in `jev_ultrafast/formatter.py` (applies to live too, keeping parity),
update the `context_text` test expectations, rerun Steps 1–2, and record the change.

- [ ] **Step 3: Put the Kaggle kernels in the repo**

`training/kaggle/laya_kernel.py` — same flow as the published run (dataset discovery, install, `torchrun`):

```python
"""Kaggle kernel: fine-tune Laya on Mind2Web items (2x T4, DDP). Mind2Web train data only; no test data here."""

import glob
import os
import subprocess
import sys

OUT = "/kaggle/working/laya_browser_v2"


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    found = glob.glob("/kaggle/input/**/train_ddp.py", recursive=True)
    if not found:
        raise FileNotFoundError("train_ddp.py not found under /kaggle/input")
    data = os.path.dirname(found[0])
    run([sys.executable, "-m", "pip", "install", "-q", "laya>=0.3.4", "transformers>=4.48", "safetensors",
         "huggingface_hub", "peft", "torchao>=0.16.0"])
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config

    model_dir = snapshot_download("convaiinnovations/laya")
    _fix_tokenizer_config(model_dir)
    run(["torchrun", "--standalone", "--nproc_per_node=2", f"{data}/train_ddp.py", model_dir,
         f"{data}/train_items.pt", f"{data}/dev_items.pt", OUT])


if __name__ == "__main__":
    main()
```

`training/kaggle/laya-kernel-metadata.json`:

```json
{
  "id": "quantumjohn08/laya-mind2web-fine-tune-2xt4",
  "title": "Laya Mind2Web Fine-tune 2xT4",
  "code_file": "laya_kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": ["quantumjohn08/laya-mind2web-items"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

`training/kaggle/kev_kernel.py` — clones Kev, installs it, trains from `kev-0.8b` on `kev_train.jsonl` using the
flags recorded in Task 19 Step 2 (write them in as literal arguments; `STEPS` comes from env so the smoke run is the
same file):

```python
"""Kaggle kernel: fine-tune Kev-0.8B on exported Mind2Web examples (single T4)."""

import glob
import os
import subprocess
import sys

STEPS = os.environ.get("KEV_STEPS", "50")
OUT = "/kaggle/working/kev_m2w"


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main():
    data = glob.glob("/kaggle/input/**/kev_train.jsonl", recursive=True)[0]
    run(["git", "clone", "--depth", "1", "https://github.com/jaredpalmer/kev.git", "/kaggle/working/kev"])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", "/kaggle/working/kev"])
    # Flag names from `python -m kev.train --help` (Task 19 Step 2); T4 has no bf16, so fp16 or fp32.
    run([sys.executable, "-m", "kev.train", "--data", data, "--init_from", "jaredpalmer/kev-0.8b",
         "<STEP-LIMIT FLAG>", STEPS, "<OUTPUT FLAG>", OUT, "<PRECISION FLAG>", "fp16"],
        cwd="/kaggle/working/kev")


if __name__ == "__main__":
    main()
```

Replace the three `<… FLAG>` strings with the literal flags recorded in Task 19 Step 2 before committing (they are
the only values this plan cannot know in advance; the task report must show them). `kev-kernel-metadata.json` is
the Laya one with `"id": "quantumjohn08/kev-mind2web-0-8b"`, `"title": "Kev Mind2Web 0.8B"`,
`"code_file": "kev_kernel.py"`, and `"dataset_sources": ["quantumjohn08/laya-mind2web-items"]`.

- [ ] **Step 4: Upload the data (new dataset version) and train Laya-v2**

```bash
KAGGLE="uvx --from kaggle kaggle"
mkdir -p training/out/kaggle_dataset
uv run python -m training.kev_export --cases training/out/train_v2.jsonl --out training/out/kaggle_dataset/kev_train.jsonl
cp training/out/train_items.pt training/out/train_items.meta.json training/out/dev_items.pt \
   training/out/dev_items.meta.json training/train_ddp.py training/out/kaggle_dataset/
printf '{"title":"laya-mind2web-items","id":"quantumjohn08/laya-mind2web-items","licenses":[{"name":"CC-BY-4.0"}]}' \
  > training/out/kaggle_dataset/dataset-metadata.json
$KAGGLE datasets version -p training/out/kaggle_dataset -m "v2: step-conditioned, context, hard negatives"
mkdir -p training/out/kernel_laya && cp training/kaggle/laya_kernel.py training/out/kernel_laya/ && \
  cp training/kaggle/laya-kernel-metadata.json training/out/kernel_laya/kernel-metadata.json
$KAGGLE kernels push -p training/out/kernel_laya
```

Poll every 10 minutes with `$KAGGLE kernels status quantumjohn08/laya-mind2web-fine-tune-2xt4` until `COMPLETE`
(or `ERROR`: fetch `$KAGGLE kernels output ... -p training/out/kernel_laya_out` and read the log; fix and re-push
once; a second identical failure → stop and report). Then:

```bash
$KAGGLE kernels output quantumjohn08/laya-mind2web-fine-tune-2xt4 -p checkpoints/laya_browser_v2_dl
mv checkpoints/laya_browser_v2_dl/laya_browser_v2 checkpoints/laya_browser_v2
ls checkpoints/laya_browser_v2   # model.safetensors encoder/ tokenizer/ rl_agent_config.json
```

- [ ] **Step 5: Kev smoke run, then full run**

Push `kev_kernel.py` with `KEV_STEPS=50` (set in the file header via `os.environ.setdefault` for the smoke push),
wait for `COMPLETE`. If it fails on precision, switch the precision flag to fp32 and retry once; if it fails again,
record the error and continue with Laya only (Kev drops out of the comparison; say so in the report). On success,
set the full step count to cover 4 epochs of `kev_train.jsonl` (lines ÷ batch size × 4; batch size from Kev's
defaults), push, wait, download to `checkpoints/kev_m2w`, and start Kev's server from `~/kev` serving that adapter
on port 8090 (command from Kev's README; record it in `.env.example` as a comment).

- [ ] **Step 6: Dev evaluation (selection) and τ**

```bash
for mode in step goal; do
  uv run python -m training.evaluate --cases training/out/dev${mode}_dev.jsonl --predictor laya \
    --checkpoint checkpoints/laya_browser_mind2web --out training/out/eval_v1_dev_${mode}.json
  uv run python -m training.evaluate --cases training/out/dev${mode}_dev.jsonl --predictor laya \
    --checkpoint checkpoints/laya_browser_v2 --out training/out/eval_v2_dev_${mode}.json \
    --dump-picks training/out/picks_v2_dev_${mode}.jsonl
  uv run python -m training.evaluate --cases training/out/dev${mode}_dev.jsonl --predictor kev \
    --out training/out/eval_kev_dev_${mode}.json --dump-picks training/out/picks_kev_dev_${mode}.jsonl
done
uv run python -m training.fit_tau training/out/picks_v2_dev_step.jsonl
uv run python -m training.fit_tau training/out/picks_kev_dev_step.jsonl
```

Note: the v1 checkpoint was trained without context; its dev numbers here (context-rendered rows) are a reference
only. τ is fitted on dev step-mode picks only: the baseline's live decisions used the old request shape, so they
cannot calibrate the new one (a deliberate narrowing of spec §5.5).

- [ ] **Step 7: One confirmatory test run per actor and mode**

```bash
for s in task website domain; do for mode in step goal; do
  uv run python -m training.evaluate --cases training/out/test_${s}_${mode}.jsonl --predictor laya \
    --checkpoint checkpoints/laya_browser_v2 --out training/out/eval_v2_test_${s}_${mode}.json
  uv run python -m training.evaluate --cases training/out/test_${s}_${mode}.jsonl --predictor kev \
    --out training/out/eval_kev_test_${s}_${mode}.json
done; done
```

Run this once. Do not change anything after seeing test numbers.

- [ ] **Step 8: Replay probe with each actor**

```bash
LAYA_CHECKPOINT=checkpoints/laya_browser_v2 uv run python scripts/replay_probe.py
ACTOR_BACKEND=kev uv run --env-file .env python scripts/replay_probe.py   # after making replay_probe use default_predict()
```

(Change `replay_probe.py` to call `default_predict()` from `jev_ultrafast.actor` instead of `laya_predict`; commit
with this task.)

- [ ] **Step 9: Offline report and stop**

Write `docs/superpowers/reports/<date>-gate-c-offline.md`: per actor, dev and test tables in both modes
(`element_acc_given_op`, macro element accuracy, step SR, success rate, with the published checkpoint's goal-mode
test numbers beside them); overflow count; fitted τ with acceptance and accepted accuracy; replay-probe score;
median actor latency on the M2 (from the probe); the goal-mode regression check (≤ 2 points drop in macro element
accuracy versus the published card). Commit:

```bash
scripts/commit.sh "docs: add Gate C offline report and Kaggle kernels"
```

**STOP.** Report to the user and ask which actors go to the live comparison.

---

### Task 22: Gate C — live comparison and report

**Files:**
- Create: `docs/superpowers/reports/<date>-gate-c.md`

- [ ] **Step 1: Run the suite once per actor the user approved**

Same session, same machine, Chrome and planner server as in Task 15:

```bash
POLICY_BACKEND=planner ACTOR_BACKEND=laya LAYA_CHECKPOINT=checkpoints/laya_browser_v2 ACTOR_CONTEXT=1 \
  ACTOR_TAU=<fitted τ for v2> uv run --env-file .env python scripts/live_eval.py
POLICY_BACKEND=planner ACTOR_BACKEND=kev ACTOR_CONTEXT=1 ACTOR_TAU=<fitted τ for Kev> \
  uv run --env-file .env python scripts/live_eval.py
```

- [ ] **Step 2: Summarise and tag**

`scripts/live_summary.py` on each run directory; tag failures exactly as in Task 15 Step 4.

- [ ] **Step 3: Report**

Table with baseline (7/25), Gate A+B result, and each actor: pass count, categories, routes, median actor /
planner / decision ms, wall s/action, failure tags. Verdict against **≥ 18/25 and no category below 2/3**; adopt
the best configuration within the speed budget (ties to the faster). Set the adopted values as defaults in
`.env.example`. Commit:

```bash
scripts/commit.sh "docs: add Gate C live comparison report"
```

**STOP.** Report to the user. Publishing a new model card to Hugging Face happens only if they approve it.
