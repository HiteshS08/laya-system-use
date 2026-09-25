# First-principles redesign — implementation plan

Spec: `docs/superpowers/specs/2026-09-26-first-principles-design.md`. Branch: `redesign/first-principles`.

Every task is test-first: write the test, run it and see it fail, implement, see it pass, then run
`uv run pytest -q` and `uv run ruff check .`, then commit with the task's message via
`scripts/commit.sh "<message>"`. Library code: functions under 50 lines, frozen dataclasses for records, type hints
on public functions, `logging` not `print`, no bare `except`. Every model call in tests is a fake. Tests that need
Chrome skip when `BU_CDP_URL` is not reachable (same pattern as `tests/test_snapshot_live.py`).

Order: tasks 1–8 produce a runnable `POLICY_BACKEND=program` backend (value lands early); 9 proves it end-to-end in a
real browser on local fixtures; 10–11 make the Mac evaluation cheap and honest; 12–13 prepare training data. Tasks
D1–D6 need hardware this environment lacks and are **DEFERRED TO MAC**.

Review checkpoints (re-read the diff since the last checkpoint and fix what is found): after tasks 3, 7, 10 and 13.

Nothing is deleted in tasks 1–13; the old planner stack is deleted in D5, conditionally.

---

## Task 1 — Program grammar

Files: `jev_ultrafast/program.py` (new), `tests/test_program.py` (new).

Interface:
```python
KINDS: tuple[str, ...] = ("FIND", "OPEN", "JUMP", "SCROLL", "FILL", "SELECT", "CLICK", "SUBMIT")
VALUE_KINDS = ("FILL", "SELECT"); MAX_SUBGOALS = 8; MAX_TARGET_CHARS = 80; MAX_VALUE_CHARS = 200; MAX_ORDINAL = 50

@dataclass(frozen=True)
class Subgoal:
    kind: str
    target: str
    value: str = ""
    ordinal: int = 0          # 1-based; 0 = no ordinal

@dataclass(frozen=True)
class Program:
    subgoals: tuple[Subgoal, ...]
    done_text: str = ""
    source: str = "compiler"  # compiler | cache | fallback

def parse_program(text: str) -> Program            # raises ValueError
def render_program(program: Program) -> str
def fallback_program(goal: str) -> Program          # Program((Subgoal("DO", goal),), source="fallback")
```

Test first (`tests/test_program.py`):
```python
import pytest

from jev_ultrafast.program import Program, Subgoal, fallback_program, parse_program, render_program


def test_parses_kinds_values_ordinals_and_done_text():
    p = parse_program("FIND Alan Turing\nOPEN comments @2\nFILL Where to? = London\nSUBMIT\nDONE_WHEN Search results")
    assert p.subgoals == (Subgoal("FIND", "Alan Turing"), Subgoal("OPEN", "comments", ordinal=2),
                          Subgoal("FILL", "Where to?", "London"), Subgoal("SUBMIT", ""))
    assert p.done_text == "Search results" and p.source == "compiler"


def test_tolerates_numbering_bullets_fences_and_think_blocks():
    raw = "<think>plan</think>```\n1. FIND Ada Lovelace\n- JUMP References\n```"
    assert [s.kind for s in parse_program(raw).subgoals] == ["FIND", "JUMP"]


@pytest.mark.parametrize("raw", ["", "GO somewhere", "FILL Name", "FIND", "OPEN x @0", "DO anything",
                                 "\n".join(["FIND a"] * 9), "FIND " + "x" * 81, "DONE_WHEN only"])
def test_rejects_invalid_programs(raw):
    with pytest.raises(ValueError):
        parse_program(raw)


def test_render_round_trips():
    p = parse_program("FIND Ada\nOPEN comments @3\nSELECT Size = M\nSUBMIT Search\nDONE_WHEN Added")
    assert parse_program(render_program(p)) == p


def test_fallback_is_a_single_do_subgoal():
    assert fallback_program("Buy milk") == Program((Subgoal("DO", "Buy milk"),), source="fallback")
```

Requirements: strip `<think>…</think>` and code fences; per line strip whitespace, leading `-`, `*` or `N.`/`N)`;
skip blank lines. `DONE_WHEN text` is only valid as the last non-blank line and needs ≥ 1 subgoal before it. The first
word (upper-cased) is the kind; `DO` is not accepted from text. For FILL/SELECT split on the first ` = `; both sides
non-empty. A trailing ` @N` (1 ≤ N ≤ 50) sets `ordinal`. SUBMIT may have an empty target; every other kind needs one.
Limits as above. `render_program` writes `KIND target[ @N][ = value]` lines and `DONE_WHEN` last.

Commands: `uv run pytest tests/test_program.py -q`. Commit: `feat: add the subgoal program grammar`.

## Task 2 — Compiler (one LLM call per task, cached)

Files: `jev_ultrafast/textmodel.py` (add `complete_text`), `jev_ultrafast/compiler.py` (new),
`tests/test_compiler.py` (new), `tests/test_textmodel.py` (one test added).

Interface:
```python
# textmodel.py
def complete_text(system: str, user: str, *, max_tokens: int = 96, extra: Mapping | None = None) -> tuple[str, dict]
# compiler.py
COMPILER_SYSTEM: str
def cache_key(goal: str) -> str                      # " ".join(goal.casefold().split())
class ProgramCache:                                   # JSON file {key: rendered program}
    def __init__(self, path: Path) -> None
    def get(self, goal: str) -> Program | None        # source="cache"
    def put(self, goal: str, program: Program) -> None
def default_cache() -> ProgramCache                   # $LAYA_CACHE_DIR or ~/.cache/laya-browser, programs.json
def compile_goal(goal: str, *, complete: Callable = complete_text,
                 cache: ProgramCache | None = None) -> tuple[Program, dict]
```

Test first (`tests/test_compiler.py`):
```python
from evals.live_tasks import TASKS
from jev_ultrafast.compiler import COMPILER_SYSTEM, ProgramCache, compile_goal
from jev_ultrafast.program import fallback_program


def fake(*outputs):
    calls = []

    def complete(system, user, **kw):
        calls.append(user)
        out = outputs[len(calls) - 1]
        if isinstance(out, Exception):
            raise out
        return out, {"latency_ms": 5, "usage": {"completion_tokens": 7}}
    return complete, calls


GOAL = "Find Alan Turing's article, then open the Turing Award article from it."


def test_compiles_goal_into_program():
    complete, calls = fake("FIND Alan Turing\nFIND Turing Award")
    program, meta = compile_goal(GOAL, complete=complete)
    assert [s.target for s in program.subgoals] == ["Alan Turing", "Turing Award"]
    assert meta["attempts"] == 1 and meta["source"] == "compiler" and len(calls) == 1 and GOAL in calls[0]


def test_retries_once_with_the_error_then_falls_back():
    complete, calls = fake("GO somewhere", "still bad")
    program, meta = compile_goal("Do a thing", complete=complete)
    assert program == fallback_program("Do a thing") and meta["attempts"] == 2 and "error" in meta
    assert "invalid" in calls[1]


def test_model_failure_falls_back():
    complete, _ = fake(RuntimeError("server down"))
    program, meta = compile_goal("Do a thing", complete=complete)
    assert program.source == "fallback" and "server down" in meta["error"]


def test_cache_hit_skips_the_model(tmp_path):
    cache = ProgramCache(tmp_path / "programs.json")
    complete, calls = fake("FIND Ada Lovelace")
    compile_goal("Find Ada", complete=complete, cache=cache)
    program, meta = compile_goal("  find ADA ", complete=complete, cache=ProgramCache(tmp_path / "programs.json"))
    assert len(calls) == 1 and meta["source"] == "cache" and program.source == "cache" and meta["attempts"] == 0


def test_fallback_programs_are_not_cached(tmp_path):
    cache = ProgramCache(tmp_path / "programs.json")
    complete, _ = fake("bad", "bad")
    compile_goal("Do a thing", complete=complete, cache=cache)
    assert cache.get("Do a thing") is None


def test_prompt_examples_are_not_suite_goals():
    for task in TASKS.values():
        assert task.goal not in COMPILER_SYSTEM
```
Added to `tests/test_textmodel.py`:
```python
def test_complete_text_returns_plain_content(monkeypatch):
    monkeypatch.setattr(model, "post_json", Mock(return_value=reply("<think>x</think>\nFIND Ada\n")))
    text, meta = textmodel.complete_text("sys", "user", max_tokens=32)
    assert text == "FIND Ada" and meta["usage"] == {"total_tokens": 5} and meta["attempts"] == 1
```

Requirements: `complete_text` shares request building with `complete_json` (extract a `_chat(system, user,
max_tokens, extra) -> tuple[str, dict]` helper returning content and meta); strips `<think>` blocks; model from
`COMPILER_MODEL` if set, else `TEXT_MODEL`. `compile_goal` sends the goal as the user message wrapped as
`Task: <goal>`, `max_tokens=96`, `extra=DISABLE_THINKING` (the dict from `planner.py`, moved to `textmodel.py` and
re-exported by `planner.py`). On `ValueError` from parsing, retry once with
`"Task: <goal>\nYour previous answer was invalid: <error>. Answer again with program lines only."`. On `ValueError`
or `RuntimeError` from the model, or a second parse failure, return `fallback_program(goal)` with `meta["error"]`.
`meta` = `{"source", "attempts", "latency_ms", "raw", "completion_tokens"[, "error"]}`. Log a warning on fallback.
The system prompt is the grammar table from spec §4.1 plus four worked examples from domains outside the suite.

Commit: `feat: compile the goal into a subgoal program once per task`.

## Task 3 — Observation additions and completion checks

Files: `jev_ultrafast/snapshot.js`, `tests/fixtures/context_page.html`, `tests/test_snapshot_live.py`,
`jev_ultrafast/checks.py` (new), `tests/test_checks.py` (new).

Interface (`checks.py`):
```python
def title_subject(title: str) -> str                 # text before " - ", " – ", " — ", " | ", " · "
def is_about(page: Mapping, name: str) -> bool
def fragment_names(url: str, section: str) -> bool
def heading_in_view(page: Mapping, text: str) -> bool
def text_shown(page: Mapping, text: str) -> bool     # normalized text in page text or title
def satisfied(subgoal: Subgoal, page: Mapping, start_url: str) -> bool | None
```

Tests first — `tests/test_checks.py`:
```python
from jev_ultrafast.checks import fragment_names, heading_in_view, is_about, satisfied, text_shown, title_subject
from jev_ultrafast.program import Subgoal

URL = "https://w.test/wiki/Alan_Turing"


def page(**kw):
    base = {"url": URL, "title": "Alan Turing - Wikipedia", "text": "Alan Turing\nEarly life",
            "headings": [{"text": "Alan Turing", "level": 1, "id": "firstHeading", "in_viewport": True},
                         {"text": "References", "level": 2, "id": "References", "in_viewport": False}]}
    return {**base, **kw}


def test_title_subject():
    assert title_subject("Alan Turing - Wikipedia") == "Alan Turing"
    assert title_subject("Issues · browser-use/browser-use") == "Issues"
    assert title_subject("Plain") == "Plain"


def test_is_about_uses_title_or_h1_and_ignores_case_accents_and_parentheticals():
    assert is_about(page(), "alan turing")
    python = page(title="Python (programming language) - Wikipedia", headings=[])
    assert is_about(python, "Python programming language") and is_about(python, "Python")
    assert is_about(page(title="Kurt Gödel - Wikipedia", headings=[]), "Kurt Godel")
    assert is_about(page(title="Search results", headings=[{"text": "Ada Lovelace", "level": 1,
                                                            "id": "", "in_viewport": True}]), "Ada Lovelace")
    assert not is_about(page(), "Alan Turing Institute")
    assert not is_about(page(title="Alan Turing Institute - Wikipedia", headings=[]), "Alan Turing")


def test_fragment_names_section():
    assert fragment_names(URL + "#External_links", "External links")
    assert fragment_names(URL + "#References", "references")
    assert not fragment_names(URL, "References")
    assert not fragment_names(URL + "#cite_note-5", "References")


def test_heading_in_view_and_text_shown():
    assert heading_in_view(page(), "Alan Turing") and not heading_in_view(page(), "References")
    assert text_shown(page(), "early LIFE") and text_shown(page(), "wikipedia") and not text_shown(page(), "")


def test_satisfied_by_kind():
    assert satisfied(Subgoal("FIND", "Alan Turing"), page(), "https://w.test/") is True
    assert satisfied(Subgoal("JUMP", "References"), page(url=URL + "#References"), URL) is True
    assert satisfied(Subgoal("JUMP", "References"), page(), URL) is False
    assert satisfied(Subgoal("SCROLL", "Alan Turing"), page(), URL) is True
    assert satisfied(Subgoal("OPEN", "Issues"), page(url="https://g.test/r/issues"), "https://g.test/r") is True
    assert satisfied(Subgoal("OPEN", "Issues"), page(url=URL + "#x"), URL) is False
    assert satisfied(Subgoal("FILL", "Where to?", "London"), page(), URL) is None
```
Added to `tests/test_snapshot_live.py` (fixture gains `<link rel="search" type="application/opensearchdescription+xml"
href="/opensearch.xml">` and `id="External_links"` on its heading):
```python
def test_headings_and_opensearch_link(observed):
    _, page = observed
    headings = {h["text"]: h for h in page["headings"]}
    assert headings["From today's featured article"]["in_viewport"] is True
    assert headings["External links"]["in_viewport"] is False and headings["External links"]["id"] == "External_links"
    assert headings["External links"]["level"] == 2
    assert page["opensearch"].endswith("/opensearch.xml")
```

Requirements: `snapshot.js` returns `headings` (first 200 visible `h1–h6`: `text` squashed ≤ 80, `level`, `id`,
`in_viewport` = its box intersects the viewport vertically) and `opensearch` (href of the first
`link[rel="search"][type="application/opensearchdescription+xml"]`, else ""). Neither enters the fingerprint's
`marker` beyond what already changes with scroll. `is_about`: normalized name equals the normalized title subject,
the subject without a trailing parenthetical, or the first h1 (same two forms). `fragment_names`: URL-decoded
fragment with `_` → space, normalized, equals the normalized section. `satisfied`: FIND → `is_about`; JUMP →
fragment or heading in view; SCROLL → heading in view; OPEN → document URL (no fragment) differs from `start_url`'s;
other kinds → `None`.

Commands: `uv run pytest tests/test_checks.py -q`; with Chrome: `BU_CDP_URL=http://127.0.0.1:9333 uv run pytest
tests/test_snapshot_live.py -q`. Commit: `feat: observe headings and check subgoal completion on the page`.

**Checkpoint A** — review the diff of tasks 1–3.

## Task 4 — Ordinal groups

Files: `jev_ultrafast/groups.py` (new), `tests/test_groups.py` (new).

Interface: `shape(label: str) -> str`; `ordinal_pick(elements: Sequence[Mapping], description: str, n: int) ->
Mapping | None`.

Test first:
```python
from jev_ultrafast.groups import ordinal_pick, shape


def el(i, label, landmark="main", role="link"):
    return {"index": str(i), "label": label, "role": role, "landmark": landmark, "operations": ["CLICK"]}


HN = [el(1, "comments", landmark="nav"), el(2, "Story one"), el(3, "48 comments"), el(4, "Story two"),
      el(5, "1 comment"), el(6, "Story three"), el(7, "120 comments"), el(8, "More")]


def test_shape_masks_numbers_and_plurals():
    assert shape("48 comments") == shape("1 comment") == "# comment"
    assert shape("Comments") == "comment"


def test_picks_the_nth_member_of_the_matching_repeated_group():
    assert [ordinal_pick(HN, "comments", n)["index"] for n in (1, 2, 3)] == ["3", "5", "7"]
    assert ordinal_pick(HN, "comments link", 2)["index"] == "5"


def test_returns_none_without_a_large_enough_matching_group():
    assert ordinal_pick(HN, "comments", 4) is None
    assert ordinal_pick(HN, "reviews", 1) is None
```

Requirements: `shape` = `normalize(label)` with digit runs → `#` and words longer than 3 letters ending in `s`
singularized. Groups keyed by `(role, landmark, shape)` over elements with `CLICK` in `operations`, page order kept.
A group is eligible if its shape shares a (singularized) word with the description and has ≥ n members; choose the
eligible group with the most members (ties: earliest first member). Return member n-1.

Commit: `feat: resolve ordinal targets among repeated elements`.

## Task 5 — Search templates, SUBMIT and host-checked GOTO

Files: `jev_ultrafast/search.py` (new), `jev_ultrafast/tools.py`, `jev_ultrafast/browser.py`,
`tests/test_search.py` (new), `tests/test_tools.py`.

Interface:
```python
# search.py
def template_from_opensearch(xml_text: str, page_url: str) -> str | None
def learn_template(url: str, query: str) -> str | None
def render(template: str, query: str) -> str
def matches_template(url: str, template: str) -> bool
class SearchTemplates:
    def __init__(self, path: Path | None = None) -> None     # None = in memory only
    def get(self, url: str) -> str | None                   # by hostname
    def put(self, url: str, template: str) -> None
def default_templates() -> SearchTemplates                  # $LAYA_CACHE_DIR/search_templates.json
OPENSEARCH_JS: str                                           # async expression -> XML text or null
def discover(browser, page_url: str) -> str | None
# tools.py
def run_tool(browser, operation: str, arg: str, *, templates: Sequence[str] = ()) -> bool
# browser.py
Browser.evaluate(self, expression: str, await_promise: bool = False)
Browser.press_enter(self) -> None
```

Tests first — `tests/test_search.py`:
```python
from unittest.mock import Mock

from jev_ultrafast.search import (SearchTemplates, discover, learn_template, matches_template, render,
                                  template_from_opensearch)

OSD = """<?xml version="1.0"?><OpenSearchDescription xmlns="http://a9.com/-/spec/opensearch/1.1/">
<Url type="application/x-suggestions+json" template="https://en.wikipedia.org/w/api.php?search={searchTerms}"/>
<Url type="text/html" method="get"
 template="https://en.wikipedia.org/w/index.php?title=Special:Search&amp;search={searchTerms}&amp;page={startPage?}"/>
</OpenSearchDescription>"""
WIKI = "https://en.wikipedia.org/wiki/Main_Page"
T = "https://en.wikipedia.org/w/index.php?title=Special:Search&search={q}"


def test_opensearch_html_template_for_the_same_host():
    assert template_from_opensearch(OSD, WIKI) == T
    assert template_from_opensearch(OSD, "https://evil.test/") is None
    assert template_from_opensearch("not xml", WIKI) is None
    assert template_from_opensearch(OSD.replace('type="text/html"', 'type="x"'), WIKI) is None


def test_learn_template_from_an_observed_search():
    assert learn_template("https://ex.test/find?q=ada+lovelace&lang=en", "Ada Lovelace") == \
        "https://ex.test/find?q={q}&lang=en"
    assert learn_template("https://ex.test/wiki/Ada_Lovelace", "Ada Lovelace") is None


def test_render_and_match_are_host_bound():
    url = render(T, "Ada Lovelace")
    assert url == "https://en.wikipedia.org/w/index.php?title=Special:Search&search=Ada+Lovelace"
    assert matches_template(url, T)
    assert not matches_template("https://evil.test/w/index.php?title=Special:Search&search=x", T)
    assert not matches_template(url + "&x=1#frag", T)


def test_store_persists_by_host(tmp_path):
    SearchTemplates(tmp_path / "t.json").put(WIKI, T)
    assert SearchTemplates(tmp_path / "t.json").get("https://en.wikipedia.org/wiki/Other") == T
    assert SearchTemplates().get(WIKI) is None


def test_discover_reads_the_pages_own_description():
    assert discover(Mock(evaluate=Mock(return_value=OSD)), WIKI) == T
    assert discover(Mock(evaluate=Mock(return_value=None)), WIKI) is None
    assert discover(Mock(evaluate=Mock(side_effect=RuntimeError("gone"))), WIKI) is None
```
Added to `tests/test_tools.py`:
```python
def test_goto_accepts_only_urls_rendered_from_a_handed_template():
    from jev_ultrafast.search import render
    t = "https://ex.test/find?q={q}"
    browser = Mock()
    assert tools.run_tool(browser, "GOTO", render(t, "ada"), templates=(t,)) is True
    browser.navigate.assert_called_once()
    with pytest.raises(ValueError):
        tools.run_tool(browser, "GOTO", "https://evil.test/find?q=ada", templates=(t,))


def test_submit_presses_enter():
    browser = Mock()
    assert tools.run_tool(browser, "SUBMIT", "") is True
    browser.press_enter.assert_called_once()
```
Added to `tests/test_agent.py`:
```python
def test_press_enter_dispatches_key_events_then_waits_for_navigation(monkeypatch):
    from jev_ultrafast import browser as b

    br = b.Browser.__new__(b.Browser)
    br.call = Mock()
    urls = iter(["https://ex.test/", "https://ex.test/", "https://ex.test/results?q=a"])
    br.evaluate = Mock(side_effect=lambda expr, **kw: next(urls) if expr == "location.href" else "complete")
    monkeypatch.setattr(b.time, "sleep", Mock())
    br.press_enter()
    kinds = [c.kwargs.get("type") for c in br.call.call_args_list if c.args[0] == "Input.dispatchKeyEvent"]
    assert kinds == ["keyDown", "keyUp"]
```

Requirements: `template_from_opensearch` parses with `xml.etree.ElementTree` (namespace-agnostic), takes the first
`Url` with `type="text/html"` whose `template` contains `{searchTerms}` and whose host equals the page host;
replaces `{searchTerms}` with `{q}`; removes query parameters whose value is an optional `{name?}` placeholder;
returns `None` on any parse error or other required placeholders. `learn_template` finds a query parameter whose
decoded, normalized value equals the normalized query and replaces its value with `{q}` (other parameters kept, order
kept). `render` substitutes `quote_plus(query)`. `matches_template` compiles `re.escape(template)` with `{q}` →
`[^&#]+` and uses `fullmatch`; the host is checked implicitly because it is literal in the template. `discover`
evaluates `OPENSEARCH_JS` (reads the link element, `fetch`es it same-origin with a 3 s timeout, returns text) with
`await_promise=True`; errors or `None` → `None`. `run_tool` GOTO: allowed if `is_allowed_goto(arg)` (existing static
registry, kept for the old backend) or `matches_template(arg, t)` for any handed `t`. `press_enter`: record
`location.href`, dispatch `keyDown` (`key="Enter"`, `code="Enter"`, `windowsVirtualKeyCode=13`, `text="\r"`) and
`keyUp`; poll `location.href` up to `NAVIGATION_WAIT_S` for a change (tolerating `StalePage`/`RuntimeError`), then wait
for `document.readyState == "complete"` up to 15 s as `navigate` does. `Agent._act_tool` passes
`templates=tool.get("templates", ())`.

Commit: `feat: discover site search templates and add SUBMIT`.

## Task 6 — Shared instruction templates and tactics

Files: `jev_ultrafast/instructions.py` (new), `jev_ultrafast/tactics.py` (new), `tests/test_tactics.py` (new).

Interface:
```python
# instructions.py
def instruction(kind: str, target: str = "", value: str = "") -> str
# tactics.py
SEARCH_ROLES = frozenset({"searchbox", "combobox", "textbox"})
OPTION_ROLES = frozenset({"option", "menuitem", "menuitemradio", "gridcell"})

@dataclass(frozen=True)
class Step:
    operation: str                  # CLICK | TYPE_TEXT | SELECT | SCROLL_TO_TEXT | GOTO | SUBMIT | DO
    target_text: str
    value: str
    instruction: str
    ordinal: int = 0
    index: str | None = None        # preset element (deterministic tactic)
    roles: frozenset[str] = frozenset()
    templates: tuple[str, ...] = ()
    purpose: str = ""               # act | search | open_search | pick | result

@dataclass(frozen=True)
class Progress:
    actions: int = 0
    misses: int = 0
    typed: bool = False
    submitted: bool = False
    searched: bool = False
    opened_search: bool = False
    picked: bool = False
    acted: bool = False             # an effective action of the subgoal's own kind
    scrolled: bool = False

def next_step(subgoal: Subgoal, page: Mapping, elements: Sequence[Mapping], progress: Progress,
              template: str | None = None) -> Step | None
```

Tests first (`tests/test_tactics.py`):
```python
from jev_ultrafast.instructions import instruction
from jev_ultrafast.program import Subgoal
from jev_ultrafast.tactics import Progress, next_step

URL = "https://w.test/wiki/Main_Page"
T = "https://w.test/w/index.php?search={q}"


def els(*specs):
    out = []
    for i, (label, role, ops, *extra) in enumerate(specs, 1):
        out.append({"index": str(i), "label": label, "role": role, "operations": list(ops),
                    "href": extra[0] if extra else "", "landmark": "main"})
    return out


PAGE = {"url": URL, "title": "Main Page", "text": "", "headings": []}
SEARCH = els(("Search Wikipedia", "searchbox", ["TYPE_TEXT", "CLICK"]), ("Go", "button", ["CLICK"]))


def test_instruction_templates():
    assert instruction("OPEN", "Issues tab") == "Click the Issues tab."
    assert instruction("FILL", "Where to?", "London") == 'Type "London" into the Where to? field.'
    assert instruction("SELECT", "Size", "M") == 'Select "M" in the Size dropdown.'
    assert instruction("RESULT", "Alan Turing") == "Click the search result for Alan Turing."
    assert instruction("SEARCH", value="Ada") == 'Type "Ada" into the search box.'
    assert instruction("LINK", "Charles Babbage") == "Click the link to Charles Babbage."


def test_find_clicks_an_exact_link_first():
    elements = els(("Charles Babbage", "link", ["CLICK"], URL + "x"), *[(e["label"], e["role"], e["operations"])
                                                                          for e in SEARCH])
    step = next_step(Subgoal("FIND", "Charles Babbage"), PAGE, elements, Progress(), T)
    assert (step.operation, step.index) == ("CLICK", "1")


def test_find_prefers_the_search_template_then_the_search_box():
    step = next_step(Subgoal("FIND", "Ada Lovelace"), PAGE, SEARCH, Progress(), T)
    assert step.operation == "GOTO" and step.target_text == "https://w.test/w/index.php?search=Ada+Lovelace"
    assert step.templates == (T,)
    step = next_step(Subgoal("FIND", "Ada Lovelace"), PAGE, SEARCH, Progress(searched=True), T)
    assert (step.operation, step.value, step.index, step.purpose) == ("TYPE_TEXT", "Ada Lovelace", "1", "search")


def test_find_submits_after_typing_then_picks_the_result():
    assert next_step(Subgoal("FIND", "Ada"), PAGE, SEARCH, Progress(typed=True)).operation == "SUBMIT"
    step = next_step(Subgoal("FIND", "Ada"), PAGE, SEARCH, Progress(typed=True, submitted=True))
    assert (step.operation, step.target_text, step.purpose) == ("CLICK", "Ada", "result")


def test_find_opens_a_collapsed_search_control_first():
    elements = els(("Search", "button", ["CLICK"]), ("Home", "link", ["CLICK"], URL))
    step = next_step(Subgoal("FIND", "Ada"), PAGE, elements, Progress())
    assert (step.operation, step.index, step.purpose) == ("CLICK", "1", "open_search")
    assert next_step(Subgoal("FIND", "Ada"), PAGE, elements, Progress(opened_search=True)).purpose == "result"


def test_open_carries_the_ordinal():
    step = next_step(Subgoal("OPEN", "comments", ordinal=2), PAGE, [], Progress())
    assert (step.operation, step.target_text, step.ordinal) == ("CLICK", "comments", 2)


def test_jump_clicks_the_same_document_fragment_link_else_scrolls():
    elements = els(("References", "link", ["CLICK"], URL + "#References"),
                   ("References", "link", ["CLICK"], "https://other.test/#References"))
    step = next_step(Subgoal("JUMP", "References"), PAGE, elements, Progress())
    assert (step.operation, step.index) == ("CLICK", "1")
    step = next_step(Subgoal("JUMP", "References"), PAGE, elements[1:], Progress())
    assert (step.operation, step.target_text) == ("SCROLL_TO_TEXT", "References")


def test_fill_types_then_picks_the_matching_suggestion():
    field = els(("Where to?", "combobox", ["TYPE_TEXT", "CLICK"]))
    sub = Subgoal("FILL", "Where to?", "London")
    assert next_step(sub, PAGE, field, Progress()).operation == "TYPE_TEXT"
    options = field + els(("Paris", "option", ["CLICK"]), ("London, United Kingdom", "option", ["CLICK"]))[1:]
    options[-1]["index"] = "3"
    step = next_step(sub, PAGE, options, Progress(typed=True))
    assert (step.operation, step.index, step.purpose) == ("CLICK", "3", "pick")
    assert next_step(sub, PAGE, field, Progress(typed=True)) is None
    assert next_step(sub, PAGE, options, Progress(typed=True, picked=True)) is None


def test_select_prefers_a_visible_option_then_a_native_select_then_opens_the_field():
    sub = Subgoal("SELECT", "Trip type", "One way")
    option = els(("One way", "option", ["CLICK"]))
    assert next_step(sub, PAGE, option, Progress()).purpose == "pick"
    native = els(("Trip type", "combobox", ["SELECT"]))
    assert next_step(sub, PAGE, native, Progress()).operation == "SELECT"
    custom = els(("Trip type", "combobox", ["CLICK"]))
    step = next_step(sub, PAGE, custom, Progress())
    assert (step.operation, step.target_text) == ("CLICK", "Trip type")


def test_scroll_submit_click_and_do():
    assert next_step(Subgoal("SCROLL", "External links"), PAGE, [], Progress()).operation == "SCROLL_TO_TEXT"
    assert next_step(Subgoal("SUBMIT", ""), PAGE, [], Progress()).operation == "SUBMIT"
    assert next_step(Subgoal("SUBMIT", "Search"), PAGE, [], Progress()).operation == "CLICK"
    assert next_step(Subgoal("CLICK", "One way"), PAGE, [], Progress()).operation == "CLICK"
    assert next_step(Subgoal("DO", "Buy milk"), PAGE, [], Progress()).operation == "DO"
```

Requirements: instruction templates exactly as the test shows, plus `CLICK` → `Click the {target}.`, `OPTION` →
`Click the "{value}" option.`, `SUBMIT` → `Press Enter to submit.`, `SCROLL` → `Scroll to "{target}".`,
`JUMP` → `Click the {target} section link.`, `OPEN_SEARCH` → `Click the search control to open the search box.`,
`DO` → the target unchanged. FIND order: exact link (normalized label or alias equals the name, role `link`) →
template GOTO (if a template and not `searched`) → search field (TYPE_TEXT element whose role is `searchbox`, or whose
label contains "search", or the only TYPE_TEXT element on the page; not `typed`) → open-search click (CLICK element with
"search" in its label, when no field and not `opened_search`) → SUBMIT (when `typed` and not `submitted`) → result
click. Suggestions/options: elements with a role in `OPTION_ROLES` whose normalized label contains every normalized
word of the value. JUMP same-document test: link href without fragment equals the page URL without fragment and
`fragment_names(href, section)`. FILL returns `None` when typed and no unpicked matching option is visible.

Commit: `feat: map subgoals to concrete steps with generic tactics`.

## Task 7 — Controller

Files: `jev_ultrafast/actor.py` (accept any step with `operation`, `target_text`, `instruction`),
`jev_ultrafast/controller.py` (new), `tests/test_controller.py` (new).

Interface:
```python
MAX_ACTIONS_PER_SUBGOAL = 4
MAX_MISSES_PER_SUBGOAL = 2

class Controller:
    def __init__(self, goal: str, program: Program, *, compile_meta: Mapping | None = None,
                 predict: Predict | None = None, fallback: Callable | None = None,
                 discover: Callable[[str], str | None] | None = None,
                 templates: SearchTemplates | None = None) -> None
    @classmethod
    def from_goal(cls, goal: str, *, compile_fn: Callable = compile_goal, **deps) -> "Controller"
    plans: list[dict]          # [compile record], read by live_eval/live_summary
    trace: list[dict]          # one entry per decision
    actor_calls: int
    def decide(self, page: Mapping, history: Sequence[Mapping]) -> dict
```

Tests first (`tests/test_controller.py`):
```python
from unittest.mock import Mock

from jev_ultrafast.controller import Controller
from jev_ultrafast.program import Program, Subgoal, fallback_program, parse_program
from jev_ultrafast.search import SearchTemplates

HOME = "https://w.test/wiki/Main_Page"
ADA = "https://w.test/wiki/Ada_Lovelace"
T = "https://w.test/w/index.php?search={q}"


def act(i, label, kind="click", role="link", href=""):
    return {"id": f"e{i}", "kind": kind, "label": label, "role": role, "node": i, "value": "", "href": href}


BASE = [act(1, "Search Wikipedia", "fill", "searchbox"), act(2, "Open Search Wikipedia", "click", "searchbox"),
        act(3, "Issues 5"), act(4, "Code"), act(5, "Charles Babbage", href="https://w.test/wiki/Charles_Babbage")]


def page(url=HOME, title="Main Page", actions=BASE, text="", headings=()):
    return {"url": url, "title": title, "text": text, "headings": list(headings), "actions": list(actions)}


def entry(label, op="CLICK", before=HOME, after=HOME, changed=True, kind="click", role="link", **extra):
    return {"operation": op, "kind": kind, "action": label, "role": role, "url_before": before, "url": after,
            "page_changed": changed, **extra}


def answer(*ranked):
    probs = {i: p for i, p in ranked}
    return {"answers": {"click_target": {"choice": ranked[0][0], "confidence": ranked[0][1],
                                         "probabilities": probs}}}


def controller(text, **kw):
    kw.setdefault("predict", Mock(side_effect=AssertionError("actor not expected")))
    kw.setdefault("fallback", Mock(side_effect=AssertionError("fallback not expected")))
    kw.setdefault("discover", Mock(return_value=None))
    kw.setdefault("templates", SearchTemplates())
    program = text if isinstance(text, Program) else parse_program(text)
    return Controller("goal", program, **kw)


def test_find_via_discovered_template_then_done_when_the_page_is_about_it():
    c = controller("FIND Ada Lovelace", discover=Mock(return_value=T))
    d = c.decide(page(), [])
    assert (d["choice"], d["tool"]["operation"], d["route"]) == ("TOOL", "GOTO", "tactic")
    assert d["tool"]["templates"] == [T]
    history = [entry(d["tool"]["arg"], op="GOTO", kind="tool", after=ADA, tool_ok=True)]
    d = c.decide(page(url=ADA, title="Ada Lovelace - Wikipedia"), history)
    assert d["choice"] == "DONE" and "Ada Lovelace" in d["evidence"]
    assert c.actor_calls == 0 and len(c.plans) == 1


def test_find_via_search_box_learns_the_template():
    templates = SearchTemplates()
    c = controller("FIND Ada Lovelace", templates=templates)
    d1 = c.decide(page(), [])
    assert (d1["choice"], d1["value"]) == ("e1", "Ada Lovelace")
    h = [entry("Search Wikipedia", op="TYPE_TEXT", kind="fill", role="searchbox", changed=False, text="Ada Lovelace")]
    d2 = c.decide(page(), h)
    assert d2["tool"]["operation"] == "SUBMIT"
    results = "https://w.test/w/index.php?search=Ada+Lovelace&title=Special:Search"
    h.append(entry("", op="SUBMIT", kind="tool", after=results, tool_ok=True))
    c.decide(page(url=results, title="Search results"), h)
    assert templates.get(HOME) == "https://w.test/w/index.php?search={q}&title=Special:Search"


def test_actor_grounds_descriptions_and_a_no_effect_element_is_excluded():
    predict = Mock(side_effect=[answer(("3", 0.6), ("4", 0.3), ("5", 0.1)), answer(("4", 0.7), ("5", 0.3))])
    c = controller("OPEN Issues tab", predict=predict)
    d1 = c.decide(page(), [])
    assert (d1["choice"], d1["route"]) == ("e3", "actor")
    d2 = c.decide(page(), [entry("Issues 5", changed=False)])
    assert d2["choice"] == "e4" and c.actor_calls == 2
    assert "3" not in predict.call_args.args[1]["click_target"]["criteria"]


def test_ordinal_route_needs_no_model():
    rows = [act(1, "comments"), act(2, "Story one"), act(3, "48 comments"), act(4, "Story two"),
            act(5, "3 comments")]
    d = controller("OPEN comments @2").decide(page(actions=rows), [])
    assert (d["choice"], d["route"]) == ("e5", "ordinal")


def test_fill_uses_the_program_value_then_picks_the_suggestion_then_moves_on():
    field = [act(1, "Where to?", "fill", "combobox"), act(2, "Search", role="button")]
    c = controller("FILL Where to? = London\nCLICK Search")
    d1 = c.decide(page(actions=field), [])
    assert (d1["choice"], d1["operation"], d1["value"], d1["route"]) == ("e1", "TYPE_TEXT", "London", "resolver")
    h = [entry("Where to?", op="TYPE_TEXT", kind="fill", role="combobox", changed=True, text="London")]
    with_options = field + [act(3, "London, United Kingdom", role="option")]
    d2 = c.decide(page(actions=with_options), h)
    assert (d2["choice"], d2["route"]) == ("e3", "tactic")
    h.append(entry("London, United Kingdom", role="option"))
    d3 = c.decide(page(actions=field), h)
    assert d3["choice"] == "e2"


def test_blocked_after_two_misses():
    predict = Mock(return_value=answer(("3", 0.9), ("4", 0.1)))
    c = controller("OPEN Issues tab", predict=predict)
    c.decide(page(), [])
    c.decide(page(), [entry("Issues 5", changed=False)])
    d = c.decide(page(), [entry("Issues 5", changed=False), entry("Code", changed=False)])
    assert d["choice"] == "BLOCKED"


def test_done_text_and_an_already_satisfied_goal_stop_at_once():
    c = controller(Program((Subgoal("OPEN", "Issues"),), done_text="3 open issues"))
    assert c.decide(page(text="There are 3 open issues"), [])["choice"] == "DONE"
    c = controller("FIND Ada Lovelace")
    assert c.decide(page(url=ADA, title="Ada Lovelace - Wikipedia"), [])["choice"] == "DONE"


def test_unexecuted_decision_is_not_counted_as_progress():
    c = controller("FIND Ada Lovelace")
    first = c.decide(page(), [])
    again = c.decide(page(), [])  # the Agent raised StalePage and asked again with the same history
    assert first["choice"] == again["choice"] == "e1"


def test_do_fallback_delegates_to_the_goal_mode_policy():
    fallback = Mock(return_value={"choice": "e4", "operation": "CLICK", "probabilities": {"e4": 0.8}})
    c = controller(fallback_program("Buy milk"), fallback=fallback)
    d = c.decide(page(), [])
    assert (d["choice"], d["route"]) == ("e4", "goal_fallback") and fallback.call_args.args[1] == "goal"


def test_from_goal_compiles_once_and_records_it():
    compile_fn = Mock(return_value=(parse_program("FIND Ada"), {"source": "compiler", "latency_ms": 7,
                                                                "attempts": 1}))
    c = Controller.from_goal("Find Ada", compile_fn=compile_fn, predict=Mock(), fallback=Mock(),
                             discover=Mock(return_value=None), templates=SearchTemplates())
    assert compile_fn.call_count == 1 and c.plans[0]["latency_ms"] == 7 and c.plans[0]["program"] == "FIND Ada"


def test_decisions_have_the_shape_the_agent_needs():
    d = controller("FIND Ada Lovelace").decide(page(), [])
    for key in ("choice", "operation", "target", "confidence", "probabilities", "latency_ms", "usage", "route",
                "instruction", "value", "tool", "evidence", "actor_ms"):
        assert key in d
    assert d["probabilities"][d["choice"]] == 1.0
```

Requirements:
- `decide`: (1) attribute new history entries to the pending decision (only when `len(history)` grew; otherwise
  drop the pending record — the Agent did not execute it); update `Progress` (`actions`+1; TYPE_TEXT → `typed`;
  SUBMIT → `submitted`; GOTO → `searched`; purpose `open_search` → `opened_search`; purpose `pick` → `picked`;
  effective action of the subgoal's own kind → `acted`; SCROLL_TO_TEXT with `tool_ok` → `scrolled`); a miss is a
  failure by `memory.is_failure` semantics (tool_ok False, or no change for a non-TYPE_TEXT, non-focus click) and adds
  the element index to this subgoal's exclusions; after a SUBMIT or search-box GOTO that lands on a URL, call
  `learn_template(url, query)` and store it. (2) `done_text` shown → DONE. (3) advance while the current subgoal is
  complete (page predicate true; or FILL typed with no unpicked matching option, or picked; SELECT/CLICK/SUBMIT/OPEN
  `acted` — OPEN also completes on its URL predicate after `acted`; SCROLL `scrolled`), resetting `Progress` and the
  subgoal's `start_url`. (4) past the last subgoal, or the last subgoal's page predicate already true → DONE with
  evidence (`title shows "<subject>"`, `heading "<text>" in view`, `url fragment #<frag>`, or the done text).
  (5) `misses ≥ 2` or `actions ≥ 4` on the current subgoal → BLOCKED with the reason as evidence. (6) `next_step`
  with the host template (store first, then `discover` once per host) → tool decision, or grounding: preset index →
  ordinal (`groups.ordinal_pick`) → `resolve` → `actor.pick_target` (top-1; `actor_calls` += 1 when a model was
  asked) over elements supporting the operation, minus exclusions, restricted to `roles` if given. No grounding →
  BLOCKED. `DO` → `fallback(page, goal, history)` with route `goal_fallback`. SELECT maps the value to an option
  with `resolve` over option labels; no match → BLOCKED.
- Decision dict: the keys the test lists plus `operation_probabilities`, `target_probabilities`,
  `target_confidence`, `raw_answers`, `model`, `request`; `tool` for tools is
  `{"operation", "arg", "templates": [...]}`.
- `trace` entry: `{"subgoal": i, "kind", "route", "step": asdict(step) or None, "progress": asdict(progress)}`.
- `from_goal` calls `compile_fn(goal, cache=default_cache())` (tests pass a fake) and records
  `{"program": render_program(program), **meta}` in `plans`.
- `actor.py`: replace the `PlanStep` type hints with a `Protocol` (`operation`, `target_text`, `instruction`) and drop
  the import of `planner`.

Commit: `feat: drive subgoals with a controller that calls no LLM per step`.

**Checkpoint B** — review the diff of tasks 4–7.

## Task 8 — Agent integration, timings and records

Files: `jev_ultrafast/agent.py`, `scripts/live_eval.py`, `.env.example`, `tests/test_agent.py`,
`tests/test_live_eval.py` (new).

Tests first:
```python
# tests/test_agent.py
def test_program_backend_creates_a_controller(monkeypatch):
    monkeypatch.setenv("POLICY_BACKEND", "program")
    browser = Mock(observe=Mock(return_value=page()))
    monkeypatch.setattr(loop, "Browser", Mock(return_value=browser))
    made = Mock()
    monkeypatch.setattr(loop.Controller, "from_goal", Mock(return_value=made))
    agent = loop.Agent("https://example.test/", "Find a book")
    assert agent.pilot is made
    assert loop.Controller.from_goal.call_args.args[0] == "Find a book"


def test_history_records_observe_and_act_timings(runner):
    runner.state["decision"] = decision("e3")
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    entry = runner.state["history"][-1]
    assert isinstance(entry["observe_ms"], int) and isinstance(entry["act_ms"], int)


def test_tool_templates_reach_run_tool(runner, monkeypatch):
    run_tool = Mock(return_value=True)
    monkeypatch.setattr(loop, "run_tool", run_tool)
    runner.state["decision"] = {**decision("TOOL"), "operation": "GOTO", "route": "tactic",
                                "tool": {"operation": "GOTO", "arg": "https://ex.test/?q=a",
                                         "templates": ["https://ex.test/?q={q}"]}, "probabilities": {"TOOL": 1.0}}
    runner.command("act", {"fingerprint": runner.state["page"]["fingerprint"]})
    assert run_tool.call_args.kwargs["templates"] == ["https://ex.test/?q={q}"]
```
```python
# tests/test_live_eval.py
from types import SimpleNamespace

from scripts.live_eval import pilot_fields


def test_pilot_fields_for_the_program_backend():
    pilot = SimpleNamespace(plans=[{"program": "FIND Ada", "latency_ms": 5, "attempts": 1, "source": "compiler"}],
                            trace=[{"route": "resolver"}], actor_calls=2)
    fields = pilot_fields(pilot)
    assert fields == {"planner_calls": pilot.plans, "program": "FIND Ada", "trace": pilot.trace,
                      "actor_calls": 2, "llm_calls": 1}


def test_pilot_fields_for_the_planner_backend_and_none():
    assert pilot_fields(SimpleNamespace(plans=[{"latency_ms": 1}, {"error": "x"}]))["llm_calls"] == 2
    assert pilot_fields(None) == {"planner_calls": [], "llm_calls": 0}
```
Requirements: `POLICY_BACKEND=program` → `Controller.from_goal(task, discover=lambda url: discover(self.browser, url),
templates=default_templates())` (browser created first; the lambda reads `self.browser` lazily). History entries gain
`observe_ms` (the post-action observe) and `act_ms` (the `browser.act`/`run_tool` call). `_act_tool` passes
`templates=tool.get("templates", ())`. `live_eval.py` gains `pilot_fields(pilot) -> dict` (`llm_calls` = sum of
`attempts` in `plans` when present, else the number of plan records) used in `run`. `.env.example` documents
`POLICY_BACKEND=program`, `COMPILER_MODEL` and `LAYA_CACHE_DIR`.

Commit: `feat: run the program backend from the agent loop and record its costs`.

## Task 9 — End-to-end fixture suite in a real browser

Files: `tests/fixtures/site/{home,results,ada,babbage,list,item2,form,done}.html` (new),
`tests/test_e2e_fixture.py` (new).

Tests (skip without Chrome; the compiler is faked with `monkeypatch.setattr(controller_module, "compile_goal", …)`
and the actor with a lexical fake):
```python
import re
from pathlib import Path

import pytest

from tests.test_snapshot_live import _chrome_up

SITE = Path(__file__).parent / "fixtures" / "site"
pytestmark = pytest.mark.skipif(not _chrome_up(), reason="needs a throwaway Chrome on BU_CDP_URL")


def lexical_predict(state, questions):
    words = set(re.findall(r"[a-z0-9]+", state["goal"].lower()))
    name, question = next(iter(questions.items()))
    scores = {i: len(words & set(re.findall(r"[a-z0-9]+", text.lower()))) for i, text in question["criteria"].items()}
    best = max(scores, key=lambda i: (scores[i], -int(i)))
    probs = {i: (0.9 if i == best else 0.1 / max(len(scores) - 1, 1)) for i in scores}
    return {"answers": {name: {"choice": best, "confidence": 0.9, "probabilities": probs}}}


def run(monkeypatch, tmp_path, start, program_text, steps=8):
    from jev_ultrafast import agent as loop
    from jev_ultrafast import controller as ctl
    from jev_ultrafast.program import parse_program

    monkeypatch.setenv("POLICY_BACKEND", "program")
    monkeypatch.setenv("LAYA_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(ctl, "compile_goal", lambda goal, **kw: (parse_program(program_text), {"attempts": 1}))
    monkeypatch.setattr(ctl.policy, "laya_predict", lexical_predict)
    with loop.Agent((SITE / start).as_uri(), "fixture goal") as agent:
        for _ in range(steps):
            agent.command("tick")
            if agent.state["status"] in {"done", "blocked"}:
                break
        return agent.state


def test_find_by_typing_into_search_then_opening_the_result(monkeypatch, tmp_path):
    state = run(monkeypatch, tmp_path, "home.html", "FIND Ada Lovelace")
    assert state["status"] == "done" and state["page"]["url"].endswith("ada.html")


def test_find_then_follow_an_on_page_link(monkeypatch, tmp_path):
    state = run(monkeypatch, tmp_path, "ada.html", "FIND Ada Lovelace\nFIND Charles Babbage")
    assert state["status"] == "done" and state["page"]["url"].endswith("babbage.html")
    assert len(state["history"]) == 1


def test_jump_sets_the_fragment(monkeypatch, tmp_path):
    state = run(monkeypatch, tmp_path, "ada.html", "JUMP References")
    assert state["status"] == "done" and state["page"]["url"].endswith("#References")


def test_scroll_until_heading_visible(monkeypatch, tmp_path):
    state = run(monkeypatch, tmp_path, "ada.html", "SCROLL External links")
    assert state["status"] == "done" and state["page"]["scroll"]["y"] > 1000


def test_open_the_nth_repeated_item(monkeypatch, tmp_path):
    state = run(monkeypatch, tmp_path, "list.html", "OPEN comments @2")
    assert state["status"] == "done" and state["page"]["url"].endswith("item2.html")


def test_fill_with_suggestion_then_submit(monkeypatch, tmp_path):
    state = run(monkeypatch, tmp_path, "form.html",
                "FILL Where to? = London\nCLICK Search\nDONE_WHEN Results for London")
    assert state["status"] == "done" and state["page"]["url"].endswith("done.html")
```
Fixture requirements: plain static HTML, no network. `home.html` has a GET `<form action="results.html">` with an
`<input type="search" name="search" aria-label="Search Fixturepedia">`; `results.html` lists links to `ada.html`
("Ada Lovelace") and two distractors; `ada.html` has title "Ada Lovelace - Fixturepedia", `<h1>Ada Lovelace</h1>`, a
TOC with `href="#References"` and `href="#External_links"`, 1,500 px blocks between sections, `h2#References`,
`h2#External_links`, and a link "Charles Babbage" → `babbage.html` (title "Charles Babbage - Fixturepedia").
`list.html` has a `<nav>` "comments" link and a table of 3 story rows each followed by a "N comments" link to
`item1..3.html` (only `item2.html` needs to exist). `form.html` has an `<input role="combobox" aria-label="Where
to?">` whose `input` handler renders `<li role="option">` suggestions ("London, United Kingdom", "Londrina,
Brazil") that set the field value on click, and a "Search" button that navigates to `done.html` (text "Results for
London").

Commands: start the throwaway Chromium (see the report), then
`BU_CDP_URL=http://127.0.0.1:9333 uv run pytest tests/test_e2e_fixture.py -q`.
Commit: `test: run the program backend end to end on local fixture pages`.

## Task 10 — Honest summaries and paired comparison

Files: `scripts/live_summary.py`, `scripts/compare_runs.py` (new), `tests/test_live_summary.py`,
`tests/test_compare_runs.py` (new).

Interface:
```python
# live_summary.py (added keys)
"llm_calls_per_task", "actor_calls_per_task", "median_observe_ms", "median_act_ms", "median_wall_s_per_action"
# compare_runs.py
def outcomes(records: Sequence[dict]) -> dict[str, bool]
def majority(runs: Sequence[Mapping[str, bool]]) -> dict[str, bool]
def mcnemar_exact(only_a: int, only_b: int) -> float
def compare(a: Mapping[str, bool], b: Mapping[str, bool]) -> dict
```
Tests first:
```python
# tests/test_live_summary.py (added)
def test_summary_reports_model_calls_and_timings():
    records = [
        {**rec("a", "x", True, 2, 6.0, []), "llm_calls": 1, "actor_calls": 1,
         "steps": [{"observe_ms": 300, "act_ms": 100}, {"observe_ms": 500, "act_ms": 300}]},
        {**rec("b", "x", False, 1, 2.0, []), "llm_calls": 0, "actor_calls": 3,
         "steps": [{"observe_ms": 400, "act_ms": 200}]},
    ]
    s = summarize(records)
    assert s["llm_calls_per_task"] == 0.5 and s["actor_calls_per_task"] == 2.0
    assert s["median_observe_ms"] == 400 and s["median_act_ms"] == 200
    assert s["median_wall_s_per_action"] == 2.5
```
```python
# tests/test_compare_runs.py
from scripts.compare_runs import compare, majority, mcnemar_exact, outcomes


def test_outcomes_and_majority():
    assert outcomes([{"task": "a", "success": True}, {"task": "b", "success": False}]) == {"a": True, "b": False}
    runs = [{"a": True, "b": False}, {"a": True, "b": True}, {"a": False, "b": False}]
    assert majority(runs) == {"a": True, "b": False}


def test_mcnemar_exact_two_sided():
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(0, 6) - 0.03125) < 1e-9
    assert abs(mcnemar_exact(1, 7) - 0.0703125) < 1e-9


def test_compare_counts_discordant_pairs_on_shared_tasks():
    a = {"t1": True, "t2": False, "t3": True, "t4": False}
    b = {"t1": True, "t2": True, "t3": False, "t4": True, "t5": True}
    c = compare(a, b)
    assert (c["n"], c["a_passed"], c["b_passed"], c["only_a"], c["only_b"]) == (4, 2, 3, 1, 2)
    assert c["p_value"] == mcnemar_exact(1, 2)
```
Requirements: `median_wall_s_per_action` = median over tasks with ≥ 1 step of `seconds / steps`. `llm_calls` falls
back to `len(planner_calls)` when a record lacks it. `majority` = strictly more than half of the runs passed.
`mcnemar_exact(b, c)` = `min(1, 2 · Σ_{k ≤ min(b, c)} C(b+c, k) / 2^(b+c))`, 1.0 when `b + c = 0`. CLI:
`uv run python scripts/compare_runs.py --a RUN [RUN ...] --b RUN [RUN ...]` prints JSON (majority per side when
several runs are given).

Commit: `feat: report model calls and timings, and compare runs with a paired test`.

**Checkpoint C** — review the diff of tasks 8–10.

## Task 11 — Compiler benchmark script (run deferred)

Files: `scripts/bench_compile.py` (new), `evals/heldout_goals.py` (new), `tests/test_bench_compile.py` (new).

Interface: `HELDOUT: list[tuple[str, str]]` (start URL, goal; 12 entries on sites outside the suite, read-only);
`bench(goals: Sequence[str], complete: Callable) -> dict` returning `{"n", "valid", "fallback", "median_ms",
"median_completion_tokens", "programs": {goal: text}}`.

Test first:
```python
from evals.heldout_goals import HELDOUT
from evals.live_tasks import TASKS
from scripts.bench_compile import bench


def test_bench_counts_valid_and_fallback_programs():
    outputs = iter(["FIND Ada", "nonsense", "still nonsense"])
    s = bench(["Find Ada", "Do x"], lambda system, user, **kw: (next(outputs), {"latency_ms": 10,
                                                                             "usage": {"completion_tokens": 4}}))
    assert (s["n"], s["valid"], s["fallback"]) == (2, 1, 1) and s["programs"]["Find Ada"] == "FIND Ada"


def test_heldout_goals_are_new_sites():
    suite_hosts = {t.url.split("/")[2] for t in TASKS.values()}
    assert len(HELDOUT) == 12 and not {u.split("/")[2] for u, _ in HELDOUT} & suite_hosts
```
Commit: `feat: add a compiler benchmark over suite and held-out goals`.

## Task 12 — Step-mode training items from Mind2Web (run deferred)

Files: `training/step_items.py` (new), `tests/test_step_items.py` (new).

Interface:
```python
KIND_FOR_OP = {"CLICK": "OPEN", "TYPE_TEXT": "FILL", "SELECT": "SELECT"}
def describe(label: str, rng: random.Random, drop_p: float = 0.3) -> str
def step_row(task: dict, index: int, parsed: ParsedStep, history: Sequence[str], rng: random.Random,
             *, goal_mode_p: float = 0.3, k: int = DEFAULT_K) -> dict | None
def main(argv: list[str] | None = None) -> None     # --input shards --out file --seed --dev-mod
```
Test first:
```python
import random

from jev_ultrafast.instructions import instruction
from tests.m2w_fixtures import make_step, make_task
from training.mind2web import parse_step
from training.step_items import describe, step_row


def test_describe_drops_at_most_one_word():
    rng = random.Random(0)
    outs = {describe("Find flights now", rng, drop_p=1.0) for _ in range(20)}
    assert all(len(o.split()) == 2 for o in outs) and describe("Go", rng, drop_p=1.0) == "Go"


def test_step_row_uses_the_serving_instruction_and_one_target_question():
    task = make_task(make_step("TYPE", gold="20", value="nfl"))
    row = step_row(task, 0, parse_step(task["actions"][0]), [], random.Random(1), goal_mode_p=0.0)
    assert row["state"]["goal"] == instruction("FILL", "Find", "nfl")
    assert list(row["questions"]) == ["type_text_target"]
    gold = row["gold"]["type_text_target"]["probabilities"]
    assert gold[row["gold_id"]] == 1.0 and row["mode"] == "step"


def test_goal_mode_rows_keep_the_task_goal():
    task = make_task(make_step("CLICK", gold="10"))
    row = step_row(task, 0, parse_step(task["actions"][0]), [], random.Random(1), goal_mode_p=1.0)
    assert row["state"]["goal"] == "Find NFL scores" and row["mode"] == "goal"


def test_unusable_steps_are_skipped():
    task = make_task(make_step("CLICK", gold="40"))  # a div, not interactive
    assert step_row(task, 0, parse_step(task["actions"][0]), [], random.Random(1)) is None
```
Requirements: the description is the gold label (drop one random word with probability `drop_p` when it has ≥ 2
words); the instruction is `instruction(KIND_FOR_OP[op], description, value)`; candidates are the ops-supporting pool
shortlisted with `shortlist(f"{instruction} {description}", [], pool, k)` (the serving query); the question is built
with `actor.actor_request`; rows whose gold is missing, not interactive, not shortlisted, or alone are skipped. The
row keeps the fields `prepare_items.py` reads (`state`, `questions`, `gold`, `gold_id`, `website`, `task_id`) plus
`mode`. The test splits are never read by this script (the CLI refuses paths containing `test`).

Commit: `feat: build step-mode actor items with the serving instruction templates`.

## Task 13 — Exploration items from live pages (run deferred)

Files: `scripts/explore.py` (new), `tests/test_explore.py` (new).

Interface:
```python
EXCLUDED_HOSTS: frozenset[str]          # hosts of evals/live_tasks.TASKS
def excluded(url: str) -> bool
def items_from_page(page: Mapping, rng: random.Random, n: int = 8) -> list[dict]
def main(argv: list[str] | None = None) -> None   # --seeds URL... --pages N --out file; robots.txt, 1 req/s
```
Test first:
```python
import random

from scripts.explore import excluded, items_from_page


def act(i, label, kind="click", role="link"):
    return {"id": f"e{i}", "kind": kind, "label": label, "role": role, "node": i, "value": ""}


PAGE = {"url": "https://docs.example/", "title": "Docs", "actions": [
    act(1, "Getting started"), act(2, "API reference"), act(3, "Search docs", "fill", "searchbox"),
    act(4, "Open Search docs", "click", "searchbox"), act(5, "Download"), {"id": "wait", "kind": "wait",
                                                                          "label": "Wait"}]}


def test_suite_hosts_are_excluded():
    assert excluded("https://en.wikipedia.org/wiki/X") and excluded("https://news.ycombinator.com/")
    assert not excluded("https://docs.example/")


def test_items_have_serving_shape_and_a_gold_among_the_options():
    items = items_from_page(PAGE, random.Random(0), n=4)
    assert 1 <= len(items) <= 4
    for item in items:
        (name, question), = item["questions"].items()
        assert name in {"click_target", "type_text_target"}
        assert item["gold_id"] in question["criteria"]
        assert item["gold"][name]["probabilities"][item["gold_id"]] == 1.0
        assert item["source"] == "explore" and item["website"] == "docs.example"
```
Requirements: items use `model.action_space`, `pruning.prune_actions`, `policy.candidates_from`, the serving
shortlist query and `actor.actor_request`, and `instructions.instruction` (`OPEN` for links/buttons, `FILL` with a
value sampled from the page title words); labels shorter than 2 characters or duplicated labels are skipped as gold.
`main` visits same-host links breadth-first from the seeds, checks `urllib.robotparser`, sleeps ≥ 1 s between
requests, never submits forms or types, and writes JSONL. No LLM is called.

Commit: `feat: generate actor items from explored public pages without an LLM`.

**Checkpoint D** — review the diff of tasks 11–13; then write the overnight report.

---

## Deferred tasks (DEFERRED TO MAC)

### D1 — Compiler model benchmark (DEFERRED TO MAC: MLX, model download)
```bash
uv run mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8080 &
COMPILER_MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit uv run --env-file .env python scripts/bench_compile.py > artifacts/bench_compile_4b.json
kill %1
uv run mlx_lm.server --model mlx-community/Qwen3-1.7B-4bit --port 8080 &
COMPILER_MODEL=mlx-community/Qwen3-1.7B-4bit uv run --env-file .env python scripts/bench_compile.py > artifacts/bench_compile_1_7b.json
```
Hand-judge each program in both files; pick the smallest model with ≥ 90% valid and correct programs and median ≤ 5 s;
set `COMPILER_MODEL`/`TEXT_MODEL` in `.env`.

### D2 — Live suite, 3 runs, plus held-out tasks (DEFERRED TO MAC: Chrome, Laya, live sites)
Write checks for the 12 `HELDOUT` goals as `LiveTask`s in `evals/heldout_tasks.py` and validate each check by hand
before the first run. Then the commands in the overnight report's "Evaluate this branch" section.

### D3 — Step-mode actor retraining (DEFERRED TO MAC/KAGGLE: Mind2Web data, GPUs)
```bash
uv run python training/fetch_data.py train
uv run python training/step_items.py --input data/mind2web/train/*.json --out training/out/step_train.jsonl --dev-mod 20 --seed 0
uv run python training/prepare_items.py --cases training/out/step_train.jsonl --out training/out/step_items.pt
# upload step_items.pt to Kaggle, run training/train_ddp.py as in training/KAGGLE.md, then evaluate on dev:
uv run python training/evaluate.py --predictor laya --checkpoint checkpoints/laya_step --cases training/out/step_train_dev.jsonl
```
### D4 — Exploration run (DEFERRED TO MAC; needs the user's approval of the seed sites)
```bash
uv run python scripts/explore.py --seeds https://docs.python.org/3/ https://developer.mozilla.org/en-US/ --pages 200 --out data/explore/items.jsonl
```
### D5 — Delete the per-step planner stack (DEFERRED; only if D2 shows the program backend ≥ the planner backend)
Delete `pilot.py`, `planner.py`, `verifier.py`, the pick path in `router.py`, `scripts/bench_planner.py`,
`scripts/check_guards.py`, the static `SEARCH_TEMPLATES`, and their tests; make `program` the default backend.
Commit: `refactor: remove the per-step planner backend`.

### D6 — Kev-0.8B bake-off (optional, DEFERRED TO MAC)
Only after D3; same step-mode items exported to Kev's JSONL; adopt only if live success is higher within the speed
budget.

---

## Deviations

(Recorded during implementation.)
