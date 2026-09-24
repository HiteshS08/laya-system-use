# Part 1 — Phase A+B foundations (Tasks 1–5)

Read `../2026-09-24-planner-actor.md` (Global Constraints, Review Focus) and the spec first.
Run every command from the repo root `~/laya-browser`.

---

### Task 1: Detour filter and duplicate-link merge

Removes links that lead to editors/history pages (the "View source" detour caused five browser timeouts in the
baseline) and merges links that point to the same URL (two "Mary Mallon" entries split the actor's probability).

**Files:**
- Create: `jev_ultrafast/pruning.py`
- Test: `tests/test_pruning.py`

**Interfaces:**
- Consumes: raw page actions as produced by `snapshot.js` (dicts with `id`, `kind`, `label`, `role`, `node`, and —
  after Task 2 — `href`, `aliases`).
- Produces:
  - `is_detour(href: str, goal: str) -> bool`
  - `prune_actions(actions: Sequence[Mapping], goal: str) -> list[dict]` — returns new dicts; kept link actions
    always carry `aliases: list[str]` (labels of merged duplicates). Non-click actions pass through unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pruning.py
from jev_ultrafast.pruning import is_detour, prune_actions

WIKI = "https://en.wikipedia.org"


def link(i, label, href, node=None):
    return {"id": f"e{i}", "kind": "click", "label": label, "role": "link", "node": node or i, "href": href}


def test_editor_and_history_links_are_detours():
    assert is_detour(f"{WIKI}/w/index.php?title=Main_Page&action=edit", "Open today's featured article.")
    assert is_detour(f"{WIKI}/w/index.php?title=Main_Page&action=history", "Open the article")
    assert is_detour("https://github.com/a/b/edit/main/README.md", "Open the Issues tab")
    assert not is_detour(f"{WIKI}/wiki/Mary_Mallon", "Open today's featured article.")
    assert not is_detour("", "anything")


def test_goal_that_asks_for_editing_keeps_detours():
    assert not is_detour(f"{WIKI}/w/index.php?title=X&action=edit", "View the source of this page")
    assert not is_detour(f"{WIKI}/w/index.php?title=X&action=history", "Show the revision history")


def test_detour_links_are_removed_and_other_actions_kept():
    actions = [link(1, "View source", f"{WIKI}/w/index.php?title=Main_Page&action=edit"),
               link(2, "Mary Mallon", f"{WIKI}/wiki/Mary_Mallon"),
               {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560}]
    kept = prune_actions(actions, "Open today's featured article.")
    assert [a["id"] for a in kept] == ["e2", "scroll_down"]


def test_links_to_the_same_url_merge_into_the_first_with_aliases():
    actions = [link(1, "Mary Mallon", f"{WIKI}/wiki/Mary_Mallon"),
               link(2, "Full article...", f"{WIKI}/wiki/Mary_Mallon"),
               link(3, "Mary Mallon", f"{WIKI}/wiki/Mary_Mallon")]
    kept = prune_actions(actions, "Open today's featured article.")
    assert [a["id"] for a in kept] == ["e1"]
    assert kept[0]["aliases"] == ["Full article..."]


def test_fragments_are_distinct_targets_and_placeholder_hrefs_never_merge():
    actions = [link(1, "8 References", f"{WIKI}/wiki/Ada#References"),
               link(2, "10 External links", f"{WIKI}/wiki/Ada#External_links"),
               link(3, "Menu", f"{WIKI}/wiki/Ada#"), link(4, "More", f"{WIKI}/wiki/Ada#"),
               link(5, "Run", "javascript:void(0)"), link(6, "Stop", "javascript:void(0)")]
    assert [a["id"] for a in prune_actions(actions, "Jump to References")] == ["e1", "e2", "e3", "e4", "e5", "e6"]


def test_input_actions_are_not_mutated():
    actions = [link(1, "A", f"{WIKI}/wiki/A"), link(2, "B", f"{WIKI}/wiki/A")]
    prune_actions(actions, "goal")
    assert "aliases" not in actions[0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_pruning.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jev_ultrafast.pruning'`

- [ ] **Step 3: Implement**

```python
# jev_ultrafast/pruning.py
"""Option-set hygiene before any model sees the page: drop known detours, merge links to the same place."""

import re
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

_DETOUR_QUERY = re.compile(r"(?:^|&)action=(?:edit|history|raw|info)(?:&|$)")
_DETOUR_PATH = re.compile(r"/edit(?:/|$)")
# A goal that talks about editing, history or source really wants those pages.
_DETOUR_GOAL = re.compile(r"\b(?:edit\w*|history|revisions?|source)\b", re.IGNORECASE)


def is_detour(href: str, goal: str) -> bool:
    if not href or _DETOUR_GOAL.search(goal):
        return False
    parts = urlsplit(href)
    return bool(_DETOUR_QUERY.search(parts.query) or _DETOUR_PATH.search(parts.path))


def _merge_key(href: str) -> str | None:
    """Only real destinations merge; '#' placeholders and javascript: links are distinct controls."""
    if not href.startswith(("http://", "https://")) or href.endswith("#"):
        return None
    return href


def prune_actions(actions: Sequence[Mapping], goal: str) -> list[dict]:
    kept: list[dict] = []
    first_by_href: dict[str, dict] = {}
    for action in actions:
        href = action.get("href") or ""
        is_link = action.get("kind") == "click" and action.get("role") == "link"
        if action.get("kind") == "click" and is_detour(href, goal):
            continue
        key = _merge_key(href) if is_link else None
        if key is None:
            kept.append(dict(action))
            continue
        first = first_by_href.get(key)
        if first is None:
            first = {**action, "aliases": list(action.get("aliases", []))}
            first_by_href[key] = first
            kept.append(first)
        elif action.get("label") and action["label"] != first["label"] and action["label"] not in first["aliases"]:
            first["aliases"] = [*first["aliases"], action["label"]]
    return kept
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pruning.py -v && uv run ruff check jev_ultrafast/pruning.py tests/test_pruning.py`
Expected: 6 passed; ruff `All checks passed!`

- [ ] **Step 5: Commit**

```bash
scripts/commit.sh "feat: drop detour links and merge duplicate links before choosing"
```

---

### Task 2: Whole-page snapshot with context fields; scroll into view before input

Today `snapshot.js` drops every element whose centre is outside the viewport, and the executor refuses to act on
one. This task captures the whole document (cap 400), adds `in_viewport`, `y`, `href`, `landmark`, `section`,
`row_text` per element and a headings `outline` per page, and makes the executor scroll an off-screen target into
view before its hit test.

**Files:**
- Modify: `jev_ultrafast/snapshot.js` (element loop, returned object)
- Modify: `jev_ultrafast/browser.py` (`browser_operation` act branch: scroll-into-view before hit test)
- Modify: `jev_ultrafast/model.py:63` (`action_space` copies the new fields)
- Create: `tests/fixtures/context_page.html`
- Test: `tests/test_snapshot_live.py` (needs Chrome; skipped without it), `tests/test_agent.py` (one new test)

**Interfaces:**
- Consumes: nothing new.
- Produces (element fields in `page["actions"]` and in `action_space` element dicts):
  `in_viewport: bool`, `y: int` (page-coordinate centre), `href: str` (resolved, `""` for non-links),
  `landmark: str` (one of `nav|header|footer|aside|main|form|dialog`), `section: str` (≤ 80 chars, `""` in
  nav/header/footer), `row_text: str` (≤ 80 chars, `""` outside `li`/`tr`), and page field `outline: str`
  (h1–h3 texts joined by `" | "`, ≤ 1000 chars). Task 16 mirrors the `landmark`/`section`/`row_text` rules in
  Python exactly:
  - landmark: nearest ancestor (excluding the element) whose `role` is in
    `{navigation:nav, banner:header, contentinfo:footer, complementary:aside, main:main, form:form, search:form,
    dialog:dialog}` or whose tag is in `{NAV:nav, HEADER:header, FOOTER:footer, ASIDE:aside, MAIN:main, FORM:form,
    DIALOG:dialog}`; role checked before tag at each ancestor; default `main`.
  - section: text of the last `h1`–`h6` that precedes the element in document order (a heading that contains the
    element counts as preceding it), `textContent` with whitespace collapsed, first 80 chars; `""` when landmark
    is `nav`, `header` or `footer`.
  - row_text: nearest ancestor `li` or `tr`, `textContent` whitespace-collapsed, first 80 chars.

- [ ] **Step 1: Write the fixture page**

```html
<!-- tests/fixtures/context_page.html -->
<!doctype html>
<html><head><title>Context fixture</title></head>
<body style="margin:0">
  <header><a href="/home">Home</a></header>
  <nav><a href="/comments">comments</a></nav>
  <main>
    <h2>From today's featured article</h2>
    <p><a href="/wiki/Mary_Mallon">Mary Mallon</a> was a cook. <a href="/wiki/Mary_Mallon">Full article...</a></p>
    <table><tr><td>Story one</td><td><a href="/item?id=1">48 comments</a></td></tr></table>
    <div style="height:3000px"></div>
    <h2>External links</h2>
    <ul><li>Archive <a href="/archive">Official archive</a></li></ul>
  </main>
  <footer><a href="/about">About</a></footer>
</body></html>
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_snapshot_live.py
"""Runs snapshot.js in real Chrome. Needs a throwaway Chrome on BU_CDP_URL; skipped otherwise."""

import os
from pathlib import Path

import httpx
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "context_page.html"


def _chrome_up() -> bool:
    url = os.environ.get("BU_CDP_URL")
    if not url:
        return False
    try:
        return httpx.get(url + "/json/version", timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(not _chrome_up(), reason="needs a throwaway Chrome on BU_CDP_URL")


@pytest.fixture(scope="module")
def observed():
    from jev_ultrafast.browser import Browser

    browser = Browser(FIXTURE.as_uri())
    try:
        yield browser, browser.observe(screenshot=False)
    finally:
        browser.close()


def by_label(page, label):
    return next(a for a in page["actions"] if a["label"] == label)


def test_offscreen_elements_are_captured_with_position(observed):
    _, page = observed
    archive = by_label(page, "Official archive")
    assert archive["in_viewport"] is False and archive["y"] > 780
    assert by_label(page, "Mary Mallon")["in_viewport"] is True


def test_context_fields(observed):
    _, page = observed
    assert by_label(page, "Home")["landmark"] == "header" and by_label(page, "Home")["section"] == ""
    assert by_label(page, "comments")["landmark"] == "nav"
    mallon = by_label(page, "Mary Mallon")
    assert (mallon["landmark"], mallon["section"]) == ("main", "From today's featured article")
    assert mallon["href"].endswith("/wiki/Mary_Mallon")
    assert by_label(page, "48 comments")["row_text"] == "Story one48 comments"
    assert by_label(page, "Official archive")["section"] == "External links"
    assert by_label(page, "Official archive")["row_text"] == "Archive Official archive"
    assert by_label(page, "About")["landmark"] == "footer"
    assert page["outline"] == "From today's featured article | External links"


def test_offscreen_target_is_scrolled_into_view_and_clicked(observed):
    browser, page = observed
    target = by_label(page, "Official archive")
    browser.act(target, page)
    assert browser.evaluate("location.pathname") == "/archive" or browser.evaluate("scrollY") > 0
```

Add to `tests/test_agent.py`:

```python
def test_action_space_keeps_context_fields():
    p = page()
    p["actions"][2].update(landmark="nav", section="", row_text="", href="https://example.test/go",
                           in_viewport=False, y=1200, aliases=["Go now"])
    elements, _, _ = model.action_space(p["actions"])
    go = elements[1]
    assert (go["landmark"], go["href"], go["in_viewport"], go["y"], go["aliases"]) == (
        "nav", "https://example.test/go", False, 1200, ["Go now"])
```

- [ ] **Step 3: Run the tests to verify they fail**

Start a throwaway Chrome first (keep it running for the whole plan):

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9333 \
  --user-data-dir="$(mktemp -d)" --headless=new about:blank >/dev/null 2>&1 &
export BU_CDP_URL=http://127.0.0.1:9333
```

Run: `uv run --env-file .env pytest tests/test_snapshot_live.py tests/test_agent.py::test_action_space_keeps_context_fields -v`
Expected: live tests FAIL (`KeyError: 'in_viewport'` or `StopIteration` for the off-screen label);
`test_action_space_keeps_context_fields` FAILS with `KeyError: 'landmark'`.

- [ ] **Step 4: Implement the snapshot changes**

In `jev_ultrafast/snapshot.js`, add these helpers directly after the `role` function:

```js
  const LANDMARK_ROLES={navigation:'nav',banner:'header',contentinfo:'footer',complementary:'aside',
    main:'main',form:'form',search:'form',dialog:'dialog'};
  const LANDMARK_TAGS={NAV:'nav',HEADER:'header',FOOTER:'footer',ASIDE:'aside',MAIN:'main',FORM:'form',
    DIALOG:'dialog'};
  const squash=(t,n)=>(t||'').replace(/\s+/g,' ').trim().slice(0,n);
  const landmark=e=>{
    for (let n=e.parentElement;n;n=n.parentElement) {
      const r=LANDMARK_ROLES[n.getAttribute('role')];
      if (r) return r;
      if (LANDMARK_TAGS[n.tagName]) return LANDMARK_TAGS[n.tagName];
    }
    return 'main';
  };
  // querySelectorAll returns document order, so the last heading seen precedes each element.
  const sections=new Map(); let heading='';
  for (const n of document.body.querySelectorAll('h1,h2,h3,h4,h5,h6,'+selector)) {
    if (/^H[1-6]$/.test(n.tagName)) heading=squash(n.textContent,80);
    if (n.matches(selector)) sections.set(n,heading);
  }
  const rowText=e=>squash(e.parentElement?.closest('li,tr')?.textContent,80);
```

Note: `selector` must be defined before this block — move the `const roles=[...]` and `const selector=...`
declarations above the new helpers if they are below.

Replace the geometry filter and `base` construction inside the element loop:

```js
    const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2, rname=role(e);
    // Whole document: vertical position no longer filters; horizontally hidden carousels still do.
    if (!rname || r.width<=0 || r.height<=0 || x<0 || x>=innerWidth || y+scrollY<0) continue;
    if (rname==='gridcell' && e.querySelector('button,[role="button"]')) continue;
    const mark=landmark(e);
    const base={node:identity(e),role:rname,label:name(e)||rname,
      rect:{x:r.x,y:r.y,w:r.width,h:r.height},
      in_viewport:y>=0 && y<innerHeight, y:Math.round(y+scrollY),
      href:e.tagName==='A' ? e.href : '', landmark:mark,
      section:['nav','header','footer'].includes(mark) ? '' : (sections.get(e)||''),
      row_text:rowText(e)};
```

Change the cap from 250 to 400 (both occurrences):

```js
  const omitted_actions=Math.max(0,actions.length-400);
  actions.splice(400);
```

Add the outline to the returned object:

```js
  const outline=[...document.querySelectorAll('h1,h2,h3')].map(h=>squash(h.textContent,80))
    .filter(Boolean).join(' | ').slice(0,1000);
  return {url:location.href,title:document.title,w:innerWidth,h:innerHeight,text,outline,
    scroll:{y:scrollY,height},actions,marker,page_key,guards,omitted_actions};
```

Note on `semantics`/`marker`: `in_viewport` is part of each action, so scrolling changes the marker. That is
intended (scroll position is already in the marker).

- [ ] **Step 5: Scroll into view before the hit test**

In `jev_ultrafast/browser.py`, inside the `act` JavaScript of `browser_operation`, replace

```js
              const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2;
```

with

```js
              let r=e.getBoundingClientRect();
              if (r.y+r.height/2<0 || r.y+r.height/2>=innerHeight) {
                e.scrollIntoView({block:'center',inline:'nearest',behavior:'instant'});
                r=e.getBoundingClientRect();
              }
              const x=r.x+r.width/2, y=r.y+r.height/2;
```

- [ ] **Step 6: Copy the new fields in `action_space`**

In `jev_ultrafast/model.py`, `action_space`, replace

```python
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
```

with

```python
            element = {k: action[k] for k in ELEMENT_FIELDS if k in action}
```

and add above the function:

```python
ELEMENT_FIELDS = ("role", "value", "checked", "selected", "expanded", "in_viewport", "y", "href", "landmark",
                  "section", "row_text", "aliases")
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run --env-file .env pytest tests/test_snapshot_live.py tests/test_agent.py -v`
Expected: all pass (3 live tests run, not skipped — confirm the summary says `passed`, not `skipped`).
If `test_context_fields` fails only on `row_text`, print the actual value and correct the **test's expected string**
only if the difference is whitespace from the fixture markup (the rule is `textContent`, collapsed); any other
mismatch is an implementation bug.

- [ ] **Step 8: Full suite and lint**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all pass, `All checks passed!`

- [ ] **Step 9: Commit**

```bash
scripts/commit.sh "feat: capture whole page with context fields; scroll targets into view"
```

---

### Task 3: Resolver

When the planner names an element (`target_text`) and exactly one element carries that label, act on it directly,
with no model call. This is the fast path for "Where to?", "48 comments", "Search Wikipedia".

**Files:**
- Create: `jev_ultrafast/resolver.py`
- Test: `tests/test_resolver.py`

**Interfaces:**
- Consumes: element dicts from `model.action_space` (`index`, `label`, optional `aliases`).
- Produces:
  - `normalize(text: str) -> str` — accent-folded, casefolded, punctuation → space, whitespace collapsed.
  - `resolve(target_text: str, elements: Sequence[Mapping]) -> tuple[Mapping | None, str | None]` — returns
    `(element, "resolver")` for a unique exact match, `(element, "resolver_fuzzy")` for a unique word-subset
    match, else `(None, None)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_resolver.py
from jev_ultrafast.resolver import normalize, resolve


def el(index, label, aliases=()):
    return {"index": str(index), "label": label, "aliases": list(aliases)}


def test_normalize_folds_accents_case_and_punctuation():
    assert normalize("  Kurt GÖDEL's  theorems! ") == "kurt godel s theorems"
    assert normalize("Where to?") == "where to"


def test_unique_exact_match_wins():
    elements = [el(1, "Where from?"), el(2, "Where to?"), el(3, "Departure")]
    assert resolve("where to", elements) == (elements[1], "resolver")


def test_alias_counts_as_exact():
    elements = [el(1, "Mary Mallon", ["Full article..."]), el(2, "Read")]
    assert resolve("Full article...", elements) == (elements[0], "resolver")


def test_accented_target_matches_plain_label_and_back():
    elements = [el(1, "Kurt Gödel"), el(2, "Zurich")]
    assert resolve("Kurt Godel", elements)[0] is elements[0]
    assert resolve("Zürich", elements)[0] is elements[1]


def test_repeated_exact_label_defers_to_the_actor():
    elements = [el(1, "Edit"), el(2, "Edit"), el(3, "Save")]
    assert resolve("Edit", elements) == (None, None)


def test_unique_word_subset_is_fuzzy():
    elements = [el(1, "Search Wikipedia"), el(2, "Donate")]
    assert resolve("Search", elements) == (elements[0], "resolver_fuzzy")


def test_ambiguous_fuzzy_or_empty_target_defers():
    elements = [el(1, "48 comments"), el(2, "17 comments")]
    assert resolve("comments", elements) == (None, None)
    assert resolve("  ", elements) == (None, None)
    assert resolve("Nothing like it", elements) == (None, None)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_resolver.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jev_ultrafast.resolver'`

- [ ] **Step 3: Implement**

```python
# jev_ultrafast/resolver.py
"""Deterministic element lookup: when the planner names exactly one element, no model call is needed."""

import re
import unicodedata
from collections.abc import Mapping, Sequence

_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(_PUNCT.sub(" ", plain.casefold()).split())


def _names(element: Mapping) -> list[str]:
    return [normalize(n) for n in [element.get("label", ""), *element.get("aliases", [])]]


def resolve(target_text: str, elements: Sequence[Mapping]) -> tuple[Mapping | None, str | None]:
    want = normalize(target_text)
    if not want:
        return None, None
    exact = [e for e in elements if want in _names(e)]
    if len(exact) == 1:
        return exact[0], "resolver"
    if exact:
        return None, None  # the same label twice: only the actor can tell them apart
    words = set(want.split())
    fuzzy = [e for e in elements if any(words <= set(name.split()) for name in _names(e))]
    if len(fuzzy) == 1:
        return fuzzy[0], "resolver_fuzzy"
    return None, None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_resolver.py -v && uv run ruff check jev_ultrafast/resolver.py tests/test_resolver.py`
Expected: 7 passed; `All checks passed!`

- [ ] **Step 5: Commit**

```bash
scripts/commit.sh "feat: resolve uniquely named elements without a model call"
```

---

### Task 4: StepMemory

Records what was tried on each page and what happened, so actions that did nothing, or were already repeated,
are not offered again. Replaces the fingerprint-only loop rule that missed four repeated "View source" clicks.

**Files:**
- Create: `jev_ultrafast/memory.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: `resolver.normalize`; agent history entries (dicts with `operation`, `kind`, `action` (label), `text`,
  `url` (after), `url_before` (added in Task 10; falls back to `url`), `page_changed`).
- Produces:
  - `Attempt` frozen dataclass: `url: str` (fragment removed), `operation: str`, `label: str` (normalised),
    `value: str`, `outcome: str` (`url_changed|page_changed|no_change`).
  - `StepMemory()` with `sync(history) -> None` (ingests entries not yet seen), `attempts -> tuple[Attempt, ...]`,
    `excluded(url: str) -> frozenset[tuple[str, str]]` of `(operation, normalised label)`,
    `failed(url: str) -> list[str]` (human-readable, for the planner), `last_failed() -> bool`,
    `streak_without_url_change() -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_memory.py
from jev_ultrafast.memory import StepMemory

P = "https://en.wikipedia.org/wiki/Main_Page"


def entry(label, op="CLICK", before=P, after=P, changed=True, text=None):
    return {"operation": op, "kind": "click", "action": label, "text": text,
            "url_before": before, "url": after, "page_changed": changed}


def test_sync_ingests_only_new_entries_and_classifies_outcomes():
    history = [entry("View source", after=P + "?action=edit"), entry("Read", changed=False)]
    m = StepMemory()
    m.sync(history)
    m.sync(history)
    assert [a.outcome for a in m.attempts] == ["url_changed", "no_change"]
    history.append(entry("Talk", changed=True))
    m.sync(history)
    assert [a.outcome for a in m.attempts] == ["url_changed", "no_change", "page_changed"]


def test_no_change_click_is_excluded_on_that_page_only():
    m = StepMemory()
    m.sync([entry("Read", changed=False)])
    assert m.excluded(P) == frozenset({("CLICK", "read")})
    assert m.excluded("https://en.wikipedia.org/wiki/Other") == frozenset()
    assert m.excluded(P + "#History") == frozenset({("CLICK", "read")})
    assert m.failed(P) == ["CLICK Read (no effect)"]


def test_typing_without_page_change_is_not_a_failure():
    m = StepMemory()
    m.sync([entry("Search Wikipedia", op="TYPE_TEXT", changed=False, text="Ada")])
    assert m.excluded(P) == frozenset() and not m.last_failed()


def test_action_repeated_twice_is_excluded_even_when_it_changed_the_page():
    m = StepMemory()
    m.sync([entry("Toggle References subsection"), entry("Toggle References subsection")])
    assert ("CLICK", "toggle references subsection") in m.excluded(P)
    assert "CLICK Toggle References subsection (repeated)" in m.failed(P)


def test_streak_counts_trailing_actions_without_url_change():
    m = StepMemory()
    m.sync([entry("A", after=P + "#x"), entry("B"), entry("C", changed=False), entry("D")])
    assert m.streak_without_url_change() == 3
    assert m.last_failed() is False
    m.sync([entry("A", after=P + "#x"), entry("B"), entry("C", changed=False), entry("D"),
            entry("E", changed=False)])
    assert m.last_failed() is True


def test_missing_url_before_falls_back_to_url():
    m = StepMemory()
    m.sync([{"operation": "CLICK", "action": "Go", "url": P, "page_changed": False}])
    assert m.attempts[0].url == P and m.attempts[0].outcome == "no_change"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_memory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jev_ultrafast.memory'`

- [ ] **Step 3: Implement**

```python
# jev_ultrafast/memory.py
"""What the agent already tried on each page, so failed or repeated actions are not offered again."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urldefrag

from .resolver import normalize

REPEAT_LIMIT = 2


@dataclass(frozen=True)
class Attempt:
    url: str
    operation: str
    label: str
    display: str
    value: str
    outcome: str  # url_changed | page_changed | no_change


def _outcome(entry: Mapping) -> str:
    before = entry.get("url_before")
    if before is not None and entry.get("url") != before:
        return "url_changed"
    return "page_changed" if entry.get("page_changed") else "no_change"


def _attempt(entry: Mapping) -> Attempt:
    url = urldefrag(entry.get("url_before") or entry.get("url") or "")[0]
    operation = entry.get("operation") or str(entry.get("kind", "")).upper()
    label = str(entry.get("action") or "")
    return Attempt(url, operation, normalize(label), label, entry.get("text") or "", _outcome(entry))


def _is_failure(attempt: Attempt) -> bool:
    # Typing rarely changes the page by itself; its effect is inside the field.
    return attempt.outcome == "no_change" and attempt.operation != "TYPE_TEXT"


class StepMemory:
    def __init__(self) -> None:
        self._attempts: tuple[Attempt, ...] = ()

    @property
    def attempts(self) -> tuple[Attempt, ...]:
        return self._attempts

    def sync(self, history: Sequence[Mapping]) -> None:
        new = tuple(_attempt(h) for h in history[len(self._attempts):])
        self._attempts = (*self._attempts, *new)

    def _on(self, url: str) -> list[Attempt]:
        page = urldefrag(url)[0]
        return [a for a in self._attempts if a.url == page]

    def _reasons(self, url: str) -> dict[tuple[str, str], tuple[str, str]]:
        here = self._on(url)
        counts = Counter((a.operation, a.label) for a in here)
        reasons: dict[tuple[str, str], tuple[str, str]] = {}
        for a in here:
            key = (a.operation, a.label)
            if _is_failure(a):
                reasons[key] = (a.display, "no effect")
            elif counts[key] >= REPEAT_LIMIT and key not in reasons:
                reasons[key] = (a.display, "repeated")
        return reasons

    def excluded(self, url: str) -> frozenset[tuple[str, str]]:
        return frozenset(self._reasons(url))

    def failed(self, url: str) -> list[str]:
        return [f"{op} {display} ({why})" for (op, _), (display, why) in self._reasons(url).items()]

    def last_failed(self) -> bool:
        return bool(self._attempts) and _is_failure(self._attempts[-1])

    def streak_without_url_change(self) -> int:
        streak = 0
        for a in reversed(self._attempts):
            if a.outcome == "url_changed":
                break
            streak += 1
        return streak
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_memory.py -v && uv run ruff check jev_ultrafast/memory.py tests/test_memory.py`
Expected: 6 passed; `All checks passed!`

- [ ] **Step 5: Commit**

```bash
scripts/commit.sh "feat: add step memory that excludes failed and repeated actions"
```

---

### Task 5: Tools — search registry, GOTO validation, SCROLL_TO_TEXT, navigate

Two actions the planner can take besides element actions: scroll to text (reaches headings like "External links"
without guessing scroll distance) and go to a registered site-search URL (skips the three error-prone steps of
typing, picking a suggestion and submitting). GOTO is only allowed for URLs that match a registry template.

**Files:**
- Create: `jev_ultrafast/tools.py`
- Modify: `jev_ultrafast/browser.py` (`Browser.__init__` load loop → `_load`; new `navigate`, `scroll_to_text`)
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `Browser` instance.
- Produces:
  - `SEARCH_TEMPLATES: dict[str, str]` (host → template with `{q}`).
  - `search_template(page_url: str) -> str | None`
  - `is_allowed_goto(url: str) -> bool`
  - `run_tool(browser, operation: str, arg: str) -> bool` — `GOTO` navigates (raises `ValueError` if not
    allowed) and returns `True`; `SCROLL_TO_TEXT` returns whether text was found and scrolled to; other
    operations raise `ValueError`.
  - `Browser.navigate(url: str) -> None`, `Browser.scroll_to_text(text: str) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools.py
from unittest.mock import Mock

import pytest

from jev_ultrafast import tools
from jev_ultrafast.browser import Browser

WIKI_SEARCH = "https://en.wikipedia.org/w/index.php?search=Ada+Lovelace&title=Special%3ASearch&go=Go"


def test_search_template_by_host():
    assert tools.search_template("https://en.wikipedia.org/wiki/Main_Page").startswith("https://en.wikipedia.org/")
    assert tools.search_template("https://news.ycombinator.com/") is None


def test_goto_allows_only_rendered_registry_templates():
    assert tools.is_allowed_goto(WIKI_SEARCH)
    assert tools.is_allowed_goto("https://github.com/search?q=browser-use&type=repositories")
    assert not tools.is_allowed_goto("https://en.wikipedia.org/wiki/Ada_Lovelace")
    assert not tools.is_allowed_goto("https://evil.test/w/index.php?search=x&title=Special%3ASearch&go=Go")
    assert not tools.is_allowed_goto(WIKI_SEARCH + "&extra=1")
    assert not tools.is_allowed_goto("https://en.wikipedia.org/w/index.php?search=&title=Special%3ASearch&go=Go")


def test_run_tool_dispatches_and_rejects():
    browser = Mock(scroll_to_text=Mock(return_value=True))
    assert tools.run_tool(browser, "GOTO", WIKI_SEARCH) is True
    browser.navigate.assert_called_once_with(WIKI_SEARCH)
    assert tools.run_tool(browser, "SCROLL_TO_TEXT", "External links") is True
    with pytest.raises(ValueError, match="not a registered search URL"):
        tools.run_tool(browser, "GOTO", "https://en.wikipedia.org/wiki/X")
    with pytest.raises(ValueError, match="Unknown tool"):
        tools.run_tool(browser, "CLICK", "x")


def test_browser_scroll_to_text_passes_the_needle_as_json():
    b = Browser.__new__(Browser)
    b.evaluate = Mock(return_value=True)
    assert b.scroll_to_text('External "links"') is True
    assert '("External \\"links\\"")' in b.evaluate.call_args.args[0]


def test_browser_navigate_waits_for_load():
    b = Browser.__new__(Browser)
    b.call = Mock()
    b.evaluate = Mock(side_effect=["loading", "complete"])
    b.navigate("https://en.wikipedia.org/wiki/X")
    b.call.assert_called_once_with("Page.navigate", url="https://en.wikipedia.org/wiki/X")
    assert b.evaluate.call_count == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jev_ultrafast.tools'`

- [ ] **Step 3: Add `_load`, `navigate`, `scroll_to_text` to `Browser`**

In `jev_ultrafast/browser.py`, replace the navigation/wait block at the end of `Browser.__init__`

```python
        self.call("Page.navigate", url=url)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.evaluate("document.readyState") == "complete":
                break
            time.sleep(0.02)
```

with

```python
        self.navigate(url)
```

and add these methods to `Browser` (after `evaluate`):

```python
    def navigate(self, url: str) -> None:
        self.call("Page.navigate", url=url)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.evaluate("document.readyState") == "complete":
                break
            time.sleep(0.02)

    def scroll_to_text(self, text: str) -> bool:
        return bool(self.evaluate(f"{SCROLL_TO_TEXT}({json.dumps(text)})"))
```

and this module constant after `MARKER`:

```python
# Headings first (the "External links" heading, not its table-of-contents entry), then any visible text.
SCROLL_TO_TEXT = """(needle => {
  const want=needle.toLowerCase().replace(/\\s+/g,' ').trim();
  if (!want) return false;
  const shown=e=>e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
  const has=e=>e.textContent.toLowerCase().replace(/\\s+/g,' ').includes(want);
  const go=e=>{e.scrollIntoView({block:'center',behavior:'instant'});return true;};
  for (const h of document.querySelectorAll('h1,h2,h3,h4,h5,h6')) if (has(h) && shown(h)) return go(h);
  const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
  let n;
  while ((n=walker.nextNode())) {
    const p=n.parentElement;
    if (p && !p.closest('script,style,noscript,template') && has(n.parentElement) && shown(p)) return go(p);
  }
  return false;
})"""
```

- [ ] **Step 4: Implement `tools.py`**

```python
# jev_ultrafast/tools.py
"""Non-element actions the planner may take: scroll to text, and go to a registered site-search URL."""

import re
from urllib.parse import urlsplit

# Sites with a stable GET search. The planner may only navigate to a rendering of one of these.
SEARCH_TEMPLATES = {
    "en.wikipedia.org": "https://en.wikipedia.org/w/index.php?search={q}&title=Special%3ASearch&go=Go",
    "github.com": "https://github.com/search?q={q}&type=repositories",
}
_ALLOWED = [re.compile(re.escape(t).replace(re.escape("{q}"), r"[^&#]+")) for t in SEARCH_TEMPLATES.values()]


def search_template(page_url: str) -> str | None:
    return SEARCH_TEMPLATES.get(urlsplit(page_url).hostname or "")


def is_allowed_goto(url: str) -> bool:
    return any(pattern.fullmatch(url) for pattern in _ALLOWED)


def run_tool(browser, operation: str, arg: str) -> bool:
    if operation == "GOTO":
        if not is_allowed_goto(arg):
            raise ValueError(f"GOTO target is not a registered search URL: {arg!r}; not navigating.")
        browser.navigate(arg)
        return True
    if operation == "SCROLL_TO_TEXT":
        return browser.scroll_to_text(arg)
    raise ValueError(f"Unknown tool operation {operation!r}")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tools.py tests/test_agent.py -v && uv run ruff check .`
Expected: all pass; `All checks passed!`

- [ ] **Step 6: Live check of scroll_to_text** (Chrome from Task 2 running)

```bash
uv run --env-file .env python - <<'EOF'
from pathlib import Path
from jev_ultrafast.browser import Browser
b = Browser(Path("tests/fixtures/context_page.html").resolve().as_uri())
print(b.scroll_to_text("External links"), b.evaluate("scrollY") > 1000, b.scroll_to_text("no such text"))
b.close()
EOF
```

Expected: `True True False`

- [ ] **Step 7: Commit**

```bash
scripts/commit.sh "feat: add scroll-to-text and registry-checked search navigation tools"
```
