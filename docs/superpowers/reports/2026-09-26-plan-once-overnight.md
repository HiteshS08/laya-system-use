# Plan once per page: overnight report

**Date:** 2026-09-26. **Branch:** `feat/plan-once-per-page` (from `main` at `9b0300b`). **Brief:**
`docs/superpowers/briefs/2026-09-26-plan-once-brief.md`.

No live evaluation ran. This environment has no MLX, planner server, Chrome session or Laya checkpoint. Every number
below comes from unit tests and a scripted replay that uses fakes. None of it shows that the live pass rate
improved. The live Gate A+B re-run (commands at the end) is still needed.

Commits (both made with `scripts/commit.sh`, which pushed without trouble):
- `ad3d407` fix(planner): show focus_text around the last found scroll target so below-fold completion is visible
- `b084f49` feat(pilot): plan once per page with done_when completion, a retry before replanning, and 5-step plans
- this report

## Work 1: below-fold fix

Tests came first; each failed before its change.

- **`planner.focus_text(text, target)`.** Returns up to `FOCUS_CHARS` = 300 characters of the page's visible text,
  centred on the first case-insensitive occurrence of the scroll target. Whitespace between words may differ. If
  the target is absent it returns `""`.
- **`planner_view(..., focus="")`.** When the focus text is non-empty, the view gains a `focus_text` field. Its
  characters come out of `visible_text`'s 600, so `visible_text` shrinks to about 300 whenever `focus_text` is
  present. Without that trade, the request would break the 4,800-character budget.
- **`plan(..., focus="")`.** Passes the target through to the view.
- **`StepMemory.last_found_scroll(url)`.** Returns the target of the latest `SCROLL_TO_TEXT` on this page (fragment
  ignored) that reported `tool_ok=True`. `Pilot._replan` passes it as `focus=`. A failed scroll is never used.
- **Prompt.** It describes `focus_text` and allows evidence to be quoted "from visible_text/focus_text/title".
- **Evidence check confirmed.** `parse_plan` → `_shown` searches the full `page["text"]` and the title, not the
  view. A new test puts the quote at character 3,000 and it is accepted.
- **Budget.** The worst-case budget test still passes, and a new variant with `focus_text` also passes.

In the replay, the traced below-fold failure reproduces on `main`: `turing_external_links` makes 8 planner calls
and ends BLOCKED. After Work 1 it takes 2 calls and ends DONE.

## Work 2: call the planner far less

The spec section came first, then tests, then code.

1. **Deterministic completion (`done_when`).**
   - `Plan` has a new field, `done_when: str = ""`, parsed only for continue plans.
   - A non-string, or a phrase over `DONE_WHEN_CHARS` = 80 characters, raises `ValueError`, the same way other
     invalid fields do. That uses the existing single retry.
   - A phrase the current page already shows is dropped (set to `""`, logged), because its appearance can never
     signal a change. So is a phrase with no letters or digits.
   - `planner.shows_phrase(phrase, page)` normalizes both sides with `resolver.normalize` and matches whole words
     in the full text or the title.
   - `Pilot.decide` checks the phrase first, after absorbing the latest action and before the stall check or any
     planner call. It needs at least one action since the current plan.
   - When the phrase is found, the Pilot returns `DONE`, with the phrase as evidence and route `done_when`.
   - The phrase stays active across URL changes until a new plan replaces it. The multi-page savings depend on this.
2. **Retry before replanning.**
   - Each decision is now a frozen `Pending` record: history index, step, chosen element, the actor's ranking, and
     whether it is a retry.
   - When an element step had no effect (`memory.is_failure`) and its decision carried an actor ranking (routes
     `actor`, `planner_pick`), the next decision retries the same step. It uses the highest-ranked other candidate
     that is still usable, after StepMemory exclusions. The route is `retry` and there is no model call (no actor
     call either).
   - If that also has no effect, or there is no usable alternative, the Pilot replans.
   - These cases have no alternative and replan as before: a resolver-routed step, a tool step, a sole candidate,
     or a step with no routable element.
3. **Replan triggers.** `_needs_plan` now fires only on a URL change (fragment included), an empty queue, or the
   `_must_replan` flag set when the retry rule is exhausted. It no longer fires on "the last action failed".
4. **Longer plans.**
   - `MAX_STEPS` goes from 3 to 5.
   - The prompt asks for "every step you can foresee on this page, up to one that loads a new page". This merges the
     old "Stop after loading a new page" rule into the steps description.
   - `PLAN_MAX_TOKENS` goes from 256 to 384. That is an estimate: five steps at their field caps plus `done_when`
     would not fit in 256. I did not observe the truncation.

**Supporting change.** `scripts/live_summary.py` now reports `planner_calls` and `planner_calls_per_url`, so the
live re-run measures the same target as the replay. `Pilot.plans` entries also record `done_when`.

**Prompt compression (a behaviour risk; see below).** The new fields pushed the system prompt past the budget.
Rather than cut the element, outline or text budgets, I shortened wording:
- "instruction: imperative, names the element, <=12 words"
- "If the goal's item isn't an element here, search for it first"
- the GOTO rule in one line
- evidence no longer repeats "showing goal met"; the done rule still says it.

No rule was removed. The worst case is now 4,797 of 4,800 characters with `focus_text` and 4,781 without, leaving
3 characters of headroom.

## Spec section added

This was added to `docs/superpowers/specs/2026-09-23-planner-actor-design.md` as **5.10 Planner call budget**:

> Three live runs spent most of their time in the planner (median call 11.6–15.2 s against a 3 s target), and the
> planner was re-called after every failed step and every emptied queue. Target: on average ≤ 1 planner call per
> distinct URL in scripted Pilot scenarios (`scripts/count_plan_calls.py`). This section supersedes the "Runs:"
> line of 5.2 and the "1–3 steps" limit.
>
> - **Longer plans.** `MAX_STEPS` = 5. The planner is told to plan every step it can foresee on the current page,
>   up to one that loads a new page. The plan's `max_tokens` rises from 256 to 384 to leave room for five steps.
> - **Deterministic completion.** A continue plan may carry `done_when`: a phrase of at most 80 characters that the
>   page text or title will show once the goal is met (ideally the final page's title). It is validated like the
>   other fields (non-string or over 80 characters rejects the plan); a phrase the current page already shows, or
>   one with no letters or digits, is dropped, since it cannot signal a change. After each executed action, before
>   any planner call, the Pilot looks for the phrase with `resolver.normalize` (whole words) in the full page text
>   and the title. If found, the Pilot returns DONE with the phrase as evidence (route `done_when`) and makes no
>   planner call. The phrase stays active across URL changes until a new plan replaces it.
> - **Retry before replanning.** When an element step had no effect and it was routed through the actor
>   (`actor` or `planner_pick`), it is retried once on the actor's next-ranked candidate from that same decision
>   that is still usable (not excluded by StepMemory), route `retry`, no model call. Replan only when the retry
>   also has no effect, or when there is no alternative (a resolver-routed step, a tool step, a sole candidate,
>   or no usable next candidate).
> - **Replan triggers.** Exactly three: a URL change (fragment included); a queue exhausted without `done_when`
>   being met; the retry rule exhausted. A step with no routable element counts as "no alternative" and replans
>   as before. Nothing else triggers a replan.
> - **View.** After a successful SCROLL_TO_TEXT on the current page, the planner view adds `focus_text`: about 300
>   characters of visible text centred on the first case-insensitive occurrence of the scroll target, taken out of
>   `visible_text`'s 600 so the request budget is unchanged. Evidence may quote it; the evidence check itself
>   always searches the full page text and title.

## Tests and coverage

Commands: `uv sync`, `uv run pytest -q`, `uv run ruff check .`, `node --check jev_ultrafast/static/app.js` and
`uv build`. All pass.

- **pytest:** 301 passed, 8 skipped (the live-browser tests, which skip without Chrome as expected). `main` had 274
  passed.
- **Coverage** (`--cov=jev_ultrafast.pilot --cov=jev_ultrafast.planner --cov=jev_ultrafast.memory`):

  | Module | `main` | Now | Missing now |
  | --- | ---: | ---: | --- |
  | `memory.py` | 100% | 100% | none |
  | `pilot.py` | 96% | 97% | 204, 208–212: the SELECT option path, already uncovered on `main` |
  | `planner.py` | 98% | 98% | 3 lines, already uncovered on `main` |

- **`tests/test_pilot.py`:** every original test is unchanged and passes. `git diff origin/main --
  tests/test_pilot.py` removes no lines.
- **`tests/test_planner.py`:** two existing tests changed. The step cap moved from 3 to 5, and `max_tokens` now
  asserts `PLAN_MAX_TOKENS`.
- **New tests:**
  - focus text: centring, absent target, near the start, the prompt wording, evidence beyond the view, the budget
    with focus, the Pilot passing the last found scroll, a failed scroll never used as focus;
  - `last_found_scroll`;
  - `done_when`: kept and stripped, invalid values rejected, already-shown or wordless phrases dropped,
    whole-word matching, the prompt wording, DONE with no planner call, no check before an action, a replan when
    the queue empties without the phrase;
  - retry: on the next-ranked candidate, a failed retry replans, a same-label sibling is skipped, no usable
    alternative replans, an unexecuted retry is re-offered;
  - a found scroll with steps still queued does not replan;
  - planner calls per URL in `live_summary`;
  - the replay target.
- Every new or changed function is under 50 lines.

## Planner calls before and after

`scripts/count_plan_calls.py` replays six scripted scenarios through the real `Pilot` and the real `planner.plan`.
It fakes three things:
- the planner model, as an oracle behind `plan`'s `complete` hook;
- the actor, which uses a fixed label order;
- the browser, as a small page world.

It counts `plan_fn` calls and the distinct URLs each run saw, with the same 12-step cap as `live_eval.py`. It uses
only entry points that `main` also has, so the "before" figure is the same script run against a `main` checkout:
`git worktree add ../main-wt origin/main && PYTHONPATH=../main-wt uv run python scripts/count_plan_calls.py`.
`tests/test_plan_calls.py` pins the "after" numbers.

| Scenario | URLs | `main` | After Work 1 | After Work 2 |
| --- | ---: | ---: | ---: | ---: |
| `wiki_search_ada` (type, search, article) | 2 | 2 | 2 | **1** |
| `flights_form` (4 fields, results page) | 2 | 3 | 3 | **1** |
| `turing_external_links` (below-fold scroll) | 1 | 8, ends BLOCKED | 2 | **2** |
| `mallon_to_typhoid` (two links) | 3 | 3 | 3 | **2** |
| `gh_issues_retry` (actor's first pick has no effect) | 2 | 3 | 3 | **1** |
| `ada_references` (TOC link, fragment URL) | 2 | 2 | 2 | **2** |
| **Total** | **12** | **21 (1.75/URL)** | **15 (1.25/URL)** | **9 (0.75/URL)** |
| Mean of per-scenario calls/URL | | 2.33 | 1.33 | **0.86** |

After Work 2 every scenario ends DONE. The target (≤ 1 planner call per distinct URL on average) is met by both
measures. The router's `pick` was never called in any configuration, and the fakes never produce invalid JSON, so
planner model calls equal `plan_fn` calls here.

**What these numbers depend on:** I wrote the scenarios and the oracle.
- In four scenarios the oracle's `done_when` is the final page's title, which is what the prompt asks for.
- In the two Wikipedia in-page scenarios (`turing_external_links`, `ada_references`) the oracle gives the naive
  phrase ("External links", "References"). The table of contents already shows that phrase, so the drop rule
  removes it and those tasks still take 2 calls.
- When nothing is left to do and no evidence is visible, the oracle repeats its last step. The Gate A+B re-run
  report traced the live planner doing exactly that.

A live Qwen3-4B will not follow the prompt as faithfully as this oracle.

## Open risks

1. **`done_when` false positives.**
   - The phrase is checked on every page after the plan, including intermediate pages. It is also checked against
     visible text that may include autocomplete dropdowns.
   - Example: a naive phrase such as "Ada Lovelace" could fire on a suggestion list after typing. "Typhoid fever"
     could fire on the Mary Mallon article.
   - Mitigations: phrases the planning page already shows are dropped, matching is whole-word, and the prompt asks
     for the final page's title.
   - The live suite's independent checks will expose any false DONE. A stricter rule (title only, or re-validating
     on each new URL) would give up part of the multi-page saving.
2. **The retry is not τ-gated.** The next-ranked candidate can have low probability, and clicking it can navigate
   to an unrelated page. That is the drift pattern from both Gate A+B reports. Gating it at τ would make retries
   rare, because the second candidate is usually well below 0.5.
3. **The most common route never retries.** Resolver-routed failures (90 of 246 routed decisions in the re-run)
   replan exactly as before. The Wikipedia search-box actuation gap is unaffected.
4. **Longer outputs may raise per-call latency.** Five steps and `max_tokens` 384 mean more generated tokens per
   call, so median planner latency could rise even as the number of calls falls.
   - Later steps may name elements that only appear after earlier steps, such as suggestions. Those become
     unroutable and replan, which is the existing behaviour.
   - Wall time per action is the metric to watch.
5. **The prompt changed without a live test.** The compressed rule wording has not been run against the model. The
   budget has 3 characters of headroom, so any further prompt text needs cuts elsewhere.
6. **Less visible text with `focus_text`.** When `focus_text` is present, the start of the viewport text drops from
   600 to about 300 characters.
7. **Not addressed:** the leading failure in the re-run, where the planner emits a bare string instead of a step
   object, is outside this brief. Invalid plans still fall back to the actor-only path.
8. **Pre-existing, found while testing, not fixed:** a focus click executes the action labelled "Open Search", but
   the element's label is "Search". StepMemory therefore never excludes that element's CLICK after it has no
   effect.
9. **Dead code:** `StepMemory.last_failed` is no longer used by the Pilot. It is kept, with its tests.

## Live Gate A+B re-run (run locally on the Mac)

1. Start a throwaway headless Chrome:
   `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9333 --user-data-dir="$(mktemp -d)" --headless=new about:blank &`
2. Start the model server:
   `uv run mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8080 &`
3. Run the suite:
   `POLICY_BACKEND=planner ACTOR_TAU=0.5 uv run --env-file .env python scripts/live_eval.py`
4. Summarise:
   `uv run python scripts/live_summary.py artifacts/live/<run>`

The summary now includes `planner_calls` and `planner_calls_per_url`, next to `median_planner_ms`,
`wall_s_per_action` and the route counts (new routes: `done_when`, `retry`).
