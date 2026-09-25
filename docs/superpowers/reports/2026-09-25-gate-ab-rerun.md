# Gate A+B re-run report

**Date:** 2026-09-25. **Configuration:** `POLICY_BACKEND=planner`, `ACTOR_TAU=0.5`, `ACTOR_CONTEXT=0`,
`LAYA_CHECKPOINT=checkpoints/laya_browser_mind2web`, planner model `mlx-community/Qwen3-4B-Instruct-2507-4bit`
served locally via `mlx_lm.server`, throwaway headless Chrome. One run per task, no retries, same machine
(under ~10 GB swap pressure throughout). Raw annotated records: [`artifacts/live/20260925T124019Z/`](../../../artifacts/live/20260925T124019Z/)
(gitignored, tagged in place). Baseline: [`artifacts/live/20260923T175703Z/`](../../../artifacts/live/20260923T175703Z/),
[`baseline-2026-09-23.md`](../../../artifacts/live/baseline-2026-09-23.md). First Gate A+B run:
[`artifacts/live/20260925T040642Z/`](../../../artifacts/live/20260925T040642Z/),
[`2026-09-24-gate-ab.md`](2026-09-24-gate-ab.md). Structural fixes made before this run: Task 15b
(`.superpowers/sdd/2026-09-24-planner-actor/task-15b-report.md`).

## Result

**7/25 tasks passed (28%; n=25)** this re-run, versus **8/25 (32%)** on the first Gate A+B run and **7/25 (28%)**
on the baseline. Per the plan's honesty rule, **differences under ~15 points (4 tasks) are not distinguishable**
on this 25-task suite — all three runs (7, 8, 7) sit inside that band and are statistically indistinguishable
from each other.

**Gate verdict: MISSED.** Gate A+B requires ≥15/25 (60%). This run reached 7/25 (28%), 8 tasks short of gate.
Per protocol, no further tuning proceeds without a decision from the user.

## Category table (baseline / first run / this re-run)

| Category | Baseline | First run | This re-run |
| --- | ---: | ---: | ---: |
| Site search | 1/5 | 2/5 | 4/5 |
| In-page navigation | 2/4 | 1/4 | 1/4 |
| Repeated-label disambiguation | 0/4 | 0/4 | 0/4 |
| Multi-field forms | 0/3 | 0/3 | 0/3 |
| Below-fold targets | 4/5 | 5/5 | 1/5 |
| Multi-page flows | 0/4 | 0/4 | 1/4 |

Site search improved 2 tasks over the first run (the new `search_fallback` route and the field-confusion repair
both helped there). Below-fold targets collapsed from 5/5 to 1/5 — a new failure mode, not a regression of the
15b fixes themselves (see "New failure mode" below). Every category move here is at most 4 tasks, i.e. inside
the "not distinguishable" band on its own, and the overall score (7) is a wash against both prior runs.

## Per-task table

| Task | Category | Result | Main observation |
| --- | --- | --- | --- |
| `wiki_featured` | disambiguation | Fail | Clicked an image in the blurb, then wandered through a user page, "Articles", and a "Hide Appearance" UI control; never reached the featured article. |
| `wiki_search` | site_search | Pass | Search box focus click correctly kept the queued TYPE_TEXT/CLICK steps; reached Gödel's incompleteness theorems in 3 steps. |
| `wiki_long_page` | below_fold | Pass | Search-fallback GOTO recovered from a planner failure on the Kurt Gödel page and still reached the target. |
| `hn_comments` | disambiguation | Fail | Clicked the site-wide "comments" nav link twice (global `/newcomments`), never the top story's own comments; agent wrongly reported `done`. |
| `gh_issues` | in_page_navigation | Pass | Done in one step. |
| `flights` | multi_field_form | Fail, blocked | Destination (London) and its date field got confused with each other; origin (Zürich) was never set; hit the step-cap loop guard. |
| `wiki_search_turing` | site_search | Pass | Search box focus → type → search button, no drop, no drift. |
| `wiki_search_ada` | site_search | Pass | Reached Ada Lovelace via the featured-article/search path. |
| `wiki_search_python` | site_search | Pass | Reached Python (programming language) article. |
| `wiki_search_mallon` | site_search | Fail | Reached Mary Mallon twice (once directly, once via search fallback) but a "Mary Mallen" typo-link mis-click was the final state. |
| `turing_references` | in_page_navigation | Fail | `SCROLL_TO_TEXT References` found the heading correctly at step 3, then subsequent clicks fell into the article's image gallery and drifted to an unrelated "Peter Hogg" article. |
| `ada_references` | in_page_navigation | Fail | Found References twice via scroll, but intervening clicks dragged the run onto an external Wikisource page it never left. |
| `python_references` | in_page_navigation | Fail | First click ("reference") landed on "Pointer (computer programming)" instead of the article's own References section; looped between "Reference (computer science)" and "Wild reference" for the rest of the run. |
| `hn_second_comments` | disambiguation | Fail | "show" was resolved to the Show HN listing page, not "second story"; ended on an unrelated GitHub page. |
| `hn_third_comments` | disambiguation | Fail | Same "show"/"new"/"newest" confusion as first-run's third-story task; 11 steps cycling HN sub-pages without isolating story 3. |
| `flights_paris_rome` | multi_field_form | Fail, blocked | Origin (Paris) and destination (Rome) were both set correctly by step 5, but the run never touched the date field and looped re-clicking "Flights"/reopening origin until the step cap. |
| `flights_mumbai_delhi` | multi_field_form | Fail | Destination autocomplete picked "Delhi, New York, USA" instead of New Delhi, India; origin (Mumbai) was set correctly but a date typed into the "Where from?" location field, not a date field. |
| `turing_external_links` | below_fold | Fail | Found "External links" at step 1, then a stray link click opened the image gallery and 7 steps cycled next/previous image without leaving it. |
| `ada_external_links` | below_fold | Fail | Same gallery-loop failure as `turing_external_links`. |
| `python_external_links` | below_fold | Fail, blocked | Found "External links" at step 1, then two more scrolls with slightly different wording ("External links section heading" vs "External links") re-ran the same scroll a 3rd time before falling into a show/hide toggle loop. |
| `godel_external_links` | below_fold | Fail, blocked | Found "External links" at step 1, then a show/hide collapsible-toggle loop to the step cap. |
| `turing_to_award` | multi_page_flow | Fail, crash | Reached `Alan_Turing` correctly in 3 clean steps (focus, type, search) — then the browser connection raised `RuntimeError: no close frame received or sent` before the agent could act again. Not a planner/actor/resolver/router mistake; a genuine site/transport failure mid-run. |
| `ada_to_babbage` | multi_page_flow | Fail | Never reached Ada Lovelace's article at all — drifted through Wikipedia meta-pages ("Today's featured article" project page, its category page) instead of the actual featured article. |
| `python_to_guido` | multi_page_flow | Pass | Reached Guido van Rossum via Python (programming language). |
| `mallon_to_typhoid` | multi_page_flow | Fail | Reached `Typhoid_fever` via `Mary_Mallon` correctly at step 2 (and again at step 4) but never called done; final state drifted back to `Mary_Mallon`. |

## Routes

246 routed decisions across 25 tasks: `resolver` 90, `planner_fallback` 59, `actor` 38, `planner` (direct tool
step) 36, `planner_pick` (router) 16, `resolver_fuzzy` 5, **`search_fallback` 2 (new route)**.

`search_fallback` fired exactly twice — both on `wiki_long_page` (pass) and `wiki_search_mallon` (fail) — and in
both cases it fired only after a genuine planner failure on a page the agent had already tried to act on, never
as the very first action of a run. `planner_fallback` (the actor-only path) fired far more than in the first run
(59 vs. 18), driven by a new planner-output shape (see "New failure mode" below), not by the search-fallback
change itself. The baseline (`POLICY_BACKEND=laya`) has no comparable route breakdown — it does not use the
planner/resolver/actor/router split.

## Speed vs. targets

| Metric | Baseline | First run | This re-run | Target | Met this run? |
| --- | ---: | ---: | ---: | ---: | :---: |
| Actor / policy decision (median) | 860 ms | 688.5 ms | 878 ms | ≤ 1.0 s | Yes |
| Planner call / verifier (median) | 9,092 ms | 15,211 ms | 11,621 ms | ≤ 3.0 s | No (≈3.9×) |
| Wall time / executed action (median) | 10.38 s | 16.53 s | 19.1 s | ≤ 4.0 s | No (≈4.8×) |
| All routed decisions (mixed, median) | — | 11,727 ms | 4,317.5 ms | — | — |

The baseline's "planner call" column is its median verifier latency (`laya` backend has no planner call; this
is the closest analog, not a like-for-like figure). The actor stays inside budget in all three runs. The
planner call and wall time/action are both worse than the first run (11.6 s vs. 15.2 s planner is actually
*better*, but wall time/action is worse — 19.1 s vs. 16.53 s — consistent with `planner_fallback` firing 3× more
often this run, each fallback still paying for a failed planner call before falling back). The Mac's ~10 GB
swap pressure (noted going into this run, unchanged from the first run) remains the dominant cost driver on
both planner and wall-time metrics; no retries were used to improve either number.

## Failure-tag counts (108 wrong steps + 28 `correct=true` "reached goal, didn't stop" steps, across the 18
failed tasks; 3 steps on `turing_to_award` are `correct=true` throughout — that task's failure is a site/transport
crash between steps, not a per-step tag)

| Tag | This re-run | First run | Meaning here |
| --- | ---: | ---: | --- |
| `planner` | 55 | 39 | Wrong/missing step, continuing after the goal was met, or driving a `planner_fallback` guess in the wrong direction (dominates `turing_external_links`/`ada_external_links`'s image-gallery loops, `ada_references`' drift onto Wikisource, `mallon_to_typhoid`'s failure to stop). |
| `actuation` | 22 | 38 | Right element/tool, no further effect — mostly repeated `SCROLL_TO_TEXT` after the text was already found, and a few dead-end clicks. |
| `actor` | 15 | 19 | Actor chose wrong with confidence ≥ τ (flight field mix-ups, HN "show" instead of the Nth story, an unrelated "Hide Appearance" click). |
| `router` | 12 | 17 | `planner_pick` (the router) picked wrong from the actor's top 5 (drift through Wikipedia "Articles"/user pages, HN "show"/front-page loops). |
| `resolver` | 4 | 0 | A `resolver_fuzzy` approximate match picked the wrong candidate (`python_references`' "wild references", `hn_third_comments`' fuzzy time-link match). |
| `options` | 0 | 0 | Not observed. |
| `site` | 0 (0 per-step; 1 task-level) | 0 | No per-step `site` tag fired, but `turing_to_award` failed from a genuine browser/websocket crash (`RuntimeError: no close frame received or sent`) after 3 correct steps — counted at the task level since the crash has no corresponding step record. |

`planner` and `actuation` remain the two largest buckets, as in the first run, though their relative sizes
flipped: `planner` grew (55 vs. 39) and `actuation` shrank (22 vs. 38) — consistent with 15b's fix removing
found-scroll/focus-click false failures (fewer `actuation` steps) while a new planner-output malformation
(below) drove more genuine planner mistakes.

## New failure mode observed this run (not one of the four from 15b)

The live log shows `invalid plan, retrying once: Planner step must be an object, got 'CLICK'` (and variants:
`'CLICK: Mary Mallon'`, `'CLICK "References" (in the TOC)'`, `'CLICK link 7'`) on **19 of the 25 tasks' logs**
this run — the planner emitting a bare string instead of a step object, which the retry usually cannot repair,
so the run falls to the actor-only `planner_fallback` path. This is exactly the "planner occasionally emits a
bare string" issue flagged as a **known-but-unfixed concern** in the Task 15b report's Concerns section — it
predates this run and 15b explicitly did not attempt to fix it. It is the leading driver of the `below_fold`
category's collapse (1/5 vs. 5/5 first run): all four `external_links` tasks found the heading correctly at
step 1 via `SCROLL_TO_TEXT`, then fell into `planner_fallback` (image-gallery or show/hide-toggle loops) once
this malformation hit, instead of the agent recognizing the found heading and calling `done`.

## Recurrence of the four structural failure modes fixed in Task 15b

| # | Mode | Recurred? | Evidence |
| --- | --- | :---: | --- |
| 1 | Focus click on a search field treated as failure, then the typing step dropped | **No — 0 occurrences** | Every "open search box" focus click (`page_changed=False`) across 5 tasks (`wiki_search`, `wiki_search_turing`, `wiki_search_python`, `python_to_guido`, `turing_to_award`, plus one on `wiki_search_ada`) was immediately followed by its queued `TYPE_TEXT` step executing, never a drop/replan. |
| 2 | The same `SCROLL_TO_TEXT` re-run 3+ times | **Partial — 1 occurrence** | `python_external_links` ran `SCROLL_TO_TEXT` 3 times consecutively, but the repeat-exclusion (keyed on exact normalised `target_text`) was evaded because the wording varied step to step ("External links" → "External links section heading" → "External links"). All other tasks stayed at 2 consecutive scrolls (found, then no-effect), which the 15b fix correctly excludes. |
| 3 | The actor-only fallback navigating away from a start page before the planner acts | **No — 0 occurrences** | No task's first recorded step used `planner_fallback` (or any actor-only route); the only two `GOTO`s this run were the new, deliberate `search_fallback` route, and both fired only after the planner had already failed on a page it had previously acted on — never pre-emptively. |
| 4 | Planner field confusion (text in `target_text`, empty `value`) | **No — 0 occurrences** | No executed `TYPE_TEXT`/`SELECT` step had an empty `text`/value field; the swap repair held (or the malformed case never arose this run — this run's dominant malformation was the new "bare string instead of object" issue above, which the repair logic was never meant to catch). |

## Gate verdict

**Gate A+B (≥15/25): MISSED.** 7/25 achieved (28%), statistically indistinguishable from both the baseline
(7/25) and the first Gate A+B run (8/25). Per protocol, no further tuning proceeds without a decision from the
user; Phase C does not start automatically. The three of the four targeted structural bugs are confirmed fixed
(0 recurrences each) and the fourth (repeated-scroll) is substantially reduced (1 occurrence vs. 4+ before,
and only via a wording variant the fix didn't anticipate) — but a different, previously-known-and-flagged
planner malformation ("bare string instead of step object") is now the leading cause of failure, and it drove
a new regression in the below-fold category that offset the site-search improvement.
