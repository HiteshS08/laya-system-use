# Branch evaluation: redesign (program backend) vs plan-once-per-page vs reference

Date: 2026-09-26. Sources: `.superpowers/sdd/2026-09-24-planner-actor/evals/notes.md` (handoff), the three
redesign runs in `/Users/hiteshs/wt-redesign-first-principles/artifacts/live/`, the plan-once run in
`/Users/hiteshs/wt-feat-plan-once-per-page/artifacts/live/`, and the three reference runs in this repo's
`artifacts/live/`. No live runs were executed for this report; no code was changed.

## D1: compiler model choice (redesign branch)

25 suite goals + 12 held-out goals, hand-judged for `valid` (parses under the program grammar) and `correct`
(the right verb/target/ordinal for the goal):

| Model | Valid | Correct | Median latency |
|---|---|---|---|
| Qwen3-4B-Instruct-2507-4bit | 32/32 (100%) | 22/32 (69%) | 974.5 ms |
| Qwen3-1.7B-4bit | 31/32 (97%) | 26/32 (81%) | 515.5 ms |

Neither model cleared the report's bar (≥ 90% valid-and-correct, median ≤ 5 s). Per the fallback rule (pick the
model with more correct programs), **`COMPILER_MODEL=mlx-community/Qwen3-1.7B-4bit`** was chosen: more correct
programs, smaller, faster. 4B's failures were structural verb/target mis-picks (`JUMP` vs `OPEN`, dropped
qualifiers); 1.7B's were mostly cosmetic (redundant steps, one hallucinated step) rather than wrong destinations.

## Pass counts, all runs

| Run | Backend | Pass |
|---|---|---|
| baseline `20260923T175703Z` | planner (early) | 7/25 |
| first gate `20260925T040642Z` | planner | 8/25 |
| re-run `20260925T124019Z` | planner | 7/25 |
| redesign run 1 `20260926T074140Z` | program | 10/25 |
| redesign run 2 `20260926T074522Z` | program | 10/25 |
| redesign run 3 `20260926T074738Z` | program | 10/25 |
| plan-once `20260926T075114Z` | planner (plan-once-per-page) | 6/25 |

The three redesign runs are **identical at the per-task level** — same 10 tasks pass, same 15 fail, in all three
runs (confirmed by diffing `success` across all 25 tasks). No task differs across the three redesign runs, so
majority vote and any single run give the same 10/25.

## Category table (passed/total)

| Category | Baseline | First gate | Re-run | Redesign (×3, identical) | Plan-once |
|---|---|---|---|---|---|
| below_fold | 4/5 | 5/5 | 1/5 | **5/5** | 1/5 |
| in_page_navigation | 2/4 | 1/4 | 1/4 | **4/4** | 1/4 |
| multi_page_flow | 0/4 | 0/4 | 1/4 | 1/4 | 1/4 |
| multi_field_form | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| disambiguation | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| site_search | 1/5 | 2/5 | 4/5 | 0/5 | 3/5 |
| **Total** | **7/25** | **8/25** | **7/25** | **10/25** | **6/25** |

Redesign sweeps `below_fold` and `in_page_navigation` (deterministic ordinal/tactic routes) but goes to 0/5 on
`site_search` — every one of the five site-search wins the planner backends had came from either a redundant
compiled program or a resolver mis-pick after a correct search (see failure patterns below). No branch has moved
`multi_field_form` or `disambiguation` off the floor.

## Per-task results across all runs (Y = pass)

| Task | Category | Baseline | First gate | Re-run | Redesign | Plan-once |
|---|---|---|---|---|---|---|
| ada_external_links | below_fold | Y | Y | n | Y | n |
| godel_external_links | below_fold | n | Y | n | Y | n |
| python_external_links | below_fold | Y | Y | n | Y | n |
| turing_external_links | below_fold | Y | Y | n | Y | n |
| wiki_long_page | below_fold | Y | Y | Y | Y | Y |
| hn_comments | disambiguation | n | n | n | n | n |
| hn_second_comments | disambiguation | n | n | n | n | n |
| hn_third_comments | disambiguation | n | n | n | n | n |
| wiki_featured | disambiguation | n | n | n | n | n |
| ada_references | in_page_navigation | n | n | n | Y | n |
| gh_issues | in_page_navigation | Y | Y | Y | Y | Y |
| python_references | in_page_navigation | n | n | n | Y | n |
| turing_references | in_page_navigation | Y | n | n | Y | n |
| flights | multi_field_form | n | n | n | n | n |
| flights_mumbai_delhi | multi_field_form | n | n | n | n | n |
| flights_paris_rome | multi_field_form | n | n | n | n | n |
| ada_to_babbage | multi_page_flow | n | n | n | n | n |
| mallon_to_typhoid | multi_page_flow | n | n | n | n | Y |
| python_to_guido | multi_page_flow | n | n | Y | Y | n |
| turing_to_award | multi_page_flow | n | n | n | n | n |
| wiki_search | site_search | n | n | Y | n | Y |
| wiki_search_ada | site_search | n | Y | Y | n | n |
| wiki_search_mallon | site_search | Y | Y | n | n | Y |
| wiki_search_python | site_search | n | n | Y | n | n |
| wiki_search_turing | site_search | n | n | Y | n | Y |

`hn_comments`, `hn_second_comments`, `hn_third_comments`, `wiki_featured`, all three `flights*`, `ada_to_babbage`,
`turing_to_award` and all three `multi_field_form`/`disambiguation` tasks fail on every branch tested to date.

## Timing vs targets

Targets: actor decision ≤ 1.0 s, planner/compile ≤ 3.0 s per call, wall time ≤ 4.0 s per action.

| Branch | Median actor | Median planner/compile | Median wall s/action | vs targets |
|---|---|---|---|---|
| baseline | n/a (no LLM route recorded) | n/a | 11.13 s | wall FAIL |
| first gate | 688.5 ms | 15,211 ms | 16.53 s | actor pass, planner/wall FAIL |
| re-run | 878 ms | 11,621 ms | 19.10 s | actor pass, planner/wall FAIL |
| redesign (run 1/2/3) | 172.5 / 162 / 187 ms | 409 / 393 / 0 ms | 1.675 / 1.475 / 1.2 s | **all three PASS** |
| plan-once | 253.5 ms | 9,806 ms (planner decision median 7,245.5 ms) | 12.15 s | actor pass, planner/wall FAIL |

Redesign is the only branch inside all three timing targets, by a wide margin — its program-compile call is ~25×
cheaper than the planner call it replaces.

**LLM calls per task / planner calls per URL** (spec §8 target: ≤ 1 LLM call per task on average):

| Branch | LLM calls per task | Planner calls per URL |
|---|---|---|
| baseline | ~0 (no LLM backend) | n/a |
| first gate | 5.48 mean (137 calls / 25 tasks) | not computed |
| re-run | 8.48 mean (212 calls / 25 tasks) | not computed |
| redesign (run 1/2/3) | 1.0 / 0.8 / 0.0 | ≈ same (one compile per goal, cached on runs 2–3) — **PASS** |
| plan-once | 164 total calls | 1.93 per URL — **misses its own ≤ 1 target** |

## Route counts

| Route | Baseline | First gate | Re-run | Redesign (run 1) | Plan-once |
|---|---|---|---|---|---|
| tactic | — | — | — | 17 | — |
| controller | — | — | — | 23 | — |
| ordinal | — | — | — | 5 | — |
| resolver | — | 62 | 90 | 13 | 79 |
| resolver_fuzzy | — | 7 | 5 | 1 | 3 |
| actor | — | 29 | 38 | 22 | 59 |
| planner | — | 57 | 36 | — | 30 |
| planner_pick | — | 22 | 16 | — | 31 |
| planner_fallback | — | 18 | 59 | — | 21 |
| search_fallback | — | — | 2 | — | 3 |
| retry | — | — | — | — | 4 |
| none (untracked) | 143 | — | — | — | — |

Redesign's route mix has no `planner`/`planner_pick`/`planner_fallback` at all — those routes don't exist in the
program backend; its equivalents are `tactic` (compiled step), `controller` (subgoal/DONE bookkeeping) and
`ordinal`/`resolver`/`resolver_fuzzy` (deterministic grounding), confirming the architecture note that the
per-step LLM planner is gone.

## Failure-tag counts per branch

| Tag | Baseline | First gate | Re-run | Redesign (run 1) | Plan-once |
|---|---|---|---|---|---|
| actor | 106 | 19 | 15 | 11 | — |
| actuation | 2 | 38 | 22 | — | 33 |
| planner | — | 39 | 55 | — | 11 |
| router | — | 17 | 12 | — | — |
| resolver | — | — | 4 | 4 | — |
| compiler | — | — | — | 4 | — |
| check | — | — | — | — | 2 |

Most common patterns (redesign, run 1 — 15 failed tasks, 19 tagged steps):

1. **actor (11 steps)** — the vision actor picks a plausible but wrong element instead of the compiled step's
   target. Example: `flights` — the program calls for `FILL Departure = Zurich`, but the actor clicked a
   pre-existing "New Delhi" suggestion chip on Google Flights, then a random flight-search suggestion card,
   never touching the requested fields. All three `flights*` tasks fail identically on this pattern.
2. **compiler (4 steps)** — the compiled program is redundant or under-specified. Example: `wiki_search` —
   program is `FIND Gödel's incompleteness theorems\nOPEN Wikipedia`; the `FIND` step alone already reached the
   target article via search, but the spurious second `OPEN Wikipedia` step then navigated back to the Main
   Page.
3. **resolver (4 steps)** — ordinal/fuzzy grounding grabs the wrong on-page element. Example:
   `mallon_to_typhoid` — `OPEN typhoid fever @1` resolved to the page's own table-of-contents anchor
   ("1.3 Identified as a typhoid carrier, 1906–1909") instead of the real `Typhoid_fever` wiki link.
4. **resolver, task-level (no step to tag)** — `hn_second_comments`/`hn_third_comments` both `BLOCKED` with
   `"OPEN comments: no element for Click the comments"` before any step ran: ordinal counting found only the
   first story's "comments" link and never enumerated the second/third.
5. **site, task-level (no step to tag)** — `wiki_featured` raised `RuntimeError: Could not identify the expected
   link for wiki_featured on the starting page` before any step ran. This is the exact trap flagged in the
   overnight report's Open Risks: `OPEN today's featured article` resolves to the "Today's featured article"
   project-page link, not the featured article itself, and the article changes daily.

Most common patterns (plan-once — 19 failed tasks, 44 tagged steps):

1. **actuation (33 steps)** — a resolver or actor click registers but produces no page change (`page_changed:
   false`). Example: `ada_external_links` step 2 — a `resolver_fuzzy` click had no effect at all.
2. **planner (11 steps)** — the plan never reaches a usable `done_when`; the task runs out of its step budget
   still replanning. Example: `ada_to_babbage` — only one `search_fallback` `GOTO` ever ran; no further plan
   advanced the task to "open Charles Babbage" before it gave up. This matches notes.md's own finding for this
   run: `done_when` fired on 0 of 30 planner calls.
3. **check (2 steps)** — the controller calls a task done on the wrong final state. Example: `hn_comments` —
   status finished `"done"`, but the final URL was the site's global `/newcomments` page, not the top story's
   own comments page, so `success=False`.

## Paired comparisons (`scripts/compare_runs.py`, majority vote + exact McNemar)

| Comparison | n | a_passed | b_passed | discordant (a-only / b-only) | p-value |
|---|---|---|---|---|---|
| (a) redesign majority (3 runs) vs re-run `20260925T124019Z` | 25 | 10 | 7 | 7 / 4 | **0.549** |
| (b) redesign majority (3 runs) vs plan-once | 25 | 10 | 6 | 8 / 4 | **0.388** |
| (c) plan-once vs re-run `20260925T124019Z` | 25 | 6 | 7 | 2 / 3 | **1.000** |

- (a) redesign-only wins: `ada_external_links`, `ada_references`, `godel_external_links`,
  `python_external_links`, `python_references`, `turing_external_links`, `turing_references`. Re-run-only wins:
  `wiki_search`, `wiki_search_ada`, `wiki_search_python`, `wiki_search_turing`.
- (b) redesign-only wins: same seven plus `python_to_guido`. Plan-once-only wins: `mallon_to_typhoid`,
  `wiki_search`, `wiki_search_mallon`, `wiki_search_turing`.
- (c) plan-once-only wins: `mallon_to_typhoid`, `wiki_search_mallon`. Re-run-only wins: `python_to_guido`,
  `wiki_search_ada`, `wiki_search_python`.

None of the three comparisons reach p < 0.05: the redesign's 3-point raw lead over the re-run, and its 4-point
lead over plan-once, are each built from a small, mostly-disjoint set of discordant tasks (below_fold/in-page-nav
wins traded against site-search losses), which an exact McNemar test on 25 tasks cannot separate from chance.

## Gate verdicts

| Branch | ≥ 15/25 | Verdict |
|---|---|---|
| baseline | 7/25 | **FAIL** |
| first gate | 8/25 | **FAIL** |
| re-run | 7/25 | **FAIL** |
| redesign (majority) | 10/25 | **FAIL** |
| plan-once | 6/25 | **FAIL** |

**Redesign against spec §8's full criteria** (from the overnight report):

| Criterion | Result | Verdict |
|---|---|---|
| majority-vote ≥ 15/25 | 10/25 | **FAIL** |
| McNemar p < 0.05 vs re-run `20260925T124019Z` | p = 0.549 | **FAIL** |
| median wall time per action ≤ 4.0 s | 1.2–1.675 s | **PASS** |
| median actor decision ≤ 1.0 s | 162–187 ms | **PASS** |
| ≤ 1 LLM call per task on average | 0.0–1.0 | **PASS** |

Redesign clears every latency/cost criterion by a wide margin and clears no accuracy criterion. Per the spec's
own D5 rule ("only if D2 shows the program backend ≥ the planner backend"), this run does **not** clear the bar
to make `program` the default or to delete the planner stack — though it is the highest raw pass count of any
branch or reference run measured so far, and the failure is not statistically distinguishable from the re-run at
n = 25.

## Caveat

With 25 tasks, an exact McNemar test cannot separate two pass rates whose difference is under roughly 15 points
from chance (`scripts/compare_runs.py`'s own docstring). Every comparison above — baseline vs first gate (+1),
first gate vs re-run (-1), re-run vs redesign (+3), redesign vs plan-once (+4) — falls well inside that
undistinguishable band; none should be read as a proven improvement or regression without a larger suite or a
repeated-measures design.

## What to fix next (redesign, evidence-based)

1. **Compiler: stop emitting a redundant `OPEN` after a `FIND` that already lands on the target.** 4/19 tagged
   failed-steps were `compiler`, and two of the five `site_search` losses (`wiki_search`, and part of
   `wiki_search_python`'s qualifier drop) trace to the compiled program taking an unnecessary extra step after
   the goal was already reached.
2. **Give multi-field forms a deterministic FILL-actuation path instead of falling through to the vision
   actor.** `actor` was the single largest failure tag (11/19 steps), and all three `flights*` tasks fail with
   the *identical* wrong trace (clicking a pre-existing "New Delhi" suggestion chip, then an unrelated flight
   card) regardless of the requested origin/destination/date — the actor is guessing on a page it has no
   step-level grounding for.
3. **Add a scroll/expand-before-give-up fallback to ordinal/fuzzy resolution.** `ada_to_babbage` burned 11
   resolver retries on `OPEN Charles Babbage` without ever finding it (likely below the fold) before stalling at
   1 step; `mallon_to_typhoid`'s ordinal grabbed an in-page TOC anchor instead of the real article link.
4. **Fix HN ordinal counting for `@2`/`@3` comments targets.** `hn_second_comments`/`hn_third_comments` both
   block with "no element" — the resolver only sees the first story's "comments" link, so any ordinal beyond 1
   fails outright rather than mis-picking.
5. **`wiki_featured` needs the dedicated tactic the overnight report already flagged as an open risk**: `OPEN
   today's featured article` must be distinguished from the "Today's featured article" project-page link, and
   the harness's own `expected_link` lookup needs to tolerate the daily-changing article rather than raising
   before any step runs.
