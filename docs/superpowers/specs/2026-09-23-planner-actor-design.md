# Planner + Actor: closing the live-page gap (sub-project 1)

Date: 2026-09-23, revised 2026-09-24 after the 25-task baseline and the replay probe (section 1.2).
Builds on `2026-09-21-laya-agent-core-design.md` (all 12 tasks of that plan are done).

## 1. Evidence

### 1.1 Baseline (25 live tasks, `artifacts/live/20260923T175703Z/`, harness in `evals/live_tasks.py`)

7/25 tasks pass (28%) on independent final-page checks. 17 of 125 executed actions advanced the goal; 106 were
actor mistakes; 2 were provisional actuation failures (GitHub Issues click without navigation). 4 of 9 agent
`done` statuses were false. Median actor decision 860 ms; median verifier call 9.1 s; amortised 10.4 s/action.
Five runs ended in browser observation timeouts, all after the agent opened a Wikipedia "View source" editor page.

| Category | Passed |
|---|---|
| Site search | 1/5 |
| In-page navigation | 2/4 |
| Repeated-label disambiguation | 0/4 |
| Multi-field forms | 0/3 |
| Below-fold targets | 4/5 |
| Multi-page flows | 0/4 |

### 1.2 Replay probe (recorded requests through the same checkpoint, one input changed at a time)

| Task | As recorded | Context tags on options | Goal replaced by one step instruction |
|---|---|---|---|
| wiki_search_ada | logo 0.32 | "Main Page" 0.22 | Search box 0.82 |
| wiki_featured | View source 0.32 | View source 0.35 | Mary Mallon 0.22 + 0.21 (duplicate entries) |
| mallon_to_typhoid | Mary Mallon 0.48 | unchanged | Mary Mallon 0.72 |
| hn_comments | nav "comments" 0.95 | unchanged | story title 0.55 |
| ada_references | Toggle References 0.86 | unchanged | "8 References" 0.73 |
| flights | Where from? 0.93 | unchanged | click: Where to? 0.67; type: Where from? 0.49 vs Where to? 0.38 |

Conclusions that drive this design:
1. The correct element was in the shortlist in all six probed decisions. The dominant failure is the question,
   not the options.
2. A step instruction **in place of** the goal fixes 5 of 6 target choices. Goal plus appended step does not:
   the model anchors on the goal.
3. Context tags have no effect without training on them.
4. The operation question is unreliable even with an explicit step ("Type … into the search box" gives CLICK
   0.43). The operation must come from the planner.
5. Duplicate entries for the same link split probability mass.

### 1.3 Option-set defects (real, secondary)
- Scroll is never a choosable action: the policy scrolls only when no element exists.
- `snapshot.js` captures only elements inside the viewport; targets further down the page never reach the actor.
- On the Wikipedia Main Page the shortlist was identical for 7 different goals: nothing visible matched the
  goal's words, so ranking fell back to page order.
- Edit/history/view-source links are always offered; "View source" loads a heavy editor that caused all five
  observation timeouts.
- Identical labels (HN "comments", two "Mary Mallon") carry no disambiguating context.

## 2. Goals and non-goals

Goals
- The actor stays fast: a single-pass typed-decision model (Laya, or Kev if it wins the bake-off). The
  specific checkpoint is replaceable; the model family is a requirement.
- A local planner decomposes the task, owns the operation and value, tracks progress, and judges completion
  with quoted evidence.
- The actor answers only "which element", conditioned on a step instruction.
- The option set is complete (whole page, scroll as an action), deduplicated, free of known detours, and
  ranked by the step.
- The actor is retrained on step-conditioned, context-annotated data with hard negatives, and later on
  self-collected live data.

Non-goals (later sub-projects): skill notes per site, trajectory replay memory, macOS control, task UI,
screenshot models.

## 3. Success criteria

Measured on the 25-task suite, one run per task per configuration, same day, same machine.

- **Gate A+B (no retraining):** at least 15/25 (60%, +32 points over baseline).
- **Gate C (retrained actor):** at least 18/25, and no category below 2/3 of its tasks.
- **Speed:** median actor decision ≤ 1.0 s; median planner call ≤ 3.0 s; median wall time per executed action
  ≤ 4.0 s (planner, actor, text value and browser included). All reported with sample sizes.
- **Honesty:** differences under ~15 points on this suite are "not distinguishable". A failed gate is reported
  with a failure-tag breakdown, and no further tuning happens without a decision from the user.
- **Offline (Gate C only):** the adopted actor, in goal-only mode on Mind2Web test splits, does not drop more
  than 2 points macro element accuracy below the published checkpoint.

## 4. Architecture

```
          ┌──────────────── Planner (Qwen3-4B via mlx-lm, JSON) ────────────────┐
goal ───► │ status continue|done|blocked, evidence, steps[1..3]                │
          │ PlanStep {operation, target_text, value, instruction}             │
          │   operation ∈ CLICK | TYPE_TEXT | SELECT | SCROLL_TO_TEXT | GOTO    │
          └──────────────────────────────┬──────────────────────────────────────┘
                                         ▼
      Resolver: target_text matches exactly one element label ──► execute (no model call)
                                         │ otherwise
                                         ▼
      Actor (Laya|Kev): one target question, state.goal = step instruction,
      candidates = elements supporting the operation, ranked by step text, K=20
                                         │
             confidence ≥ τ ──► execute  │  confidence < τ ──► planner picks from actor top 5
                                         ▼
      StepMemory records (url, op, label, value, outcome); re-plan triggers; loop guard
```

## 5. Components

### 5.1 Observation — `snapshot.js`, `browser.py`
- Capture interactive elements for the **whole document** (cap 400), each with `in_viewport` and `y` (page
  offset). Elements off-screen are valid targets; `browser.act` scrolls the node into view (`scrollIntoView`
  with `block: "center"`) and re-hit-tests before input.
- **Deduplicate** link elements that share a resolved `href` (fragment kept, since fragments are distinct
  in-page targets); keep the first, attach the others' labels as `aliases`.
- **Context fields** per element: `landmark` (nearest `nav|header|footer|aside|main|form|dialog` by tag or
  role, else `main`), `section` (nearest preceding heading text within the landmark, ≤ 40 chars),
  `href_path` (path + fragment, no query, ≤ 40 chars). Also `row_text` for elements inside `tr`/`li`
  (the row's first 60 characters of text), which disambiguates HN-style lists.
- **Detour filter** (`DETOUR_PATTERNS` in `candidates.py`): links whose `href` query contains
  `action=edit|history|raw|info` or whose path contains `/edit` are removed unless the goal contains "edit",
  "history" or "source". Small fixed list; no per-site rules.
- Page text for the planner: the visible text (existing) plus the document's headings outline (`h1–h3`,
  ≤ 1,000 chars).

### 5.2 Planner — `jev_ultrafast/planner.py`
- Input (one JSON user message, ≤ ~1,200 tokens): goal; URL and title; headings outline; visible text
  (≤ 1,500 chars); up to 40 elements as `id | label | role | landmark › section | row_text | offscreen`
  ranked by relevance to the goal; completed steps; failed attempts from StepMemory.
- Output (validated): `status`, `evidence` (required when `done`; must appear verbatim, case-insensitively,
  in page text or title), `steps` (1–3 PlanSteps).
- PlanStep: `operation` (`CLICK|TYPE_TEXT|SELECT|SCROLL_TO_TEXT|GOTO`), `target_text` (label as shown, or the
  text to scroll to, or the URL for GOTO), `value` (TYPE_TEXT/SELECT only), `instruction` (≤ 20 words,
  imperative, names the element, e.g. `Type "Ada Lovelace" into the Search Wikipedia box`).
- `GOTO` is allowed only for URLs produced by the search-shortcut registry (5.7); anything else is rejected.
- Runs: at start, on URL change, when its step queue is empty, and after a failed step. Steps from one call
  are consumed in order while the page stays on the same URL.
- `done` without verbatim evidence is rejected as an invalid plan (one retry, then `planner_fallback`).
  `verifier.py` is not used by the planner backend; it stays only on the `laya` backend for comparison runs.
- Invalid JSON after one retry: `planner_fallback` (actor-only on the raw goal for this page), logged and
  counted.
- Model: `TEXT_MODEL` (Qwen3-4B-Instruct-2507-4bit default). `max_tokens` 200, temperature 0. For Qwen3
  hybrid models, requests include `chat_template_kwargs: {"enable_thinking": false}`. A benchmark task picks
  between 4B and 1.7B on recorded pages (quality and latency) before live runs.

### 5.3 Resolver — `jev_ultrafast/resolver.py`
- Normalise (casefold, collapse whitespace, strip punctuation) `target_text` and each candidate's label and
  aliases for the step's operation. Exactly one exact match → that element, confidence 1.0, route `resolver`.
- No exact match: token-set containment (every target word in the label) with exactly one hit → route
  `resolver_fuzzy`. Otherwise defer to the actor.

### 5.4 Actor — `jev_ultrafast/actor.py`, changes to `policy.py`
- One question only: `<op>_target` for the planner's operation. No operation question.
- `state = {"goal": step.instruction, "recent_actions": last 3}`. The full task is **not** included (1.2 #2).
- Candidates: elements supporting the operation, detours removed, ranked by the shortlister with
  `instruction + target_text` as the query, K = 20, kept in page order after selection (existing behaviour).
- Backend: `ACTOR_BACKEND=laya|kev` (Kev added in Phase C).
- Rendering: labels only until the retrained actor (Phase C) ships; then `render_option` appends context
  (`[nav]`, `[main › Section]`, row text) in the trained format.

### 5.5 Router — `jev_ultrafast/router.py`
- Accept the actor's pick when target confidence ≥ τ (per actor). Otherwise ask the planner a single
  `choice` prompt over the actor's top 5 (labels with context) plus "none of these"; "none" triggers a re-plan.
- τ fitted on Mind2Web dev in step mode (the lowest confidence at which accepted picks are ≥ 90% accurate); never
  on test. The baseline's live decisions used the old request shape, so they cannot calibrate the new one.
  Default before fitting: 0.5.

### 5.6 StepMemory and loop guard — `jev_ultrafast/memory.py`
- Records `(url without fragment, operation, normalised label, value)` and outcome
  `url_changed | page_changed | no_change`.
- Excludes from candidates on the current URL: actions executed twice already, and actions whose last run
  produced `no_change` (except TYPE_TEXT, whose effect is within the field). Excluded actions are listed to the
  planner as failed attempts.
- Blocked when 4 consecutive actions produce no URL change and the planner reports no progress
  (`completed` list did not grow).

### 5.7 Tools — `jev_ultrafast/tools.py`
- `SCROLL_TO_TEXT(text)`: scrolls to the first element whose text contains `text`; fails if not found.
- Search-shortcut registry: `{host: url_template}` for sites with a stable GET search, starting with
  `en.wikipedia.org → https://en.wikipedia.org/w/index.php?search={q}&title=Special:Search&go=Go` and
  `github.com → https://github.com/search?q={q}`. The planner may emit `GOTO` with `target_text` equal to a
  rendered template; the executor validates it against the registry before navigating.

### 5.8 Values and actuation
- TYPE_TEXT and SELECT values come from `PlanStep.value`. The separate text-model call runs only when the
  planner left `value` empty. Any non-string or empty value is rejected (fixes the literal `false`).
- SELECT option choice uses `value` matched against option labels (resolver rules), falling back to
  `choose_option`.
- GitHub Issues click: diagnosed first from a live repro (element at the hit-test point versus the chosen
  node), then fixed with a regression test. No speculative change to `browser.py`.

### 5.9 Failure tags and records
Each live step records: planner call (prompt size, output, latency), route
(`resolver|resolver_fuzzy|actor|planner_pick|planner_fallback`), actor request/answer/confidence, executed
action, URL before/after, value. Post-run tags: `planner | resolver | actor | router | options | actuation |
site`.

## 6. Actor retraining (Phase C)

Data: the existing Mind2Web train pipeline with three changes, applied identically for every actor.
1. **Step conditioning.** For 70% of items, `state.goal` is a templated step from the gold action
   (`Click "{label}"`, `Type "{value}" into "{label}"`, `Select "{value}" in "{label}"`), with one random word
   dropped from the label with p = 0.3, and 1 of 3 phrasing templates. The other 30% keep the task goal
   (keeps goal-only ability for fallback). The operation question is still trained (unused live, harmless).
2. **Context.** `mind2web.py` derives `landmark`, `section`, `row_text` and `href_path` from the raw DOM
   with rules identical to `snapshot.js`; a parity test renders one fixture through both paths.
3. **Hard negatives.** Items whose shortlist contains a same-label or same-section distractor, or a nav/header
   element whose label shares a word with the gold, are oversampled ×2.

Runs (Kaggle 2×T4, existing CLI flow): Laya-v2 with the published schedule; Kev-0.8B via `kev.train` from
`jaredpalmer/kev-0.8b` on the same items exported to Kev's JSONL. Kev opens with a 50-step smoke run; if T4
fails, fp16/fp32 settings, then local M2 LoRA; time cost reported.

Offline evaluation (single confirmatory test run per mode; thresholds from dev only): goal-only mode
(comparable to the card) and step mode (step from gold). Live: {Laya-v2, Kev-0.8B} in the Phase A+B loop.
Adopt by live success within the speed budget; ties go to the faster one.

## 7. Self-collected live data (Phase D, gated)

- Explorer: the Phase A+B agent runs on read-only public sites (the suite's hosts plus up to 10 more chosen with
  the user), with exploration goals sampled from each page's own headings and links.
- Hindsight relabelling (NNetNav-style): the planner writes the instruction each executed step accomplished,
  given the before/after page; steps whose effect cannot be stated are dropped.
- Items are rendered through the same formatter (step-conditioned, context), deduplicated, and added to the
  Mind2Web items for a Laya-v3 run. Suite tasks' exact goals are excluded from exploration seeds.
- Respects `robots.txt`, one request per second per host, no logins or form submissions that create data.
- Starts only after Gate C is reported and the user approves the site list.

## 8. Resource budget (M2, 16 GB)

| Component | Memory |
|---|---|
| macOS + Chrome (1 tab) | ~5–6 GB |
| Planner Qwen3-4B 4-bit (1.7B option) | ~2.3 GB (~1.0 GB) |
| Actor Laya fp16 or Kev-0.8B | ~0.9 / ~1.7 GB |
| Headroom | ≥ 5 GB |

## 9. Error handling
- Planner or text server unreachable: fail the run with endpoint and operation; no silent model switch.
- Actor answer not among offered ids: existing `ValueError`, nothing executed.
- `GOTO` outside the registry, SCROLL_TO_TEXT miss, or resolver/actor finding no candidate for the operation:
  step fails, StepMemory records it, planner re-plans.
- Browser observation timeout: recorded as `site`/`actuation`, run ends with its record written (existing).

## 10. Testing
- Unit (pytest, fakes for planner/actor/browser): planner validation and evidence check; resolver exact/fuzzy
  and ambiguity; actor request shape (single target question, step as goal); router threshold and top-5 pick;
  StepMemory exclusions and blocked rule; tools registry validation; value guard; detour filter; dedupe;
  snapshot context fields (rendered HTML fixture through `scripts/render_fixture.py`); train/serve context
  parity; step-template generation.
- Coverage ≥ 80% on new modules.
- Offline replay regression: the six probe decisions in 1.2 become a script (`scripts/replay_probe.py`) that
  reports the actor's pick for each under the new request shape; run after every actor change.
- The live suite is an evaluation, not CI.

## 11. Constraints carried over
No Jev outputs as labels (TypeSafe terms 2.3(b)); Mind2Web test evaluation-only; `uv` only; ruff line length
120; commits via `scripts/commit.sh`; training on Kaggle (Kev T4 fallback noted). No new dependency: Kev runs from
its own checkout behind its local server; its licence and its Qwen3.5 base licence are recorded in `NOTICE.md`
before weights are pulled.

## 12. Gates
1. Phase A+B built → planner model benchmark and replay regression → live suite run → report (Gate A+B).
2. Phase C data + both trainings → offline report → user picks actors for live → live report (Gate C).
3. Phase D only on user approval of the site list.
4. Model cards/HF publish only on user approval.

## 13. Open risks
- Planner quality at 4B is now the likely ceiling (it plans, picks operations, and judges completion).
- Planner latency: a ~1,200-token prompt on M2 may exceed 3 s; mitigations are steps-per-call, prompt size,
  and the 1.7B option, measured before live runs.
- Whole-page capture raises element counts; the shortlist (K=20) protects the actor, the 40-element planner
  view may miss targets on very long pages (SCROLL_TO_TEXT covers this).
- Live sites change between runs; same-session runs and the `site` tag limit the noise.
