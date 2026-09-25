# First-principles redesign: compile the goal once, act with typed decisions

Date: 2026-09-26. Status: design for the `redesign/first-principles` branch. Supersedes the per-step planner of
`2026-09-23-planner-actor-design.md`. Keeps the fine-tuned Laya actor and the observation code of
`2026-09-21-laya-agent-core-design.md`.

## 1. The problem, restated

One natural-language goal comes in, for example "Find Alan Turing's article, then open the Turing Award article from
it". The agent must leave the browser in a state where that goal is visibly met, and then stop. It runs locally on a
MacBook Air M2 (16 GB, already ~10 GB into swap), with an open-weight, single-pass typed-decision model (Laya today,
maybe Kev later) at the centre of choosing actions. It has to be fast: ≤ 1.0 s per actor decision and ≤ 4.0 s wall time
per action. It must not learn from Jev/TypeSafe outputs. The browser comes first; the Mac comes later.

Three things decide success on the 25-task suite, and the design should be judged on each:

1. **Knowing what to do next**: turning the goal into the next concrete intent on this page.
2. **Grounding**: turning that intent into one observed element or tool call.
3. **Knowing when to stop**: the suite checks the *final* page, so every action taken after the goal is met can
   only lose. That is what happened on `mallon_to_typhoid`, `turing_references` and the four `*_external_links` tasks.

## 2. Evidence this design rests on

| Fact | Source | What it implies |
|---|---|---|
| 7/25, 8/25, 7/25 on three live runs; all inside the noise band | reports 09-24, 09-25 | The planner+actor loop did not move the pass rate. |
| Wrong steps by cause: planner 39–55, actor 15–19, router 12–17, actuation 22–38 | same | The per-step LLM planner is the largest error source, not the actor. |
| Planner call median 11.6–15.2 s; wall time 16.5–19.1 s per action; Qwen3-4B decodes ~11.5 tok/s under swap | same, brief | At 11.5 tok/s a 100–200-token JSON plan costs 9–17 s *in decode alone*. Output tokens, not the prompt, are the cost. |
| Actor decision 0.69–0.88 s median | same | The actor already meets its budget. |
| Replay probe: a step instruction in place of the goal fixed 5/6 recorded actor mistakes; goal + appended step did not | spec 09-23 §1.2 | Laya grounds well when asked "which element is *this*", badly when asked "what advances *this goal*". |
| Gold element was in the K=20 shortlist in 6/6 probed decisions | same | The live shortlist is not the bottleneck when the query names the element. |
| Offline macro element accuracy 0.48–0.50 on Mind2Web test splits; 21–27% of gold elements lost by the lexical shortlist when queried with the task goal | model card | Goal-mode recall is weak; step-mode recall is unmeasured and should be measured. |
| Planner malformations: bare strings instead of step objects on 19/25 task logs; value/target swaps; continuing after success; repeated scrolls with reworded targets | report 09-25 | Asking a 4B model for free-form JSON every step produces a long tail of format and judgement errors; each fix exposed the next one. |
| Completion missed because the planner's 600-character view excluded the scrolled-to heading | brief B | Completion was judged by an LLM on a truncated view instead of checked directly on the page. |

## 3. First-principles analysis: every assumption kept, changed or dropped

| # | Current assumption | Verdict | Reason and evidence |
|---|---|---|---|
| A1 | A local LLM plans every step (1–3 steps per call, re-called on URL change, empty queue or failure) | **Dropped** | Largest error source (39–55 wrong steps) and 4× over the speed budget. Nothing it decides per step needs generation: operation, target and value are typed choices once the goal's slots are known. |
| A2 | An LLM is needed at all | **Changed: once per task, tiny output** | Extracting names, values and the sequence of intents from free text is language understanding a 421M encoder cannot do (it has no span or text output). One call per task whose output is a ~40-token line program (not JSON) costs ~3–5 s once, and 0 s when the goal is cached. |
| A3 | Planner/actor split | **Kept, moved** | The split is right (the probe proves grounding improves with a step instruction), but the "planner" becomes a compiler run once, and the per-step policy becomes deterministic tactics plus Laya. |
| A4 | The planner judges completion from a truncated page view | **Dropped** | Completion is a deterministic predicate per subgoal, checked on the full observation after every action: title/h1 names the entity, URL fragment or a visible heading names the section, the field holds the value, the URL changed. The run stops the moment the last predicate holds. |
| A5 | The actor answers "which element advances the goal" (goal mode) | **Changed** | The actor always gets a templated step instruction (`Click the search result for Alan Turing.`). Templates come from one function shared with training (`instructions.py`), so train and serve inputs match. |
| A6 | The actor also answers the operation question | **Dropped** | Probe: the operation head is unreliable even with an explicit step. The subgoal kind fixes the operation. |
| A7 | Laya is the actor | **Kept** | Meets the speed budget, Apache-2.0, fine-tunable on Kaggle. Kev-0.8B is a bake-off candidate only: its Gated DeltaNet kernels have no MPS implementation (0.33 s for five questions on an M5 in bf16; slower on M2) and its README lists ~4 GB memory versus ~0.9 GB for Laya. |
| A8 | Router: when actor confidence < τ, a second LLM call picks from the top 5 | **Dropped** | The pick call is another 2–15 s LLM call and was wrong 12–17 times. Instead: take the actor's top-1; if the action has no effect, exclude that element and take the next-ranked (a new decision on a new observation, never a retried mutation). |
| A9 | Resolver: exact or token-set label match skips the model | **Kept** | Free and never wrong on its own terms (0 resolver errors in run 1). It now matches subgoal targets that the compiler copied from the goal. |
| A10 | DOM candidate pipeline (snapshot.js: whole document, dedupe by href, landmark/section/row context, detour filter) | **Kept, extended** | It is effectively an accessibility-tree view with stable node ids and pre-input hit testing, which is what AgentOccam and Region4Web argue for (compact, semantic observations). Added: a `headings` list (text, level, id, in-viewport) for completion checks, and the page's OpenSearch link. |
| A11 | K=20 lexical shortlist | **Kept for now, re-queried** | The query is the step instruction plus target phrase, which names the element; recall was 6/6 in the probe. Step-mode recall@K is to be measured offline on Mind2Web dev before any change of K (DEFERRED TO MAC, needs the data). |
| A12 | Screenshots | **Dropped from the loop** | The actor does not consume pixels; vision models that fit (ShowUI-2B) or ground best (UI-TARS-1.5-7B, 4-bit MLX ~5 GB) cost memory the machine does not have and seconds per step. Kept only as optional recordings. |
| A13 | Action set: CLICK, TYPE_TEXT, SELECT, SCROLL_TO_TEXT, GOTO (registered templates only), WAIT, SCROLL_* | **Changed** | Added SUBMIT (press Enter in the focused field), the most general way to run a search or a form, and FRAGMENT (go to an observed heading's own `id` anchor when no contents link is visible). GOTO is now allowed only for URLs *rendered by code* from a search template discovered on that host (OpenSearch) or learned from an observed search; the model never emits a URL. WAIT/SCROLL_* stay available only to the goal-mode fallback. |
| A14 | Step memory with exclusions, repeat limits and a 4-action stall rule | **Simplified** | Memory is per subgoal: elements that produced no effect are excluded for that subgoal (keyed by page and label, since element numbers change between observations); a subgoal gets at most 4 actions and 2 no-effect actions, then the run stops BLOCKED. Stopping early is better than wandering because the final page is what is checked. |
| A15 | Hardcoded per-site search registry (Wikipedia, GitHub) | **Changed** | Per-site knowledge is discovered, not written: the OpenSearch description (`<link rel="search">`, a web standard both sites publish) or a template learned the first time a typed search lands on a URL containing the query. Stored in a small cache. |
| A16 | Workflow memory / replay | **Minimal** | Only deterministic facts are cached: compiled programs by goal text, and search templates by host. A 2026 budget-matched study found that AWM, ASI and ReasoningBank did not beat a vanilla actor given the same token budget (arXiv 2606.15017). AWM's reported gains (+24.6% relative on Mind2Web, +51.1% on WebArena) came from LLM agents, not a typed actor. |
| A17 | Mind2Web-only training | **Changed** | Mind2Web stays (CC-BY-4.0, train split only). Add (a) step-mode items rendered with the serving templates and (b) self-collected items from exploring public pages with no LLM, where the label comes from the page itself (NNetNav / OS-Genesis / Explorer / SynWeaver all show interaction-first synthesis works). Nothing from Jev, ever. |
| A18 | Evaluation: one run per task on 25 tasks; gate at 15/25 | **Changed** | Three runs per task, paired comparison per task (exact McNemar), a held-out task set on other sites, and a check that model-call counts match the design. |
| A19 | Verifier LLM for DONE on the `laya` backend | **Dropped for the new backend** | Replaced by predicates (A4). The old backend keeps it until deletion. |
| A20 | The TEXT model writes TYPE_TEXT values per field | **Changed** | Values are copied from the goal by the compiler (`FILL Where to? = London`). Field text generation stays only in the goal-mode fallback. |

## 4. Architecture

```
goal ──► Compiler (once per task; cached)                    Qwen3 via mlx_lm.server, ~40 output tokens
         "FIND Alan Turing\nFIND Turing Award"  ──►  Program(subgoals, done_text)
                                   │
 per action:  observe ──► Controller ──► subgoal satisfied? (checks.py) ──► advance / DONE
                              │ no
                              ▼
                          Tactic (tactics.py): subgoal + page ──► Step {operation, target, value,
                              │                                         instruction, ordinal, index, roles}
                              ▼
            tool? (SCROLL_TO_TEXT | GOTO rendered from template | SUBMIT)  ──► execute
                              │ element step
                              ▼
            preset index ─► ordinal group (groups.py) ─► resolver ─► Laya actor (top-1, next-ranked on no effect)
                              ▼
                          execute (browser.py, hit-tested) ──► log ──► observe
```

Model calls per task: **1 compiler call** (0 on a cache hit; 2 only if the first output fails to parse), plus
**one actor call per element step not resolved deterministically**. No other LLM calls.

### 4.1 Program grammar (`jev_ultrafast/program.py`)

One subgoal per line: `KIND target`, `KIND target = value`, optional `@N` ordinal at the end of the target.

| Kind | Meaning | Page predicate (checks.py) |
|---|---|---|
| `FIND name` | reach the page about a named thing | title (site suffix and parenthetical stripped) or first h1 equals the name |
| `OPEN description [@N]` | click the link/tab described (the Nth of a repeated item; "top" is `@1`) | document URL changed since the subgoal started, after a click for it |
| `JUMP section` | go to a section of this page | URL fragment names the section, or a heading with that text is in the viewport |
| `SCROLL text` | scroll until the text is visible | a heading with that text is in the viewport, or the scroll tool found it |
| `FILL field = value` | type a value, then pick the matching suggestion if one appears | typed, and no matching suggestion is left unpicked |
| `SELECT field = value` | set a dropdown (native or custom listbox) | executed |
| `CLICK description` | press a control that need not navigate | executed with a page change |
| `SUBMIT [description]` | press Enter in the last field, or click the named button | executed with a page change |
| `DONE_WHEN text` | optional final line: text visible only when finished | text in page text or title |
| `DO goal` | fallback when compilation fails: goal-mode Laya (the baseline policy) | none (bounded by the step cap) |

Records: `Subgoal(kind: str, target: str, value: str = "", ordinal: int = 0)` and
`Program(subgoals: tuple[Subgoal, ...], done_text: str = "", source: str = "compiler")`, both frozen.
`parse_program(text) -> Program` raises `ValueError` on any unknown kind, empty target where one is required, more
than 8 subgoals, or over-long fields. `render_program(program) -> str` is its inverse (cache and logs).

### 4.2 Compiler (`jev_ultrafast/compiler.py`)

`compile_goal(goal, *, complete=complete_text, cache=None) -> tuple[Program, dict]`. System prompt: the grammar
plus four examples from domains *not* in the suite. Output is plain lines, `max_tokens` 96, temperature 0, thinking
disabled. One retry with the parse error appended; then `fallback_program(goal)` (`DO goal`). The goal is untrusted
data and the output is only parsed, never executed. Cache: JSON file keyed by the normalized goal in
`LAYA_CACHE_DIR` (default `~/.cache/laya-browser`); `LAYA_PROGRAM_CACHE=0` turns the program cache off. Model: `COMPILER_MODEL`, default `TEXT_MODEL`; the Mac benchmark
(D1) picks between Qwen3-4B-Instruct-2507 and Qwen3-1.7B.

### 4.3 Checks (`jev_ultrafast/checks.py`)

Pure functions over the observed page dict: `title_subject`, `is_about(page, name)`, `fragment_names(url, section)`,
`heading_in_view(page, text)`, `text_shown(page, text)`, and `satisfied(subgoal, page, start_url) -> bool | None`
(`None` = no page predicate; the controller decides from execution). All matching uses `resolver.normalize`
(case, accents, punctuation).

### 4.4 Tactics (`jev_ultrafast/tactics.py`)

`next_step(subgoal, page, elements, progress: Progress) -> Step | None`, with `Progress` a frozen record of what this
subgoal already did (`typed`, `submitted`, `searched`, `opened_search`, `picked`, `clicked`). Generic rules, no site
names:

- **FIND**: a link whose label resolves exactly to the name → click it. Else, if a search template is known for the
  host and not yet used → GOTO the rendered URL. Else a search field → type the name (opening a collapsed search
  control first if no field is visible); then SUBMIT (if Enter has no effect, click the search button); then click
  the result that names it (resolver, then actor).
- **OPEN**: click the described element; with `@N`, the Nth member of the repeated group that matches the description.
- **JUMP**: click the same-document link whose fragment names the section (deterministic); else go to the anchor of
  the observed heading with that text (FRAGMENT, id from the page, passed as data); else SCROLL_TO_TEXT.
- **SCROLL**: SCROLL_TO_TEXT.
- **FILL**: type the value into the described field; next, if an option naming the value is visible, click it.
- **SELECT**: a visible option naming the value → click it; else a native select → SELECT; else click the field to
  open it.
- **CLICK** / **SUBMIT**: click the described control / press Enter.

Instructions come from `instructions.instruction(kind, target, value)`, shared with training.

### 4.5 Groups (`jev_ultrafast/groups.py`)

`shape(label)` normalizes, replaces numbers with `#` and strips plurals (`48 comments` → `# comment`).
`ordinal_pick(elements, description, n)` groups click targets by (role, landmark, shape), keeps groups that share a
word with the description and have ≥ n members, prefers the larger group, and returns member n in page order. This
handles "the comments of the second story" without site rules. Known limit: a list whose members have different
label shapes ("discuss" for a story with no comments) shifts the count.

### 4.6 Search templates (`jev_ultrafast/search.py`)

`template_from_opensearch(xml, page_url)`, `learn_template(url, query)`, `render(template, query)`,
`matches_template(url, template)`, and `SearchTemplates` (JSON cache by host). `discover(browser, page_url)` fetches
the page's own OpenSearch description (same origin) once per host. `run_tool` accepts GOTO only if the URL matches a
template it was handed, and only on the page's own host.

### 4.7 Controller (`jev_ultrafast/controller.py`)

`Controller(goal, program, *, predict, fallback, discover, templates)` with `decide(page, history) -> dict` returning
the same decision shape as `Pilot.decide` (so `Agent` and `live_eval.py` are unchanged apart from the backend switch).
It exposes `plans` (the compile record, for `live_summary.py`), `actor_calls` and `trace` (per decision: subgoal,
route, check results). `Controller.from_goal(goal, **deps)` compiles. Selected with `POLICY_BACKEND=program`.

### 4.8 Browser changes

- `snapshot.js`: `headings` (≤ 200: text ≤ 80 chars, level, id, in_viewport) and `opensearch` (href or "").
- `Browser.press_enter()`: Enter key down/up via CDP to the focused element, then waits for navigation (as
  `navigate` does). Used by SUBMIT.
- `Browser.evaluate(expression, await_promise=False)` so OpenSearch discovery can `fetch`.
- Invariants unchanged: node ids from code, never from a model; freshness checked before input; execution logged
  before observing; no mutation retried.

### 4.9 What is deleted

Once the new backend has been compared live (D2) and is at least as good, delete: `pilot.py`, `planner.py`
(`plan`, `pick`, `search_query`, `planner_view`), the pick path of `router.py`, `verifier.py`,
`scripts/bench_planner.py`, `scripts/check_guards.py`, the static `SEARCH_TEMPLATES` in `tools.py`, and their tests.
They stay until then so that one checkout can run both backends on the same day (D5). Nothing else is removed.

## 5. Data and training

**Legal constraints (unchanged, restated).** Never train or tune on TypeSafe/Jev outputs (terms §2.3(b)); nothing in
this design calls Jev. Mind2Web is CC-BY-4.0: attribute it; train split only; the test splits are evaluation-only and
never enter training, prompts or examples. Qwen3 outputs (Apache-2.0 weights) may be used for relabelling. Explored
pages: public, read-only, `robots.txt` respected, ≤ 1 request/s per host, no logins or data-creating submissions;
only derived items (labels, roles, context strings) are stored, privately.

**Actor data, in order of expected value:**

1. **Step-mode Mind2Web items** (`training/step_items.py`): every usable train step becomes one target question
   whose `state.goal` is `instruction(kind, description, value)` from the serving templates; the description is the
   gold label with one word dropped with p = 0.3. The shortlist is queried with the instruction, as at serving. 30% of
   items keep the task goal (goal mode) so the `DO` fallback does not regress.
2. **Self-collected exploration items** (`scripts/explore.py`, no LLM): on each visited page, sample elements and
   render instructions from the page's own structure: label, ordinal within its group ("the 3rd '# comment' link"),
   section links, field labels/placeholders. Gold is the sampled element. Hosts of the 25-task suite are excluded, so
   the suite stays an honest test.
3. **Later (Phase D, user-approved):** hindsight descriptions from a local Qwen for executed steps (NNetNav-style),
   to teach paraphrase ("origin field" → `Where from?`).

**Training** stays on Kaggle 2×T4 with the existing `train_ddp.py` (proper-scoring-rule objective, then temperature
calibration). Selection on Mind2Web dev (held out by website); one confirmatory test run.

## 6. Speed budget per action (M2 16 GB)

| Stage | Budget | Basis | How it is measured |
|---|---:|---|---|
| Observe (snapshot.js via CDP) | 0.3 s | whole-document capture, ≤ 400 elements; not yet timed | `observe_ms` per step (new field in history) |
| Controller: checks + tactic + groups + resolver | < 0.01 s | pure Python over ≤ 400 elements | `decide_ms` minus `actor_ms` |
| Laya actor, when called (~50% of element steps expected) | 0.9 s | measured median 0.69–0.88 s | `actor_ms` per decision |
| Execute + settle (click, type, Enter) | 0.2 s | 50–200 ms settle in `observe()` | `act_ms` |
| Navigation wait when the click navigates | 0–3 s | network-bound; capped at 3 s | included in `act_ms` |
| Compiler, once per task, amortized over ~4 actions | 1.0–1.5 s | ~400 prompt tokens + ~40 output tokens; 40 / 11.5 tok/s ≈ 3.5 s decode + prefill; 0 on a cache hit | `plans[0].latency_ms` / actions |
| **Expected wall time per action** | **≈ 2–3.5 s** | vs. 16.5–19.1 s today | `wall_s_per_action` in `live_summary.py` |

Memory: macOS + Chrome ~5–6 GB, Laya fp16 ~0.9 GB, compiler 4-bit Qwen3-4B ~2.3 GB or Qwen3-1.7B ~1.0 GB. Using
the smaller compiler (if D1 shows it is valid on ≥ 90% of goals) frees ~1.3 GB, which matters under swap.

## 7. Mac system-use path

The controller only needs a surface that can (a) observe elements as dicts with `role`, `label`, `value`,
`operations`, `landmark`, `section`, `in_viewport`, plus `headings`/`title`/`url`-equivalents, and (b) act on an
element id with click/type/select, plus Enter and scroll. On macOS the Accessibility API (`AXUIElement` via pyobjc,
as in `awlevin/typesafe-computer-use`) provides roles, titles, values and window/group structure: window title →
`title`, app bundle → host, AX group titles → `section`, AX windows/sheets → `landmark`. The same grammar applies
(FIND → open app/document via Spotlight, OPEN/CLICK/FILL/SELECT/SUBMIT unchanged). Apps with poor AX trees would
need a grounding model (ShowUI-2B is the only candidate that fits in memory beside Laya); that is out of scope until
the browser path meets its gate.

## 8. Evaluation plan

1. **Unit and fixture tests (here, CI-like):** pure-function tests for every new module; an end-to-end fixture suite
   (local HTML pages for search → article → section, repeated-row lists, and a combobox form) driven through the real
   `Browser` and `Agent` with a fake lexical actor and a fake compiler. Skipped without Chrome.
2. **Compiler benchmark (D1, Mac):** the 25 suite goals plus 12 held-out goals: parse-valid rate, hand-judged correct
   program, latency, output tokens, for Qwen3-4B and Qwen3-1.7B. Gate: ≥ 90% valid, median ≤ 5 s.
3. **Actor offline (D3, Mac/Kaggle):** step-mode element accuracy and recall@20 on Mind2Web dev; goal-mode must not
   drop > 2 points below the published card.
4. **Live suite (D2, Mac):** `POLICY_BACKEND=program`, 3 runs per task, same day, same machine, plus one
   `POLICY_BACKEND=planner` run for a same-day reference. Report: pass count per run, per-task majority, category
   table, routes, model calls per task, medians of actor/compile/wall per action.
5. **Honesty about sample size.** With n = 25 and p ≈ 0.3, the standard error of the difference between two
   independent runs is ≈ 13 points, so differences under ~15 points (4 tasks) are not distinguishable, and even
   larger ones need the paired test. `scripts/compare_runs.py` reports discordant pairs and the exact McNemar p-value.
   **Success criteria:** majority-vote ≥ 15/25 (the existing Gate A+B bar, +7 tasks over the best prior run), paired
   McNemar p < 0.05 against the 2026-09-25 re-run, median wall time per action ≤ 4.0 s, median actor ≤ 1.0 s, ≤ 1
   LLM call per task on average (0 on cached goals). Anything less is reported as missed, with failure tags.
6. **Held-out tasks (D2):** 12 tasks on sites outside the suite (written and validated on the Mac before the first
   run of this backend). Reported separately; a gap of more than 15 points against the suite is flagged as possible
   overfitting of the grammar or its examples to the suite.

## 9. Risks

- **Compiler quality at 1.7B–4B.** A wrong program is a wrong run. Mitigations: tiny grammar, examples, strict
  parser, one retry, `DO` fallback; measured in D1.
- **Grammar coverage.** Goals outside the grammar fall back to goal-mode Laya (the 7/25 baseline behaviour).
- **Predicate false positives** (stopping too early, e.g. OPEN counting any navigation) and **false negatives**
  (never seeing FILL complete on a custom widget). Bounded by per-subgoal budgets; reported via `trace`.
- **Multi-field forms** (Google Flights custom comboboxes and date pickers) remain the weakest category; the design
  gives them a fair chance (FILL + suggestion pick, SELECT on custom listboxes) but expects at most 1–2 of 3.
- **Ordinals** depend on consistent label shapes within a list.
- **Overfitting to the suite** through the grammar and its examples; the held-out set is the guard.
- **Search discovery** fails where a site has neither OpenSearch nor a GET search; FIND then types into the search
  box, which is slower but general.
- **Live-site drift** between runs; same-day runs and the paired test limit it.

## 10. Deliberately left out

Per-step LLM planning; an LLM verifier; screenshots or vision models in the loop; RL or online weight updates;
workflow memory beyond deterministic caches; Kev (bake-off only, D6); multi-tab flows, logins, payments; the task
platform UI; Mac control (path in §7 only).

## 11. Sources

- AgentOccam (observation/action-space refinement): https://arxiv.org/abs/2410.13825
- Region4Web (functional-region observations): https://arxiv.org/abs/2605.07134
- Agent Workflow Memory: https://arxiv.org/abs/2409.07429
- Budget-matched study of skill/memory modules: https://arxiv.org/abs/2606.15017
- NNetNav (interaction first, hindsight relabel): https://arxiv.org/abs/2410.02907
- OS-Genesis (reverse task synthesis): https://arxiv.org/abs/2412.19723
- Explorer (94K synthesized trajectories): https://arxiv.org/abs/2502.11357
- SynWeaver (website-prior co-synthesis): https://arxiv.org/abs/2608.12429
- ShowUI-2B: https://arxiv.org/abs/2411.17465
- UI-TARS-1.5-7B 4-bit MLX: https://huggingface.co/mlx-community/UI-TARS-1.5-7B-4bit
- Fara-7B: https://www.microsoft.com/en-us/research/blog/fara-7b-an-efficient-agentic-model-for-computer-use/
- GUI-Actor (single-pass attention action head): https://arxiv.org/abs/2506.03143
- Kev and Kev-0.8B: https://github.com/jaredpalmer/kev, https://huggingface.co/jaredpalmer/kev-0.8b
- Laya browser checkpoint: https://huggingface.co/Quantum08/laya-browser-mind2web
- OpenSearch description format: https://github.com/dewitt/opensearch/blob/master/opensearch-1-1-draft-6.md
