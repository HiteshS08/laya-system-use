# Planner + Actor: closing the live-page gap (sub-project 1)

Date: 2026-09-23. Status: draft for review. Supersedes nothing; builds on
`2026-09-21-laya-agent-core-design.md` (all 12 tasks of that plan are done).

## 1. Problem

The Mind2Web fine-tune scores 0.48–0.50 macro element accuracy offline, but the live evaluation (Task 11,
`scripts/live_eval.py`) completed 1 of 6 tasks with 4 of 38 steps correct. The traces in `artifacts/live/` show
the failures are mostly structural, not model-quality:

| Task | Observed | Cause |
|---|---|---|
| wiki_search | clicked the logo twice, then "View source" | no decomposition into "type the query into Search"; the actor matched words, not intent |
| wiki_featured | clicked "View source" four times | no memory that the action did not help; loop detector keys on fingerprint, which kept changing |
| hn_comments | clicked the top-nav "comments" link | several identical labels; the candidate format carries no section, landmark, or link target |
| flights | typed into "Where from?" again instead of "Where to?" | no plan state: nothing tracks which fields are done |
| gh_issues | clicked "Issues 146" (right element), URL unchanged | unconfirmed; possibly the click landed on a child node. Needs diagnosis, not an assumed fix |
| (verifier) | 2 of 3 DONE verdicts wrong, 4–17 s per call | generic "is the goal satisfied" prompt on raw page text |

Mind2Web gives the actor a step-sized decision with the right element somewhere in its shortlist. Live, the
actor gets only the top-level goal and must also plan, remember, and judge completion. It was never trained
for that.

## 2. Goals and non-goals

Goals
- A local planner that decomposes the goal into small steps, tracks progress, and decides done/blocked.
- A fast single-pass decision model (the "actor") that executes each step. The actor stays; speed is a
  requirement. The specific checkpoint is not: Laya and Kev are compared and the better one is adopted.
- Confidence routing: the actor acts alone when calibrated-confident, the planner arbitrates otherwise.
- Candidates carry enough context to disambiguate identical labels, with identical formatting in training
  and serving.
- A live evaluation suite large enough to measure change, with programmatic success checks.

Non-goals (later sub-projects, each with its own spec)
- `tools/` shortcuts, `skills/` site notes, trajectory memory/replay.
- macOS control through the accessibility tree.
- Task-submission UI/API/queue.
- Screenshot-based models (Qwen3-VL, UI-TARS).

## 3. Success criteria

Measured on the new live suite (section 5.1), one run per task per configuration, same day, same machine.

- **Primary:** task success rate of the chosen configuration is at least 30 points above the baseline agent
  (current `main`, `POLICY_BACKEND=laya`) and at least 50% absolute.
- **Speed:** median actor latency at most 1.0 s per decision (M2, warm); median wall time per action
  (planner calls amortised, text generation included) at most 3 s. Both reported per configuration.
- **Honesty:** every reported number states its sample size. With about 25 tasks, differences under about
  15 points are reported as "not distinguishable", not as wins.
- **Offline:** the adopted actor does not regress on Mind2Web test splits (goal-only mode) versus the
  published checkpoint by more than 2 points macro element accuracy.

If the primary criterion is missed, the result is still reported with a per-failure breakdown
(planner error / actor error / actuation / site change) and no further tuning happens without a new decision.

## 4. Architecture

```
goal ──► Planner (Qwen3-4B, mlx-lm) ──► PlanStep{instruction, operation, target_hint, value}
            ▲         │ status: continue | done | blocked (+ evidence)
            │         ▼
            │   Actor (Laya | Kev) on shortlisted candidates (step-conditioned)
            │         │ choice + calibrated confidence
            │         ▼
            │   Router: confidence ≥ τ ──► execute
            │           confidence < τ ──► Planner picks from actor top-5 ──► execute
            │         │
            └── StepMemory (url, op, label) outcomes; re-plan triggers
```

Only the model behind `predict(state, questions)` changes between actors. Laya and Kev both implement the
System One request shape (`state`, `questions` with `choice` criteria), so the existing `decide()` path is
reused.

## 5. Components

### 5.1 Live evaluation suite — `evals/live_tasks.py`, `scripts/live_eval.py`

- About 25 tasks on read-only public sites, no logins, no purchases, no form submission that creates data.
  Categories with at least 3 tasks each: site search, in-page navigation, disambiguation (repeated labels),
  multi-field forms (flights-style, stop at results), below-the-fold targets, multi-page flows (2+ URLs).
- Each task has a programmatic check evaluated on the final page: `url_regex`, `text_regex`, or both. No model
  judges success. Checks are written and dry-run against a manual walkthrough before any agent run.
- Each run records per step: planner call (prompt, output, latency), actor request/answer/confidence, route
  taken, executed action, URL before/after, text value, and a failure tag filled in afterwards by hand:
  `planner | actor | route | actuation | site`.
- The existing six tasks are kept, and their checks are added.
- **First deliverable of the sub-project:** suite plus a baseline run of the current agent. Nothing else is
  built until the baseline exists.

### 5.2 Planner — `jev_ultrafast/planner.py`

- Input (JSON user message): goal; URL and title; visible text (at most 2,500 characters); up to 40 candidate
  labels with context; completed steps; failed attempts from StepMemory.
- Output (JSON, validated): `status` (`continue|done|blocked`), `evidence` (a quoted page snippet, required
  when `done`), `steps` (1–3 `PlanStep`s for the current page). `PlanStep` = `instruction` (one sentence),
  `operation` (`CLICK|TYPE_TEXT|SELECT`), `target_hint` (expected label text), `value` (string, only for
  TYPE_TEXT/SELECT).
- `done` is accepted only if `evidence` appears verbatim in the page text. This replaces `verifier.py` in
  the loop; the module stays for comparison runs only.
- When it runs: at the start, on URL change, when the step queue is empty, and after a step fails (no page
  change, or StepMemory rejection). It does not run on every action. Target: at most 1 planner call per
  2 actions on average (reported).
- Model: Qwen3-4B-Instruct-2507-4bit (already served). Qwen3-8B-4bit is a configuration option, run with
  `chat_template_kwargs: {"enable_thinking": false}`, if the planner is the dominant failure tag.
- Invalid output: one retry (existing `complete_json`), then fall back to actor-only on the raw goal for
  this page, logged as `planner_fallback`.

### 5.3 Actor — `jev_ultrafast/actor.py`

- `Actor = Callable[[dict, dict], dict]` (System One `predict`). Backends: `laya` (existing in-process
  `laya_predict`) and `kev` (in-process load of `kev` weights through its package, or its local
  `/v1/systemone` server; decided in the plan after checking Kev's loading API). Selected by
  `ACTOR_BACKEND=laya|kev`.
- Request state gains a `step` field: `{"goal", "step", "recent_actions"}`. `step` is the planner
  instruction. `goal` stays the full task, so the actor is not blind when the plan is wrong.
- Shortlisting uses goal + step + target_hint as the query. K stays 20 for Laya (head budget); for Kev, K is
  a measured parameter (20 and 40), since it accepts up to 255 options in an 8k-token state.

### 5.4 Router — in `policy.decide`

- Accept the actor's action when `target_confidence ≥ τ` and (if asked) operation confidence ≥ τ_op.
- Otherwise send the actor's top-5 targets (labels + context) to the planner as a single `choice` prompt.
  The planner's pick is validated against the offered ids.
- τ is fitted on Mind2Web **dev** (never test) per actor: the largest acceptance rate at which accepted-step
  accuracy is ≥ 0.90. Reported: acceptance rate and accepted accuracy on dev, and the live routed fraction.

### 5.5 Candidate context — `snapshot.js`, `candidates.py`, `formatter.py`, `training/mind2web.py`

- New fields per candidate: `landmark` (nearest of `nav|header|footer|aside|main|form|dialog`, by tag or
  role), `section` (text of the nearest preceding heading in the same landmark, at most 40 characters),
  `href_path` (link path without the query string, at most 40 characters).
- `Candidate` gains `context: str` (defaults to empty, so existing tests keep constructing it).
  `render_option` appends e.g. ` [nav]` or ` [main › From today's featured article] /wiki/Mary_Mallon`.
  Label stays first (the Laya budget truncates from the end).
- `training/mind2web.py` derives the same three fields from the raw Mind2Web DOM with identical rules and
  truncation. A parity test renders one fixture through both paths and asserts equal strings.
- Laya budget check: re-run `prepare_items` and report overflow count. If more than 1% overflow at K=20,
  drop `href_path` for Laya only and record that.

### 5.6 Step memory and loop guard — `jev_ultrafast/memory.py`

- Records `(url_without_fragment, operation, normalised label, value)` with outcome `changed_url |
  changed_page | no_change`.
- Excludes from the candidate set, for the current URL: any action already executed twice on this URL, and
  any action whose previous execution produced `no_change`. Excluded actions are listed to the planner as
  failed attempts.
- Replaces the fingerprint-only 3-strike rule with: 3 consecutive actions without a URL change **and**
  without planner-confirmed progress → `blocked`.

### 5.7 Known defects handled in this sub-project

- **Typed value `"false"` (flights):** values come from `PlanStep.value` when present. `field_text` output
  is rejected unless it is a non-empty `str`; JSON booleans and nulls raise the existing "nothing typed"
  error. Regression test included.
- **gh_issues click with no navigation:** diagnosed first (systematic debugging on the recorded trace plus
  a live repro of the hit-test point) before any change to `browser.py`. The fix, if any, gets its own test.

## 6. Actor bake-off

Training data: the existing Mind2Web pipeline plus two changes, applied identically to both actors:
1. Candidate context (5.5).
2. Step conditioning: each training item gets a `step` string with probability 0.5, templated from the gold
   action (`Click "{label}"`, `Type "{value}" into "{label}"`, `Select "{value}" in "{label}"`). The label
   text is lightly perturbed (a random word dropped with p=0.3) so the actor learns to use the hint without
   needing an exact string match. The other 50% of items keep goal-only state, which preserves goal-only
   ability.

Runs (Kaggle 2×T4 through the existing CLI flow):
- **Laya-v2:** `train_ddp.py` on the new items, same schedule as the published checkpoint.
- **Kev-0.8B-m2w:** `kev.train` (LoRA + pointer head) from `jaredpalmer/kev-0.8b` on the same items exported
  to Kev's JSONL format. Kev documents training on H100 and M5, not T4. The plan opens with a 50-step smoke
  run on Kaggle; if T4 (no bf16) fails, fall back to fp16/fp32 settings or to local M2 LoRA training for
  0.8B, and report the time cost.
- Kev-4B is out of scope unless Kev-0.8B beats Laya-v2 offline and still misses the live target. Its measured
  latency (721 ms uncached on M5, likely slower on M2) is close to the budget.

Offline evaluation (Mind2Web test splits, single confirmatory run, thresholds from dev only):
- goal-only mode (comparable to the published card), and
- oracle-step mode (step string from gold): the upper bound for "actor given a perfect plan", which separates
  actor error from planner error.

Live evaluation matrix: {baseline, Laya-v2, Kev-0.8B} × {actor only, planner + router}. Adopt the
configuration with the best live success within the speed budget. Ties go to the faster one.

## 7. Resource budget (M2, 16 GB)

| Component | Memory |
|---|---|
| macOS + Chrome (1 tab) | ~5–6 GB |
| Planner Qwen3-4B 4-bit (8B option) | ~2.3 GB (4.6 GB) |
| Actor: Laya fp16 or Kev-0.8B bf16 (one at a time) | ~0.9 / ~1.7 GB |
| Headroom | ≥ 5 GB (≥ 3 GB with 8B) |

## 8. Error handling

- Planner or text server unreachable: fail the run loudly with the endpoint and operation. No silent
  fallback to a different model.
- Actor answer not among the offered ids: existing `ValueError`, no action executed.
- Planner JSON invalid after retry: `planner_fallback` for this page (5.2), counted in reports.
- `done` without verbatim evidence: treated as `continue`, logged.

## 9. Testing

- Unit (pytest, fakes for planner/actor/LLM): planner output validation and evidence check; router
  thresholds and top-5 arbitration; StepMemory exclusion and blocked rule; context rendering; train/serve
  parity fixture; text-value type guard; Kev request/response adapter.
- Coverage ≥ 80% on new modules (`planner`, `actor`, `memory`, changed `formatter`/`policy`).
- Live suite is an evaluation, not CI. Run manually with a throwaway Chrome profile.

## 10. Carried constraints

From `.superpowers/sdd/.../global-constraints.md`: no Jev outputs as labels (TypeSafe terms 2.3(b)); Mind2Web
test is evaluation-only; `uv` only; ruff line length 120; commits via `scripts/commit.sh`; training only on
Kaggle (or local if the Kev T4 fallback triggers). New dependency: the `kev` package (Apache-2.0). Its Qwen3.5
base licence is checked and recorded in `NOTICE.md` before any weights are pulled.

## 11. Gates

1. Suite + baseline results → report to user.
2. Offline bake-off (both actors, both modes) → report to user; user confirms which actors go live.
3. Live matrix → report, adopt, update model card(s) and publish only on user approval.

## 12. Open risks

- Planner quality at 4B may be the new ceiling; the 8B option is included, and larger models do not fit.
- Kev training on T4 is unverified (6).
- Live sites change between runs; each configuration runs within the same session window, and site-change
  failures are tagged separately.
- Step-conditioned training could make the actor over-trust wrong hints; oracle vs goal-only eval and the
  live `planner` vs `actor` failure tags will show this.
