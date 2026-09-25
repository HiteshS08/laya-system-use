# Brief B — continue the current design: below-fold fix + call the planner once per page (cloud session, Opus)

You are working unattended overnight in a cloud session on the GitHub repo **HiteshS08/laya-system-use** (private;
Python 3.12, `uv`). Create branch **`feat/plan-once-per-page`** from `main` and work only there; never push to
`main`. A second cloud session is working in parallel on branch `redesign/first-principles`; don't touch it.

## The system
A local browser agent. A local planner LLM (Qwen3-4B, via an OpenAI-compatible server) plans each step: the
operation, the target text, and the value. A fast typed-decision model (Laya, a ModernBERT classifier) picks which
page element a step means. A resolver matches labels exactly when it can.

Key code:
- `jev_ultrafast/pilot.py`: per-run control (plan queue, StepMemory, routing, fallbacks, search fallback)
- `planner.py`: prompt, view, parse/validate, `plan`, `pick`, `search_query`
- `memory.py`: attempts, failures, exclusions
- `router.py`, `actor.py`, `resolver.py`
- `tools.py`: SCROLL_TO_TEXT and the registered search GOTO
- `agent.py`: the loop that executes decisions

Spec: `docs/superpowers/specs/2026-09-23-planner-actor-design.md`. Live reports:
`docs/superpowers/reports/2026-09-24-gate-ab.md` and `docs/superpowers/reports/2026-09-25-gate-ab-rerun.md`.
Read the spec, both reports, and the modules above before changing anything.

## Where things stand
Three live runs of the 25-task suite scored 7, 8 and 7 out of 25; the gate needs ≥ 15.
- **Planner errors dominate:** 55 wrong steps against 15 for the actor.
- **Too slow:** each planner call takes ~12 s on the user's Mac, against targets of ≤ 3 s per call and ≤ 4 s per
  action.
- **Called too often:** the planner is re-called after every failed step and every time the queue empties.
- **Below-fold regression:** these tasks dropped from 5/5 to 1/5 in the re-run.

The below-fold regression has a traced cause. Take "Scroll down until the External links section heading is
visible":
1. The first `SCROLL_TO_TEXT "External links"` succeeds, which completes the task.
2. `planner_view` shows only the first `TEXT_CHARS` = 600 characters of visible text, and the scroll centres the
   heading, so the heading lies beyond those 600 characters.
3. The planner therefore cannot see evidence that the task is done.
4. It re-plans the same scroll, which is now excluded as a repeat.
5. The actor-only fallback then clicks unrelated controls.

## Work 1: below-fold fix
After a successful SCROLL_TO_TEXT, `planner_view` includes a `focus_text` field.
- It holds ~300 characters of the page's visible text, centred on the first case-insensitive occurrence of the
  scroll target.
- The Pilot passes the last successful scroll target.
- The prompt says evidence may come from `focus_text`.
- Confirm that the evidence check in `parse_plan` searches the full `page["text"]`.
- Stay within the existing budget test (≤ 4,800 characters for system prompt plus payload).
- Write the tests first.

## Work 2: call the planner far less
Add a short section to the spec, then implement test-first. Target: on average ≤ 1 planner call per distinct URL in
scripted Pilot scenarios. No existing behaviour in `tests/test_pilot.py` may be lost.
1. **Deterministic completion.** The plan gets an optional `done_when` field: a phrase of at most 80 characters
   that will appear in the page text or title once the goal is met.
   - Validate it like the other plan fields.
   - After each action, before any planner call, the Pilot checks for it using `resolver.normalize` against the
     full text and the title.
   - If it is present, the Pilot returns DONE with that phrase as evidence, and no planner call is made.
2. **Retry before replanning.** When a step had no effect, retry it once with the actor's next-ranked candidate.
   Replan only when that also fails, or when there is no alternative.
3. **Replan triggers:** a URL change, a queue that is exhausted without `done_when` being met, or rule 2
   exhausted. Nothing else triggers a replan.
4. **Longer plans:** raise `MAX_STEPS` from 3 to 5, and tell the planner to plan every step it can foresee on the
   current page.

Write a small test or script that replays scripted scenarios with fakes and counts `plan_fn` calls before and
after. Report both numbers.

## Rules
- Code style: functions under 50 lines, frozen dataclasses, type hints, ruff line length 120, no new dependencies.
- Before each commit: `uv sync`, then `uv run pytest -q` and `uv run ruff check .` must pass. Live-browser tests
  skip without Chrome; that is expected.
- Coverage on `pilot.py`, `planner.py` and `memory.py` must stay ≥ 80%.
- Commit only with `scripts/commit.sh "<conventional message>"`: no co-author line, no history rewriting, no
  force-push, and never `.env*`, `data/`, `checkpoints/` or `artifacts/`.
  - If `scripts/commit.sh` cannot push from this environment, use `git commit -m` with no co-author trailer, then
    `git push -u origin feat/plan-once-per-page`, and say so in the report.
- There is no separate reviewer tonight. After each work item, re-read your whole diff and fix what you find.
- This environment has no Apple Silicon/MLX, no planner server, no Chrome session with the user's setup, and no
  Laya checkpoint or recorded runs.
  - Do not attempt live evals, model downloads, Kaggle, or Hugging Face publishing.
  - All model calls in tests must be fakes.

## Final deliverable
Commit `docs/superpowers/reports/2026-09-26-plan-once-overnight.md` covering:
- what changed, per work item;
- the spec section you added;
- test and coverage results;
- planner-call counts before and after;
- open risks;
- the exact local commands for the live Gate A+B re-run:
  1. Start a throwaway headless Chrome:
     `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9333 --user-data-dir="$(mktemp -d)" --headless=new about:blank &`
  2. Start the model server:
     `uv run mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8080 &`
  3. Run the suite:
     `POLICY_BACKEND=planner ACTOR_TAU=0.5 uv run --env-file .env python scripts/live_eval.py`
  4. Summarise:
     `uv run python scripts/live_summary.py artifacts/live/<run>`
