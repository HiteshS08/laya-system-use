# Brief A — first-principles redesign, then implementation (cloud session, Opus)

You are working unattended overnight in a cloud session on the GitHub repo **HiteshS08/laya-system-use** (private;
Python 3.12, `uv`). Work on branch **`redesign/first-principles`**, created from `main`. Never push to `main`.
A second cloud session is working in parallel on branch `feat/plan-once-per-page` (incremental fixes to the current
design). Do not touch that branch.

## What the project is
A local agent that completes tasks in the browser, and later on the Mac itself, for its user. It forks
`browser-use/jev-ultrafast` and replaces the hosted TypeSafe Jev decision model with an open-weight one. The main
pieces:
- **Live pipeline** (`jev_ultrafast/`): snapshot → shortlist → decision → action.
- **Planner+actor loop:** `pilot.py`, `planner.py`, `router.py`, `actor.py`, `resolver.py`, `memory.py`, `tools.py`.
- **Training pipeline** (`training/`): Mind2Web → Kaggle fine-tuning.
- **Live eval suite** (`evals/live_tasks.py`, `scripts/live_eval.py`, `scripts/live_summary.py`).

## The user's hard requirements (keep these; they are not assumptions)
1. **Local.** It runs on the user's MacBook Air M2 with 16 GB RAM, which already sits under heavy memory pressure
   (~10 GB of swap in use).
2. **Fast.** Speed is an explicit user priority. Current targets: ≤ 1.0 s per actor decision and ≤ 4.0 s wall time
   per action.
3. **Laya-style actor at the centre.** A fast, open-weight, single-forward-pass typed-decision model (Jev-style)
   must be central to choosing actions: Laya (`convaiinnovations/laya`, a ModernBERT-large encoder with typed
   heads) or a better open alternative such as Kev (`jaredpalmer/kev`, Qwen3.5 base with a pointer head). The
   specific checkpoint may change.
4. **No Jev distillation.** Never train or tune on TypeSafe/Jev outputs (their terms, section 2.3(b), forbid it).
   Mind2Web is CC-BY-4.0; its test splits are for evaluation only.
5. **Scope.** Browser now; Mac system use later.
6. **Git hygiene.** Commit only with `scripts/commit.sh "<conventional message>"`: small messages, no co-author
   line, no history rewriting, no force-push. Never commit `.env*`, `data/`, `checkpoints/` or `artifacts/`.

## Everything else is open to challenge
Treat every current design choice as an assumption, and re-derive it from first principles. At minimum,
challenge:
- the DOM-candidate pipeline and the K=20 lexical shortlist;
- Mind2Web-only training;
- having a local LLM planner in the loop at all, and the planner/actor split;
- how progress and completion are detected;
- step memory and recovery;
- the snapshot format (DOM, accessibility tree, screenshot, or a mix);
- the action set;
- when models are called;
- caching or replaying known workflows;
- per-site skills and tools;
- how evaluation is done.

Research freely on the web for current literature and open-weight models, and cite sources. Start with AgentOccam,
Agent Workflow Memory, NNetNav, OS-Genesis, Explorer, SynWeaver, ShowUI, UI-TARS and Kev. For each, check what
exists now and what fits in 16 GB alongside Chrome.

## Evidence (read first)
- Previous designs: `docs/superpowers/specs/2026-09-21-laya-agent-core-design.md`,
  `docs/superpowers/specs/2026-09-23-planner-actor-design.md`.
- Live results: `docs/superpowers/reports/2026-09-24-gate-ab.md` and
  `docs/superpowers/reports/2026-09-25-gate-ab-rerun.md`.
- Model card for the current actor: https://huggingface.co/Quantum08/laya-browser-mind2web.
- Measured facts:
  - Offline, macro element accuracy on the Mind2Web test splits is ~0.48–0.50.
  - Live, 25 tasks on real sites: baseline 7/25; after the planner+actor redesign 8/25, then 7/25.
  - Actor decision: 0.7–0.9 s.
  - Local Qwen3-4B planner: 12–20 s per call on this Mac (~11.5 tokens/s under swap).
  - Wall time: 16–19 s per action.
  - Wrong steps by cause: planner 39–55, actor 15–19.
  - Replay probe: with a step instruction as its goal, the actor chose right on 5 of 6 recorded failures.
  - Structural failure modes seen:
    - completion not detected because the view was truncated;
    - repeated scrolls;
    - the fallback navigating away;
    - planner confusing JSON fields;
    - fixes that removed one failure mode exposed the next.
- The recorded live runs (`artifacts/`) and checkpoints are local to the user's Mac and not in the repo.

## Phase 1: design (commit before writing any implementation code)
1. `docs/superpowers/specs/2026-09-26-first-principles-design.md`:
   - the problem, restated;
   - the first-principles analysis: every current assumption kept, changed or dropped, with reason and evidence;
   - the architecture and component interfaces;
   - data and training strategy, with the legal constraints;
   - a speed budget per action on the M2 16 GB, with numbers and how each is measured;
   - the Mac system-use path;
   - an evaluation plan with success criteria that are honest about sample size (on 25 tasks, differences under
     ~15 points are not distinguishable);
   - risks, and what is deliberately left out.
2. `docs/superpowers/plans/2026-09-26-first-principles.md`: bite-sized TDD tasks. Each task gives exact files,
   interfaces (names, parameter and return types), the test code to write first, the implementation code or
   precise requirements, commands, and a commit message.
   - Order the tasks so value lands early.
   - Mark every task that needs hardware this environment lacks (live browser runs, MLX, model downloads, Kaggle
     training) as **DEFERRED TO MAC**, with the exact commands.
   - Reuse existing modules where the analysis says they are right; say explicitly what gets deleted.
3. Commit both with `scripts/commit.sh "docs: add first-principles redesign spec and plan"`.

## Phase 2: implementation (same session, after Phase 1 is committed)
Implement every task not marked DEFERRED TO MAC, in plan order, test-first. For each task:
1. Write the test.
2. See it fail.
3. Implement.
4. See it pass.
5. Run `uv run pytest -q` and `uv run ruff check .`.
6. Commit with that task's message.

Keep functions under 50 lines, frozen dataclasses for records, type hints on public functions, logging rather than
print in library code, and no bare `except`. Add a dependency only if the spec justifies it and it installs on
macOS arm64 and Linux; note each one in the final report. Every model call in tests must be a fake.

After every 3–4 tasks, re-read your own diff since the last checkpoint and fix what you find. There is no separate
reviewer tonight. If a task turns out wrong once it meets the code, record the deviation in the plan file (a
"Deviations" section), make the smallest sound change, and carry on. Stop only if every path forward is a guess.

## This environment cannot
Use Apple Silicon (MLX), a local LLM server, a real Chrome session with the user's setup, the Laya checkpoint, or
the recorded live runs. Do not attempt live evals, model downloads over ~500 MB, Kaggle, or Hugging Face
publishing.

## Final deliverable
Commit `docs/superpowers/reports/2026-09-26-redesign-overnight.md`, covering:
- the core ideas;
- tasks done, with commits;
- tasks deferred to the Mac, with exact commands;
- test and coverage results (`uv run pytest --cov=jev_ultrafast --cov-report=term-missing -q`);
- deviations from the plan;
- open risks;
- the exact local steps to evaluate this branch against the 25-task suite. That means starting the throwaway
  headless Chrome on port 9333 and the model server(s) your design needs, the `scripts/live_eval.py` invocation,
  and `scripts/live_summary.py`.

If `scripts/commit.sh` cannot push from this environment, use `git commit -m` with no co-author trailer and
`git push -u origin redesign/first-principles`. Say so in the report.
