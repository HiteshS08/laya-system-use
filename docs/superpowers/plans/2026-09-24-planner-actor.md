# Planner + Actor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise live task success on the 25-task suite from 7/25 by putting a local planner in charge of the step,
operation and value, asking the fast actor (Laya, later Kev) only "which element" for that step, fixing the option set,
and then retraining the actor on step-conditioned, context-annotated data.

**Architecture:** A `Pilot` (per run) holds a step queue from the planner (Qwen3-4B via mlx-lm), a `StepMemory`, and
routes each step through a deterministic resolver, then the actor, then planner arbitration on low confidence. The
snapshot captures the whole page with context fields; pruning removes detours and duplicate links. The existing
`Agent` loop executes the decision; tools add `SCROLL_TO_TEXT` and registry-checked `GOTO`.

**Tech Stack:** Python 3.12, `uv`, pytest, ruff; Chrome via browser-harness CDP; `laya` package; mlx-lm server
(OpenAI-compatible); Kaggle CLI for training; Kev from its own checkout (Phase C only).

**Spec:** `docs/superpowers/specs/2026-09-23-planner-actor-design.md`

## Global Constraints

- No Jev/TypeSafe outputs as labels, for training, evaluation or tuning (TypeSafe terms 2.3(b)).
- Mind2Web test splits are evaluation-only; never in training, never in a public repo or Kaggle dataset.
- Package manager `uv` only; no new dependencies. Kev (Phase C) runs from its own checkout (`~/kev`) and is reached
  over its local HTTP server, so it is not added to `pyproject.toml`.
- `requires-python >=3.12`; ruff line length 120; type hints on public functions; frozen dataclasses for records;
  functions under 50 lines; `logging` not `print` in library code (CLI scripts may print); no bare `except`.
- Commits only via `scripts/commit.sh "<conventional message>"` (stages all, commits with no co-author line, pushes).
  Never raw `git commit` / `git push`. Check `git status --short` before committing; never commit `.env*`, `data/`,
  `checkpoints/`, `artifacts/`, `training/out/`.
- Coverage ≥ 80% on new modules: `uv run pytest --cov=jev_ultrafast --cov=training --cov-report=term-missing`.
- Live runs: throwaway Chrome (`--remote-debugging-port=9333 --user-data-dir=<tmp>`), read-only public sites, no
  logins, no form submissions that create data. `BH_TELEMETRY=0`.
- Speed targets (spec §3): median actor decision ≤ 1.0 s; median planner call ≤ 3.0 s; median wall time per executed
  action ≤ 4.0 s.
- Gates (spec §12): stop and report to the user at Task 15 (Gate A+B) and Task 22 (Gate C). Phase D (Tasks 23–24)
  starts only after the user approves a site list. Nothing is published to Hugging Face without user approval.

## Review Focus

1. **No element supports the planner's operation** (e.g. TYPE_TEXT on a page with no field): the step fails, is
   recorded as a failed attempt, and the planner re-plans. No crash, no random click. Test in Task 9.
2. **Several elements share the planner's `target_text`** after dedupe (two "Edit" buttons with different targets):
   the resolver defers to the actor instead of taking the first. Test in Task 3.
3. **Accented text** in goals, labels and evidence (Gödel, Zürich): resolver and evidence check match regardless of
   accents and case. Tests in Tasks 3 and 6.
4. **Evidence taken from the goal rather than the page**: rejected; `done` only with a quote the page shows. Test in
   Task 6.
5. **URL changes while planned steps are still queued** (a click navigates earlier than planned): the queue is
   discarded and the planner is called on the new page. Test in Task 9.

## File map

| File | Responsibility | Task |
|---|---|---|
| `jev_ultrafast/pruning.py` (new) | detour filter, duplicate-link merge | 1 |
| `jev_ultrafast/snapshot.js` | whole-page capture, context fields, outline | 2 |
| `jev_ultrafast/browser.py` | scroll-into-view before hit test, `navigate`, `scroll_to_text`, navigation wait | 2, 5, 11 |
| `jev_ultrafast/model.py` | `action_space` copies new element fields | 2 |
| `jev_ultrafast/resolver.py` (new) | text normalisation, exact/fuzzy label resolution | 3 |
| `jev_ultrafast/memory.py` (new) | StepMemory: attempts, exclusions, stall streak | 4 |
| `jev_ultrafast/tools.py` (new) | search-URL registry, GOTO validation, tool execution | 5 |
| `jev_ultrafast/textmodel.py` | `extra` request fields (thinking off) | 6 |
| `jev_ultrafast/planner.py` (new) | planner view, prompt, validation, evidence, top-5 pick | 6 |
| `jev_ultrafast/formatter.py` | `context_text`, context-aware `render_option` | 7, 16 |
| `jev_ultrafast/candidates.py` | `Candidate.context` | 7 |
| `jev_ultrafast/actor.py` (new) | single target question with step as goal | 7 |
| `jev_ultrafast/router.py` (new) | resolver → actor → planner pick | 8 |
| `jev_ultrafast/pilot.py` (new) | per-run step queue, memory, decision dicts | 9 |
| `jev_ultrafast/agent.py` | Pilot wiring, tool and planned-value execution | 10 |
| `scripts/diagnose_click.py` (new) | GitHub Issues hit-test repro | 11 |
| `scripts/capture_pages.py`, `scripts/bench_planner.py` (new) | planner model benchmark | 12 |
| `scripts/replay_probe.py` (new) | actor regression on the six probe decisions | 13 |
| `scripts/live_eval.py` | records routes, planner calls, backend | 14 |
| `training/mind2web.py`, `training/build_cases.py` | context fields, step conditioning, hard negatives | 16, 17 |
| `training/prepare_items.py`, `training/evaluate.py`, `training/fit_tau.py` (new) | oversampling, step mode, τ | 17, 18 |
| `training/kaggle/*` (new) | Laya and Kev Kaggle kernels | 21 |
| `training/kev_export.py` (new), `jev_ultrafast/kev_backend.py` (new) | Kev data and serving | 19, 20 |
| `explore/explorer.py`, `explore/relabel.py` (new) | Phase D data collection | 23, 24 |

## Tasks

Phase A+B — no retraining (`2026-09-24-planner-actor/part-1.md`, `part-2.md`, `part-3.md`)
1. Detour filter and duplicate-link merge
2. Whole-page snapshot with context fields; scroll into view before input
3. Resolver
4. StepMemory
5. Tools: search registry, GOTO validation, SCROLL_TO_TEXT, navigate
6. Planner
7. Actor (single target question, step as goal)
8. Router
9. Pilot
10. Agent wiring and planned values
11. Diagnose the GitHub Issues click
12. Planner model benchmark (4B vs 1.7B)
13. Actor replay regression script
14. Live eval records routes and planner calls
15. **Gate A+B:** live suite run and report

Phase C — retrained actor (`part-4.md`)
16. Context fields in the Mind2Web pipeline with train/serve parity
17. Step conditioning and hard negatives
18. Step-mode evaluation and τ fitting
19. Kev: licence, export, smoke training
20. Kev serving backend
21. Laya-v2 and Kev-0.8B training, offline evaluation, τ fit
22. **Gate C:** live comparison and report

Phase D — self-collected live data, gated (`part-5.md`)
23. Explorer
24. Hindsight relabelling and Laya-v3 items
