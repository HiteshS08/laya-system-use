# Redesign overnight report — compile once, act with typed decisions

Date: 2026-09-26. Branch: `redesign/first-principles` (from `main` at `9b0300b`). Spec:
[`2026-09-26-first-principles-design.md`](../specs/2026-09-26-first-principles-design.md). Plan:
[`2026-09-26-first-principles.md`](../plans/2026-09-26-first-principles.md).

No live evaluation was run: this environment has no Apple Silicon, Laya checkpoint, model server or user Chrome, and
the brief rules out live evals. Nothing below is a success-rate claim. What was verified here: unit tests, and an
end-to-end fixture suite that drives the real `Agent`, `Browser` and `snapshot.js` in a local headless Chromium on
local HTML pages, with a fake compiler and a lexical fake actor.

## Core ideas

1. **The per-step LLM planner is gone.** It was the largest error source (39–55 wrong steps against 15–19 for the
   actor) and the whole speed problem: at ~11.5 tok/s, a 100–200-token JSON plan costs 9–17 s in decode alone.
2. **One LLM call per task.** A compiler turns the goal into a short line program in a closed grammar (`FIND`,
   `OPEN [@N]`, `JUMP`, `SCROLL`, `FILL … = …`, `SELECT … = …`, `CLICK`, `SUBMIT`, `DONE_WHEN`). The output is
   ~40 tokens, not JSON. It is cached by goal, so a repeated goal needs 0 LLM calls. If compilation fails, the
   program falls back to goal-mode Laya (`DO`), which is the old baseline behaviour.
3. **Completion is checked, not judged.** Each subgoal has a page predicate: the title or h1 names the entity; the
   URL fragment or a visible heading names the section; the field holds the value; the URL changed. The run stops
   the moment the last one holds. This removes the "reached the goal, kept clicking" failures seen on
   `mallon_to_typhoid`, `turing_references` and `*_external_links`.
4. **Laya always gets a step instruction, never the task.** The replay probe showed this fixes 5 of 6 recorded
   actor mistakes. The instruction templates live in `instructions.py`, which serving and training share.
5. **Cheap deterministic grounding comes first**: a preset element from a tactic, then ordinal groups ("the 2nd
   `# comment` link"), then an exact or fuzzy label match, and only then Laya. There is no LLM pick step. If an
   action has no effect, that element is excluded and the next choice is a new decision on a new observation; no
   mutation is ever retried.
6. **Per-site knowledge is discovered, not written.** Search templates come from the page's own OpenSearch
   description, or are learned from an observed search. I checked live (`curl`) that Wikipedia's description gives
   `…/w/index.php?title=Special:Search&search={q}`, and that an exact-title query 302-redirects to the article.
   The model never emits a URL; code renders it and `run_tool` checks it against the template.

Expected per-action wall time is ≈ 2–3.5 s (spec §6), against 16.5–19.1 s measured for the planner backend. This is
an estimate until D2 measures it.

## Tasks done

| Plan task | What | Commit(s) |
|---|---|---|
| Phase 1 | Spec and plan | `83b35c2` |
| 1 | Program grammar (`program.py`) | `2740866` |
| 2 | Compiler, `complete_text`, program cache (`compiler.py`, `textmodel.py`) | `6fb256d` |
| 3 | `headings` and `opensearch` in `snapshot.js`; completion checks (`checks.py`) | `a2f901d`, checkpoint A fix `332b4c6` |
| 4 | Ordinal groups (`groups.py`) | `0cbeca2` |
| 5 | Search templates (`search.py`), SUBMIT/Enter, host-checked GOTO, `evaluate(await_promise)` | `226655f`, `95ab1b2` |
| 6 | Instruction templates and tactics (`instructions.py`, `tactics.py`) | `109a8e3` |
| 7 | Controller, no LLM per step (`controller.py`); actor takes any step-like object | `e6fabc1`, checkpoint B fix `a1a5a4b` |
| 8 | `POLICY_BACKEND=program` in `Agent`; `observe_ms`/`act_ms`; `llm_calls`/`actor_calls`/`program`/`trace` in live records; `.env.example` | `dc10384` |
| 9 | End-to-end fixture suite in real Chromium (7 scenarios) | `5b417a2`, `ae688ad` |
| 10 | `live_summary.py` model-call and timing medians; `compare_runs.py` (majority, exact McNemar) | `55e7178` |
| 11 | `bench_compile.py` and 12 held-out goals on other sites | `747e9ce` |
| 12 | Step-mode Mind2Web items (`training/step_items.py`) | `554316b` |
| 13 | Exploration items without an LLM (`scripts/explore.py`) | `b3ef590` |
| Checkpoint D | FRAGMENT tool for JUMP; `@1` for "top" in the prompt | `ae688ad` |
| Docs | Deviations, spec aligned with what was built | `50fa79d` |

One commit went out red: `226655f` pinned the old `run_tool` call signature in an existing test. The next commit,
`95ab1b2`, fixed that test. My command chain did not gate the commit on pytest at that point; every later commit
went through a script that runs pytest and ruff first.

**Dependencies added: none.** OpenSearch parsing uses the standard library's `xml.etree`; robots.txt uses
`urllib.robotparser`.

**Nothing was deleted.** The planner stack (`pilot.py`, `planner.py`, the router's pick path, `verifier.py`,
`bench_planner.py`, `check_guards.py`, the static search registry) stays until D2 shows the new backend is at least
as good (D5), so that one checkout can run both backends on the same day.

## Tests and coverage

`uv run pytest -q` without Chrome: **372 passed, 16 skipped** (11 need Chrome; 5 need the Laya tokenizer in the
local Hugging Face cache). With the throwaway Chromium on port 9333: **383 passed, 5 skipped**.
`uv run ruff check .`: clean. `node --check jev_ultrafast/static/app.js`: clean. `uv build`: sdist and wheel
built.

`uv run pytest --cov=jev_ultrafast --cov-report=term-missing -q` (with Chromium):

| Module | Stmts | Miss | Cover |
|---|---:|---:|---:|
| actor.py | 44 | 0 | 100% |
| agent.py | 159 | 19 | 88% |
| browser.py | 149 | 20 | 87% |
| checks.py (new) | 41 | 1 | 98% |
| compiler.py (new) | 72 | 2 | 97% |
| controller.py (new) | 204 | 3 | 99% |
| groups.py (new) | 19 | 0 | 100% |
| instructions.py (new) | 3 | 0 | 100% |
| program.py (new) | 79 | 1 | 99% |
| search.py (new) | 78 | 1 | 99% |
| tactics.py (new) | 114 | 0 | 100% |
| tools.py | 26 | 0 | 100% |
| textmodel.py | 72 | 6 | 92% |
| demo.py (upstream UI, untested) | 103 | 103 | 0% |
| **Total** | **1920** | **176** | **91%** |

End-to-end fixture scenarios (`tests/test_e2e_fixture.py`, real browser, all passing):
- FIND via the search box: type, Enter, click the result. Exactly 3 actions.
- FIND then an on-page link, in 1 action.
- JUMP by a contents link, which sets `#References`.
- JUMP by a heading anchor with no contents link, which sets `#Legacy`.
- SCROLL until the heading is in view (scroll y > 1000).
- OPEN the 2nd repeated `N comments` item.
- FILL a combobox, pick the matching suggestion, click Search, and stop on `DONE_WHEN`.

## Deviations from the plan

These are recorded in full in the plan's "Deviations" section. In short:
- `learn_template` keeps the other parameters' original encoding.
- After a template search, FIND clicks the result instead of retyping.
- There is no "only text field" search fallback.
- Exclusions are keyed by page and label, not by element index.
- If Enter did nothing, FIND clicks the search button.
- `LAYA_PROGRAM_CACHE=0` turns the program cache off.
- Single-candidate training steps are skipped.
- The test-split guard uses Mind2Web's exact names.
- The FRAGMENT tool for JUMP was added.
- Bracketed `[edit]` suffixes are ignored in headings.
- `Progress.opened` was added for custom dropdowns.

## Open risks

- **Compiler quality** decides the run. With Qwen3-4B or 1.7B, a wrong program is a wrong run. D1 measures it on
  25 suite goals plus 12 held-out goals.
- **`wiki_featured` is a known trap.** `OPEN today's featured article` resolves exactly to the "Today's featured
  article" project-page link, not the featured article itself. No generic fix yet; I expect this task to fail.
- **Multi-field forms** (Google Flights) remain the weakest category. Custom date pickers and trip-type listboxes
  are only partly covered by FILL + suggestion pick and SELECT on custom listboxes.
- **Ordinals assume consistent label shapes** in a list. A story with "discuss" instead of "N comments" shifts the
  count.
- **Weak predicates.** OPEN completes on any effective click. A wrong but effective click ends the run early and
  wrong (still better than wandering, but still wrong).
- **Overfitting to the suite.** The grammar was designed with the suite's failures in view. The compiler's examples
  come from other domains (a test enforces that no suite goal appears in the prompt), and the 12 held-out goals are
  the guard. Their checks still have to be written and validated on the Mac (D2).
- **Not measured yet:** the actor on step instructions phrased from the *compiler's* descriptions (the probe used
  planner-style instructions); observe latency on real pages; OpenSearch fetches under the user's Chrome; Enter
  submission on real sites. The fixture suite covers only the mechanics.
- **n = 25 is small.** See the evaluation steps below for the paired test.

## Deferred to the Mac (exact commands)

Run everything from the repo root, on this branch:

```bash
git fetch origin && git checkout redesign/first-principles && uv sync
```

**D1: compiler model benchmark.** If `mlx_lm` is missing, run `uv sync --extra local-text` first.
```bash
uv run mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8080 &
COMPILER_MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit uv run --env-file .env python scripts/bench_compile.py > artifacts/bench_compile_4b.json
kill %1
uv run mlx_lm.server --model mlx-community/Qwen3-1.7B-4bit --port 8080 &
COMPILER_MODEL=mlx-community/Qwen3-1.7B-4bit uv run --env-file .env python scripts/bench_compile.py > artifacts/bench_compile_1_7b.json
kill %1
```
Hand-judge the programs in both files. Keep the smallest model with ≥ 90% valid and correct programs and a median
≤ 5 s, and set `COMPILER_MODEL` in `.env`.

**D2: live evaluation.** See the next section. Before it, write `evals/heldout_tasks.py` with a hand-validated
`LiveTask` check for each goal in `evals/heldout_goals.py`.

**D3: step-mode retraining** (Mind2Web train only; Kaggle 2×T4):
```bash
uv run python training/fetch_data.py train
uv run python training/step_items.py --input data/mind2web/data/train/*.json --out training/out/step_train.jsonl --dev-mod 20 --seed 0
uv run python training/prepare_items.py --cases training/out/step_train.jsonl --out training/out/step_items.pt
# upload training/out/step_items.pt to Kaggle and train with training/train_ddp.py as in training/KAGGLE.md, then:
uv run python training/evaluate.py --predictor laya --checkpoint checkpoints/laya_step --cases training/out/step_train_dev.jsonl
```

**D4: exploration.** Needs your approval of the seed sites; the suite's sites are excluded automatically.
```bash
uv run python scripts/explore.py --seeds https://docs.python.org/3/ https://developer.mozilla.org/en-US/ --pages 200 --out data/explore/items.jsonl
```

**D5 (only if D2 shows the program backend ≥ the planner backend):** delete the planner stack as listed in plan
D5 and make `program` the default.

**D6 (optional):** Kev-0.8B bake-off, after D3.

## Evaluate this branch against the 25-task suite (on the Mac)

1. Start a throwaway headless Chrome:
   ```bash
   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9333 --user-data-dir="$(mktemp -d)" --headless=new about:blank &
   ```
2. Check the mechanics on local pages first (about 2 s, no model needed):
   ```bash
   BU_CDP_URL=http://127.0.0.1:9333 uv run pytest tests/test_e2e_fixture.py tests/test_snapshot_live.py -q
   ```
3. Start the model server. The program backend needs one model, for the compile only (use D1's choice). The same
   server also serves the same-day planner reference in step 6.
   ```bash
   uv run mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 8080 &
   ```
4. Run the suite three times with the program backend. Run 1 compiles every goal fresh; runs 2–3 reuse the cached
   programs. Use a fresh cache directory so no earlier templates or programs leak in:
   ```bash
   export LAYA_CACHE_DIR=artifacts/cache-$(date +%Y%m%d)
   POLICY_BACKEND=program LAYA_PROGRAM_CACHE=0 uv run --env-file .env python scripts/live_eval.py
   POLICY_BACKEND=program uv run --env-file .env python scripts/live_eval.py
   POLICY_BACKEND=program uv run --env-file .env python scripts/live_eval.py
   ```
5. Summarise each run: pass counts by category, routes, `llm_calls_per_task`, `actor_calls_per_task`, and medians
   of actor, compile (`median_planner_ms`), observe, act and wall time per action:
   ```bash
   uv run python scripts/live_summary.py artifacts/live/<run>
   ```
6. Same-day reference with the old backend, then the paired comparisons. Per-task majority over the three program
   runs, against the reference and against the 2026-09-25 re-run:
   ```bash
   POLICY_BACKEND=planner ACTOR_TAU=0.5 uv run --env-file .env python scripts/live_eval.py
   uv run python scripts/compare_runs.py --a artifacts/live/<prog1> artifacts/live/<prog2> artifacts/live/<prog3> --b artifacts/live/<planner_today>
   uv run python scripts/compare_runs.py --a artifacts/live/<prog1> artifacts/live/<prog2> artifacts/live/<prog3> --b artifacts/live/20260925T124019Z
   ```
7. Success criteria (spec §8):
   - majority-vote ≥ 15/25;
   - McNemar p < 0.05 against the 2026-09-25 re-run;
   - median wall time per action ≤ 4.0 s;
   - median actor decision ≤ 1.0 s;
   - ≤ 1 LLM call per task on average.

   Tag failures by hand as before; each record now carries `program` and `trace`, which show which subgoal, tactic
   and route produced each step.
8. Stop the servers: `kill %1 %2` (or `pkill -f mlx_lm.server`, then close the throwaway Chrome).

## Environment notes

- `scripts/commit.sh` pushed from this environment without problems; every commit above went through it. No
  history was rewritten and nothing was force-pushed.
- The fixture tests used Playwright's bundled Chromium (`/opt/pw-browsers/chromium-1194/chrome-linux/chrome
  --headless=new --no-sandbox --remote-debugging-port=9333`). On the Mac, the throwaway Chrome from step 1 serves
  the same purpose.
