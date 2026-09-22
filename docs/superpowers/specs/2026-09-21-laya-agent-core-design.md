# Laya agent core: fine-tuned policy + local text model

Status: draft for review · 2026-09-21 · Sub-project 1 of 3

## Scope

This spec covers the **agent core** only: a browser agent forked from `browser-use/jev-ultrafast` (MIT) that
chooses actions with a fine-tuned open model (Laya, Apache-2.0) and writes field text with a local model, with no
hosted decision API. The **task platform** (submit a task, watch it run, get a result) and **Mac system control**
(`awlevin/typesafe-computer-use`, MIT) get their own specs and consume the interfaces defined here.

## Constraints (verified 2026-09-21)

- **No Jev outputs as training labels.** TypeSafe customer terms 2.3(b) prohibit using the Services or any Output
  for distillation, imitation training, or a competing product. The recorded spike traces
  (`spike/jev/spike/traces/`) contain Jev answers; they are kept by decision but are never training data. Jev is not
  used in evaluation either.
- **Data licences.** Mind2Web (`osunlp/Mind2Web`) is CC-BY-4.0: attribute it in the model card. Its test splits ship
  as a password-protected zip to prevent contamination: they are evaluation-only and never enter training or a
  public repo. WebLINX (CC-BY-NC-SA) and Multimodal-Mind2Web (OpenRAIL) are excluded.
- **Hardware.** M2, 16 GB, no CUDA. Training runs on Kaggle 2×T4 (user's Kaggle and Hugging Face accounts).
  Inference runs locally on MPS.
- **No third-party model with an undeclared licence.** The MindAct DeBERTa candidate ranker declares none, so it is
  not used.

## Architecture

Five units. Each is testable on its own and talks to the others through the interface shown.

| Unit | Job | Interface |
|---|---|---|
| `formatter` | Turn Jev's observed element table into a Laya request: goal and last 3 actions in the state, one short label-first string per option (`label (role, =value)`) | `(goal, history, page, elements) -> (state, questions)` |
| `shortlister` | Rank elements against goal and history, keep the top K=20 with stable ids. First version is lexical (word overlap, position tie-break) | `(goal, history, elements) -> elements[:K]` |
| `policy` | Fine-tuned Laya. Same output shape as Jev's `choose()`, so the agent loop is unchanged | `decide(page, goal, history) -> {operation, target, probabilities, confidence}` |
| `verifier` | Local text model answers "is the goal visibly satisfied?" and decides `DONE`. Runs after a page change or when policy confidence is low | `(goal, page_text) -> {done: bool, reason}` |
| `text` | Local text model replacing `field_text`; same contract, `{"text": str}` | `(context) -> (value, meta)` |

Flow per step: `observe -> shortlister -> formatter -> policy -> (verifier if needed) -> text (if TYPE_TEXT) -> execute`.
`formatter` and `shortlister` are shared by training-data generation and serving, so train and serve inputs match.

The policy chooses among `CLICK`, `TYPE_TEXT` and `SELECT` plus a target. `SCROLL_*`, `WAIT` and `BLOCKED` stay
rule-based (as today), and `DONE` belongs to the verifier. Mind2Web has no labels for any of those.

## Data pipeline

- **Source:** `osunlp/Mind2Web` train: 1,009 tasks, about 6.5 GB of raw JSON in 11 shards. Per step it holds
  `cleaned_html`, `pos_candidates`, `neg_candidates` (tag plus attributes) and `operation {op, original_op, value}`.
- **Cases:** one case per step. Goal is `confirmed_task`, history is the earlier `action_reprs`, options are the
  shortlisted candidates rendered by `formatter`. Gold is the operation and the index of the positive candidate.
  Steps whose gold element is not in the shortlist are dropped from training and counted for the recall report.
- **Operations:** map `TYPE`, `SELECT`, `CLICK` directly. How `HOVER` and other `original_op` values are handled is
  decided when the case builder is written, after looking at their frequency; unmapped steps are dropped and counted.
- **Splits:** about 5% of train tasks, held out by website, become a dev set for checkpoint selection. The test
  splits are untouched until the final evaluation.
- **Where it runs:** locally, one shard at a time. The recall gate has to run before any training, the builder must
  share `formatter`/`shortlister` with serving, and only the small tokenised `items.pt` files go to Kaggle. (Revised
  during planning; the first draft built cases inside the Kaggle notebook.)

## Training

Adapt Laya's own Kaggle DDP notebook (proper-scoring-rule policy gradient on gold labels, then temperature
calibration). Start from `convaiinnovations/laya` (ModernBERT-large, English). Raise the budgets to
`max_len=768`, `head_max_len=448` so 20 options of about 20 tokens fit (the item-preparation step logs the real
length percentiles; if p99 reaches 768 the budgets are revisited before training). Push the checkpoint to the user's
Hugging Face repo under Apache-2.0, with Mind2Web attribution in the model card.

## Text model

- **Choice:** `Qwen3-4B-Instruct-2507` (Apache-2.0), 4-bit MLX, served by `mlx-lm`'s OpenAI-compatible server on
  localhost. About 2.5 GB, so it fits beside Laya in 16 GB. Fallbacks: `Phi-4-mini-instruct` (MIT), `SmolLM3-3B`
  (Apache-2.0). Also benchmark `Qwen3-1.7B` and `Qwen3.5-2B` and keep the smallest that passes the field-value test.
- **Change to `field_text`:** drop the DeepSeek/OpenRouter reasoning parameters, add tolerant JSON extraction with one
  retry, keep the existing validation (exactly one key `text`, non-empty string, at most 2,000 characters). Env config
  stays `TEXT_MODEL_BASE_URL` / `TEXT_MODEL`, pointed at localhost.
- **To verify at implementation time:** that the 2507 instruct variant never emits thinking output, how `mlx-lm`
  handles JSON output, and resident memory with Laya loaded.

## Evaluation and decision gates

1. **Shortlister recall@K** on train, dev and test. Reported to the user before any training starts. If the gold
   element is often missing at K=20, the ceiling is low and a trained ranker comes first.
2. **Held-out Mind2Web** (`test_task`, `test_website`, `test_domain`): element accuracy, operation accuracy, step
   success. Reported beside two baselines: zero-shot Laya in the same compact format, and the shortlister's own top-1
   with no model. Published MindAct numbers are taken from the paper for context, not from memory.
3. **Live pages:** the six spike tasks plus more read-only tasks, with steps labelled by hand. No Jev in this step.
4. **Text model:** a field-value test set scoring valid-JSON rate, correctness and latency.

**Gate:** if fine-tuned Laya does not beat the shortlister-only baseline on held-out element accuracy, stop and
reconsider the policy (a generative open model) before building the platform on it.

## Risks and open questions

- **Completion detection is unproven.** Mind2Web has no post-final-action page, so no honest `DONE` labels exist.
  The verifier is a proposal. The spike showed Laya over-predicts `DONE`, which is why it is taken away from the policy.
- **Distribution shift.** Mind2Web pages are static, older snapshots; live DOM differs from Mind2Web's candidates.
  `formatter` normalises both to label plus role, and live evaluation measures the gap.
- **Shortlister ceiling.** A lexical ranker may miss the gold element on vague goals ("open the featured article").
- **Kaggle limits.** Session length and weekly GPU quota may force splitting training across sessions.
- **Small spike evidence.** The earlier numbers came from 18 decisions on 6 tasks; the held-out evaluation replaces them.
- **Fork hygiene.** Keep Jev's MIT licence and attribution in the fork; keep `BH_TELEMETRY=0` in the default env.

## Out of scope

Task platform (UI, API, queue), a planner that turns free-form tasks into start URLs, Mac system control,
multilingual support, and any screenshot-based policy.
