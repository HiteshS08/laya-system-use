# How proven Jev-based browser agents draft questions, vs. laya-system-use

Research date: 2026-09-26. Read-only; no code changed.

## What we compared

Ours:
- old (upstream-derived): `jev_ultrafast/questions.py`, `jev_ultrafast/model.py::choose`, `jev_ultrafast/formatter.py` (main repo, `/Users/hiteshs/laya-browser`)
- new (redesign we're adopting): `jev_ultrafast/actor.py`, `jev_ultrafast/instructions.py`, `jev_ultrafast/tactics.py`, `jev_ultrafast/candidates.py`, `jev_ultrafast/formatter.py` (`/Users/hiteshs/wt-redesign-first-principles`)
- model card: https://huggingface.co/Quantum08/laya-browser-mind2web — ModernBERT-large, typed choice heads, `head_max_len` ~448 tokens over ~20 options, fine-tuned on Mind2Web with a label-first option format.

External:
- `browser-use/jev-ultrafast` (our direct upstream)
- `jkudish/jev-browser`
- `AnotiaWang/awesome-jev`
- `awlevin/typesafe-computer-use`

## Per-project findings

### browser-use/jev-ultrafast (our upstream — `jev_ultrafast/questions.py`, `model.py`, `docs/design.md`)

Our old `questions.py`/`model.py` **is** this upstream, verbatim (confirmed byte-identical `questions.py` on GitHub). One TypeSafe request per step with a dynamic number of questions: an `operation` Choice over every currently-available operation (`CLICK`/`TYPE_TEXT`/`SELECT`/named controls/`DONE`/`BLOCKED`), plus one `{op}_target` Choice per operation that has candidates. Per `docs/design.md`:

> "One TypeSafe request asks which operation to perform and which target would be appropriate for each available operation... The questions run independently: a target cannot read the operation answer, so its premise explicitly names the operation it assumes."

Target criteria embed `current_value` plus `checked`/`selected`/`expanded` when present (`model.py::ELEMENT_FIELDS`). Dropdowns get a composite id `index:option_number` — a two-stage choice folded into one flat id space rather than a literal second call. History is `{action, kind, text, page_changed}` for the last 10 steps.

> "DONE requires visible evidence that ALL requirements are satisfied... BLOCKED means no supported operation can make progress." (`questions.py::NEXT_ACTION`)

DONE/BLOCKED compete as options *inside* the same `operation` Choice as every clickable element's operation — not as an independent check.

### jkudish/jev-browser

Splits state into `PageElement`s tagged with a **structural** `kind` (`click`/`type`/`select`/`submit`/`search`/`fill_password`) decided by markup, not by the model — `input[type=search]`/`role=searchbox` become `search_eN` (fill+submit fused), `button[type=submit]` becomes `submit_eN` never `click_eN`, so the trace always shows an explicit submit decision (`src/lib.ts`).

Per step it asks **three independent questions in one fan-out call** — not sequential turns:

```ts
// src/questions.ts
action: choice("Which single action best advances the task on the current page?", criteria),
goal_done: noul("The task's goal has been achieved...", {...}),
stuck: noul("The actions so far are not making progress...", {...}),
```

> "the questions cannot see each other's answers, which is what makes `goal_done` an honest cross-check on the action Choice rather than a rationalization of it." (`src/questions.ts` docstring)

Select is a genuine **two-stage** call: the primary `action` Choice can pick `select_eN`; only then is a second request sent:

```ts
// src/questions.ts
export function selectOptionQuestion(elementDescription: string, options: string[]) { ... }
```

README: "Elements come from the DOM directly, not the accessibility tree, because accessibility trees under-report inputs; the agent found DuckDuckGo's search box only after this switch."

### awlevin/typesafe-computer-use (`docs/how-a-step-works.md`)

One request, but split into **3 mutually-exclusive Choices in one call** (`kind`, then `item`/`site`/`offscreen` depending on `kind`), explicitly to keep the option space clean:

> "Splitting the decision into three questions keeps screen noise out of the action choice. Every stall found while building this came from two options that meant the same thing. Confidence measures concentration, so overlapping options always read as doubt. Keep the action set mutually exclusive."

Repeated-label disambiguation, quoted verbatim:

> "An `ax` item reads as `button 'Share' (top-right)` in the criteria, so the classifier can tell a real control from a line of text. A label that appears more than once carries its row as well: `'Buy' (middle-right; in the row of 'Coldplay', 'Oct 2')`, since the label says nothing about which and the layout does."

**Yes/no verification after acting**, directly answering the "noul verification" question:

> "After typing, a Noul scores whether the field now holds a sensible value. Under 0.5 the field gets back the value it had before... Recovery never presses keys: the focus may have moved to another field." (`type_text` in "Where free text comes from")

No such post-write check exists anywhere in our old or new pipeline.

### awesome-jev / open Jev-style models (survey, no code read beyond README)

- **PocketJev**: "Camera + 3-choice, no text generation" — small closed option sets, no free text, matches Laya's design direction (typed heads, small model writes nothing).
- **PlayJev**: "a probability over the game's option list read off the option letters, no generated text" — label/letter-first readout, same family as Laya's "label-first option format."
- **Kev**, **LitJev**: reproduce the same `/v1/systemone` schema (Choice/Score/Noul) on open weights with no training — evidence that Choice+Noul fan-out, not just Choice, is the common shape for open Jev-likes.

## Comparing to our two codebases

**Old (`jev_ultrafast/model.py::choose`)**: identical to upstream — one `operation` Choice competing directly against per-operation `{op}_target` Choices, DONE/BLOCKED folded into the operation head. This is the architecture Laya was *not* built for: Laya has no operation head and a hard ~448-token/~20-option budget, so feeding it upstream's dynamic multi-question, unbounded-operations request would blow the budget and ask a head that doesn't exist. `os.environ.get("POLICY_BACKEND") == "laya"` already routes around this in `model.py`, confirming the team knows `choose()` isn't Laya's path.

**New (`actor.py`/`tactics.py`)**: already converges on the same idea proven by `jev-browser` and `typesafe-computer-use` — decide operation/kind deterministically in code (`tactics.py::next_step`, structural `SEARCH_ROLES`/`OPTION_ROLES`, `_search_field` vs `_search_opener`), and ask Laya exactly **one** Choice per step (`actor.py::pick_target`, single `{op}_target` question). This mirrors `jev-browser`'s structural `kind` tagging and `typesafe-computer-use`'s "keep the action set mutually exclusive" principle. Good convergent design — no other project asks Laya more than it needs to.

**Gaps found, by evidence:**

1. No goal/stuck/DONE cross-check independent of the target Choice. `jev-browser`'s `goal_done`/`stuck` Nouls are asked *in the same call*, not sequentially, specifically so they can't rationalize the action choice. Our redesign has no DONE/verification question at all in `actor.py`/`tactics.py` (search turned up no `noul` usage); the old path folds DONE into the operation softmax, which dilutes it against every element candidate — the exact "two options that mean the same thing" antipattern `typesafe-computer-use` warns against.

2. No post-action verification. `typesafe-computer-use` verifies `type_text` outputs with a Noul and reverts on low score. Our `tactics.py::_fill` and `actor.py` have no equivalent — a wrong `TARGET`/`TEXT_VALUE` pick is only caught by the next step's fresh observation, not checked immediately.

3. `Candidate` (`candidates.py`) carries only `id, label, role, value, ops, context` — no `checked`/`selected`/`expanded`/`disabled`. The old `model.py::ELEMENT_FIELDS` carried `checked/selected/expanded`. Checked `training/mind2web.py` for these fields directly — none present. Since `formatter.py`'s own docstring states render_option is "One prompt format for training and serving," any option-format change here changes Laya's training distribution too.

## Recommendations, ranked by expected gain

1. **[needs retraining]** Add checked/selected/expanded/disabled state back into `Candidate`/`render_option` (`candidates.py`, `formatter.py`) for checkboxes, radios, and disabled controls. Old `model.py` had this; new `Candidate` dropped it, and Mind2Web fine-tuning data (`training/mind2web.py`) never encoded it either — this is a genuine train/serve format gap, not just an inference tweak, per `formatter.py`'s own "one prompt format for training and serving" contract. Risk: retraining cost; but without it Laya can't distinguish "check this box" from "click this row" on ambiguous controls, which is exactly the kind of confusable-option pair `typesafe-computer-use` calls the #1 source of stalls.

2. **[inference-only, safe]** Add a code-side post-`TYPE_TEXT`/`FILL` verification step in `tactics.py`/`actor.py` that re-observes the field's actual value and compares it to the intended value (deterministic string check, not a new Laya head) before advancing — modeled on `typesafe-computer-use`'s Noul-then-revert, but doable without touching Laya at all since equality-checking a DOM value is exactly the kind of fact tactics.py already computes in code (`Progress.typed`, `progress.picked`). Pure inference/orchestration change in the redesign worktree; no retraining.

3. **[needs retraining, but cheap]** If a DONE/verification signal is wanted from Laya itself rather than only from code, add it as a **separate Noul-style question in the same request** (`jev-browser`'s fan-out pattern), never folded into the target Choice's option list — folding DONE into a Choice with element candidates is the old/upstream pattern and is the specific antipattern `typesafe-computer-use` names. This needs a new head/label-scheme and thus retraining, but is architecturally the right place for it given Laya's typed-head design; keep it independent so it can't rationalize a bad target pick.

4. **[inference-only, safe]** Reuse `jev-browser`'s and `typesafe-computer-use`'s structural pre-filtering discipline more aggressively in `tactics.py`: where two candidates would only be distinguishable by phrasing (e.g., a search-like text field vs. a generic field, or a submit button vs. a generic clickable), resolve it in code the way `_search_field`/`_search_opener` already do, rather than let both reach Laya's option list. This directly reduces the "two options that mean the same thing" failure mode `typesafe-computer-use` names as its top cause of stalls, with zero effect on the trained option-rendering format.

5. **[inference-only, safe]** Where two shortlisted candidates share an identical rendered label (`render_option` truncates to `MAX_LABEL_CHARS=70`), fall back to the same landmark/section/row context (`context_text` in `formatter.py`) unconditionally rather than only `with_context=context` gated by `ACTOR_CONTEXT=1` in `actor.py::actor_request`. This is the exact repeated-label disambiguation `typesafe-computer-use` documents (`'Buy' (middle-right; in the row of 'Coldplay', 'Oct 2')`) and our `context_text` mechanism already implements it — it's currently opt-in behind an env flag, so identical-label collisions go unresolved by default. Since `render_option(..., with_context=True)` is the same function already exercised for training-adjacent rendering, verify against `training/mind2web.py` before flipping the default; likely inference-only since the field already exists and is just gated off, but confirm the training data used `with_context=True` consistently before relying on it universally.

## Sources

- `/Users/hiteshs/laya-browser/jev_ultrafast/questions.py`, `model.py`, `formatter.py`
- `/Users/hiteshs/wt-redesign-first-principles/jev_ultrafast/actor.py`, `instructions.py`, `tactics.py`, `candidates.py`, `formatter.py`
- `/Users/hiteshs/wt-redesign-first-principles/training/mind2web.py`
- https://huggingface.co/Quantum08/laya-browser-mind2web
- https://github.com/browser-use/jev-ultrafast — `docs/design.md`, `jev_ultrafast/model.py`, `jev_ultrafast/questions.py`
- https://github.com/jkudish/jev-browser — `src/questions.ts`, `src/lib.ts`, `README.md`
- https://github.com/awlevin/typesafe-computer-use — `docs/how-a-step-works.md`
- https://github.com/AnotiaWang/awesome-jev — README (PocketJev, PlayJev, Kev, LitJev, JEVfire entries)
