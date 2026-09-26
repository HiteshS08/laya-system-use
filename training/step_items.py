"""Step-mode actor items from Mind2Web train: the actor's goal is the serving instruction, not the task.

At serving the actor always gets a templated step instruction (jev_ultrafast/instructions.py) and a shortlist
queried with it; these items reproduce exactly that. A share of items keeps the task goal (goal mode) so the
goal-mode fallback does not regress. Train split only: paths naming a test split are refused.
Usage: uv run python training/step_items.py --input data/mind2web/data/train/*.json --out training/out/step.jsonl
"""

import argparse
import contextlib
import hashlib
import json
import logging
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from jev_ultrafast.actor import actor_request
from jev_ultrafast.formatter import render_history_item
from jev_ultrafast.instructions import instruction
from jev_ultrafast.shortlister import DEFAULT_K, shortlist
from jev_ultrafast.tactics import Step
from training.fetch_data import TEST_SPLITS
from training.mind2web import ParsedStep, iter_tasks, parse_step

log = logging.getLogger("step_items")
KIND_FOR_OP = {"CLICK": "OPEN", "TYPE_TEXT": "FILL", "SELECT": "SELECT"}


def describe(label: str, rng: random.Random, drop_p: float = 0.3) -> str:
    """The gold label as a planner-style description: sometimes one word short, so exact matching is not enough."""
    words = label.split()
    if len(words) >= 2 and rng.random() < drop_p:
        del words[rng.randrange(len(words))]
    return " ".join(words)


def _step(parsed: ParsedStep, task_goal: str, goal_mode: bool, rng: random.Random) -> Step:
    if goal_mode:
        return Step(parsed.op, "", "", task_goal)
    description = describe(parsed.gold.label, rng)
    return Step(parsed.op, description, parsed.value, instruction(KIND_FOR_OP[parsed.op], description, parsed.value))


def step_row(task: dict, index: int, parsed: ParsedStep, history: Sequence[str], rng: random.Random,
             *, goal_mode_p: float = 0.3, k: int = DEFAULT_K) -> dict | None:
    gold = parsed.gold
    if gold is None or not parsed.gold_interactive or parsed.op not in gold.ops:
        return None
    pool = [c for c in parsed.pool if parsed.op in c.ops]
    goal_mode = rng.random() < goal_mode_p
    step = _step(parsed, task["confirmed_task"], goal_mode, rng)
    # The same shortlist query as serving: goal + history in goal mode, instruction + description in step mode.
    chosen = (shortlist(step.instruction, history, pool, k) if goal_mode
              else shortlist(f"{step.instruction} {step.target_text}", [], pool, k))
    if len(chosen) < 2 or gold.id not in {c.id for c in chosen}:
        return None
    state, questions = actor_request(step, chosen, history)
    name = next(iter(questions))
    return {
        "task_id": task["annotation_id"], "website": task["website"], "domain": task["domain"],
        "goal": task["confirmed_task"], "step": index, "state": state, "questions": questions,
        "gold": {name: {"probabilities": {c.id: float(c.id == gold.id) for c in chosen}}}, "gold_id": gold.id,
        "gold_op": parsed.op, "mode": "goal" if goal_mode else "step", "valid_gold": True, "drop_reason": None,
    }


def task_step_rows(task: dict, rng: random.Random, goal_mode_p: float) -> list[dict]:
    rows, history = [], []
    for i, raw in enumerate(task["actions"]):
        parsed = parse_step(raw)
        row = step_row(task, i, parsed, history, rng, goal_mode_p=goal_mode_p)
        if row is not None:
            rows.append(row)
        history.append(render_history_item(parsed.op, parsed.gold.label if parsed.gold else "", parsed.value))
    return rows


def _is_dev(website: str, mod: int) -> bool:
    return mod > 0 and int(hashlib.md5(website.encode()).hexdigest(), 16) % mod == 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal-mode-p", type=float, default=0.3)
    parser.add_argument("--dev-mod", type=int, default=0, help="hold out websites where md5 %% mod == 0")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if any(p.name.startswith(TEST_SPLITS) or "test" in p.parts for p in args.input):
        raise SystemExit("Mind2Web test splits are evaluation-only; refusing to build training items from them.")
    rng, counts = random.Random(args.seed), Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dev_path = args.out.with_name(args.out.stem + "_dev.jsonl")
    with args.out.open("w") as train, (dev_path.open("w") if args.dev_mod else contextlib.nullcontext()) as dev:
        for task in iter_tasks(args.input):
            for row in task_step_rows(task, rng, args.goal_mode_p):
                split = "dev" if _is_dev(task["website"], args.dev_mod) else "train"
                (dev if split == "dev" else train).write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[f"{split}_{row['mode']}"] += 1
    log.info("rows: %s", dict(counts))


if __name__ == "__main__":
    main()
