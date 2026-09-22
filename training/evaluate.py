"""Score a predictor on Mind2Web cases.

Micro metrics (op_acc, element_acc_given_op, element_acc_overall, step_success_*) pool all scored steps flat.
Macro metrics (*_macro, success_rate) group by task first, matching the MindAct paper's protocol (Deng et al.,
NeurIPS 2023, Table 2 caption: "step-wise metrics ... macro average across tasks") so numbers compare directly
against its published Element Accuracy / Operation F1 / Step SR / SR. A step outside a task's action space
(no_gold, op_not_allowed, gold_not_interactive with no reclaim target, gold_not_shortlisted) counts as a hard
failure for the macro metrics: the deployed agent has no way to produce the right action for it. A "trivial" step
(a single candidate for its operation) counts as correct without calling the predictor, matching what the
deployed policy actually does (jev_ultrafast/policy.py skips the model when there is only one candidate).
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

Predict = Callable[[dict], dict]
log = logging.getLogger("evaluate")


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def _macro(task_scores: Sequence[float]) -> float | None:
    return round(sum(task_scores) / len(task_scores), 4) if task_scores else None


def _forced_outcome(row: dict) -> bool | None:
    """Correctness for a row the predictor is never asked about. None means 'use the real prediction'."""
    if row["drop_reason"] == "trivial":
        return True
    if row["drop_reason"] is not None:
        return False
    return None


def evaluate_rows(rows: Sequence[dict], predict: Predict) -> dict:
    valid = [r for r in rows if r["valid_gold"]]
    scored = [r for r in valid if r["gold_in_shortlist"]]
    op_hits = element_hits = step_hits = errors = 0
    outcomes: dict[tuple[str, int], dict[str, bool]] = {}
    for row in scored:
        key = (row["task_id"], row["step"])
        try:
            pred = predict(row)
        except ValueError as exc:
            errors += 1
            log.warning("predictor failed on task %s step %s: %s", row["task_id"], row["step"], exc)
            outcomes[key] = {"element": False, "op": False, "step": False}
            continue
        op_ok = pred["operation"] == row["gold_op"]
        element_ok = pred["targets"].get(row["gold_op"]) == row["gold_id"]
        op_hits += op_ok
        element_hits += element_ok
        step_hits += op_ok and element_ok
        outcomes[key] = {"element": element_ok, "op": op_ok, "step": op_ok and element_ok}

    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    element_accs, op_accs, step_accs, successes = [], [], [], []
    for task_rows in by_task.values():
        elements, ops, steps = [], [], []
        for row in task_rows:
            forced = _forced_outcome(row)
            outcome = outcomes.get((row["task_id"], row["step"]), {"element": False, "op": False, "step": False})
            elements.append(forced if forced is not None else outcome["element"])
            ops.append(forced if forced is not None else outcome["op"])
            steps.append(forced if forced is not None else outcome["step"])
        element_accs.append(sum(elements) / len(elements))
        op_accs.append(sum(ops) / len(ops))
        step_accs.append(sum(steps) / len(steps))
        successes.append(float(all(steps)))

    return {
        "steps": len(rows), "valid_gold": len(valid), "scored": len(scored), "errors": errors,
        "op_acc": _rate(op_hits, len(scored)),
        "element_acc_given_op": _rate(element_hits, len(scored)),
        "element_acc_overall": _rate(element_hits, len(valid)),
        "step_success_scored": _rate(step_hits, len(scored)),
        "step_success_overall": _rate(step_hits, len(valid)),
        "tasks": len(by_task),
        "element_acc_macro": _macro(element_accs),
        "op_acc_macro": _macro(op_accs),
        "step_success_macro": _macro(step_accs),
        "success_rate": _macro(successes),
    }


def laya_predictor(agent) -> Predict:
    def predict(row: dict) -> dict:
        questions, ops = row["questions"], row["ops_available"]
        answers = agent.system_one(row["state"], questions)["answers"] if questions else {}
        if "operation" in answers:
            operation = answers["operation"]["choice"]
        else:
            operation = ops[0] if len(ops) == 1 else None
        targets = dict(row["sole"])
        for op in ops:
            head = answers.get(f"{op.lower()}_target")
            if head:
                targets[op] = head["choice"]
        return {"operation": operation, "targets": targets}

    return predict


def ranker_predictor() -> Predict:
    """No-model baseline: always CLICK, best lexically ranked element for each operation."""
    return lambda row: {"operation": "CLICK", "targets": row["top1_by_op"]}


def load_rows(path: Path, sample: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return random.Random(0).sample(rows, sample) if 0 < sample < len(rows) else rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--predictor", choices=["laya", "ranker"], required=True)
    parser.add_argument("--checkpoint", default="convaiinnovations/laya")
    parser.add_argument("--subfolder")
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--head-max-len", type=int)
    parser.add_argument("--sample", type=int, default=0, help="uniform random subset (seed 0); 0 means all rows")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = load_rows(args.cases, args.sample)
    if args.predictor == "laya":
        import laya

        agent = laya.load(args.checkpoint, subfolder=args.subfolder)
        if args.max_len:
            agent.cfg["max_len"] = args.max_len
        if args.head_max_len:
            agent.cfg["head_max_len"] = args.head_max_len
        predict = laya_predictor(agent)
    else:
        predict = ranker_predictor()
    result = {"predictor": args.predictor, "checkpoint": args.checkpoint if args.predictor == "laya" else None,
              "cases": str(args.cases), **evaluate_rows(rows, predict)}
    log.info(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
