"""Score a predictor on Mind2Web cases: operation accuracy, element accuracy given the gold operation, step success."""

import argparse
import json
import logging
import random
from collections.abc import Callable, Sequence
from pathlib import Path

Predict = Callable[[dict], dict]
log = logging.getLogger("evaluate")


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def evaluate_rows(rows: Sequence[dict], predict: Predict) -> dict:
    valid = [r for r in rows if r["valid_gold"]]
    scored = [r for r in valid if r["gold_in_shortlist"]]
    op_hits = element_hits = step_hits = errors = 0
    for row in scored:
        try:
            pred = predict(row)
        except ValueError as exc:
            errors += 1
            log.warning("predictor failed on task %s step %s: %s", row["task_id"], row["step"], exc)
            continue
        op_ok = pred["operation"] == row["gold_op"]
        element_ok = pred["targets"].get(row["gold_op"]) == row["gold_id"]
        op_hits += op_ok
        element_hits += element_ok
        step_hits += op_ok and element_ok
    return {
        "steps": len(rows), "valid_gold": len(valid), "scored": len(scored), "errors": errors,
        "op_acc": _rate(op_hits, len(scored)),
        "element_acc_given_op": _rate(element_hits, len(scored)),
        "step_success_scored": _rate(step_hits, len(scored)),
        "step_success_overall": _rate(step_hits, len(valid)),
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
