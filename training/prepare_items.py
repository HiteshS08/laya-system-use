"""Tokenise cases into Laya's training-item format (ids, markers, target distribution)."""

import argparse
import json
import logging
import os
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

log = logging.getLogger("prepare_items")
DEFAULT_MODEL = "convaiinnovations/laya"
MAX_LEN = 768
HEAD_MAX_LEN = 448


def load_tokenizer_and_cfg(model_dir: str, max_len: int, head_max_len: int):
    from laya.agent import _fix_tokenizer_config
    from transformers import AutoTokenizer

    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    cfg = json.loads(Path(model_dir, "rl_agent_config.json").read_text())
    return tok, {**cfg, "max_len": max_len, "head_max_len": head_max_len}


def build_item(tok, cfg: dict, state: dict, question: dict, gold_q: dict) -> dict | None:
    from laya.common import QTYPES, build_sequence

    criteria = question["criteria"]
    keys = list(criteria)
    target = [float(gold_q["probabilities"].get(key, 0.0)) for key in keys]
    total = sum(target)
    if total <= 0:
        return None
    target = [v / total for v in target]
    spec = {"t": "choice", "ins": question["instructions"], "crit": criteria}
    seq, markers = build_sequence(tok, state, spec, cfg["max_len"], cfg["head_max_len"])
    if len(markers) != len(keys):  # options were cut off by the budget
        return None
    return {"ids": seq, "markers": markers, "qtype": QTYPES["choice"], "target": target,
            "label": target.index(max(target))}


def build_items(tok, cfg: dict, rows: Sequence[dict]) -> tuple[list[dict], Counter]:
    items: list[dict] = []
    stats: Counter = Counter()
    for row in rows:
        if row["drop_reason"] is not None:
            stats["rows_skipped"] += 1
            continue
        for qid, gold_q in row["gold"].items():
            item = build_item(tok, cfg, row["state"], row["questions"][qid], gold_q)
            if item is None:
                stats["items_dropped_overflow"] += 1
            else:
                items.append(item)
    stats["items"] = len(items)
    return items, stats


def main(argv: list[str] | None = None) -> None:
    import torch
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="items file, e.g. training/out/train_items.pt")
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--head-max-len", type=int, default=HEAD_MAX_LEN)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tok, cfg = load_tokenizer_and_cfg(snapshot_download(DEFAULT_MODEL), args.max_len, args.head_max_len)
    rows = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    items, stats = build_items(tok, cfg, rows)
    if not items:
        raise SystemExit(f"No training items built from {args.cases}; stats={dict(stats)}")
    lengths = sorted(len(item["ids"]) for item in items)
    log.info("items=%d stats=%s length p50=%d p99=%d max=%d", len(items), dict(stats),
             statistics.median(lengths), lengths[int(0.99 * (len(lengths) - 1))], lengths[-1])
    torch.save(items, args.out)
    args.out.with_suffix(".meta.json").write_text(json.dumps(
        {"max_len": args.max_len, "head_max_len": args.head_max_len, "n_items": len(items), "stats": dict(stats)}))


if __name__ == "__main__":
    main()
