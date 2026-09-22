"""CLI: Mind2Web JSON shards -> Laya cases (jsonl) plus a summary."""

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from jev_ultrafast.shortlister import DEFAULT_K
from training.mind2web import iter_tasks, summarize, task_rows

log = logging.getLogger("build_cases")


def is_dev_website(website: str, mod: int) -> bool:
    return mod > 0 and int(hashlib.md5(website.encode()).hexdigest(), 16) % mod == 0


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(paths: Sequence[Path], out_dir: Path, name: str, k: int, dev_mod: int, limit_tasks: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    main_rows: list[dict] = []
    dev_rows: list[dict] = []
    for count, task in enumerate(iter_tasks(paths)):
        if limit_tasks and count >= limit_tasks:
            break
        (dev_rows if is_dev_website(task["website"], dev_mod) else main_rows).extend(task_rows(task, k))
    _write_jsonl(out_dir / f"{name}.jsonl", main_rows)
    summary = {name: summarize(main_rows)}
    if dev_mod > 0:
        _write_jsonl(out_dir / f"{name}_dev.jsonl", dev_rows)
        summary[f"{name}_dev"] = summarize(dev_rows)
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("training/out"))
    parser.add_argument("--name", default="train")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--dev-mod", type=int, default=0, help="hold out websites where md5 %% mod == 0; 0 disables")
    parser.add_argument("--limit-tasks", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = build(args.input, args.out_dir, args.name, args.k, args.dev_mod, args.limit_tasks)
    log.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
