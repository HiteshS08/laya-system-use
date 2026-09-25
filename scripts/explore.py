"""Actor items from exploring public pages, with no LLM: the page itself supplies the label of every item.

Each visited page yields a few items: sample an element with a unique label, describe it with the serving
instruction template, shortlist with the serving query, and keep the element as gold. The live suite's sites are
excluded so the suite stays a fair test. Read-only: robots.txt respected, >= 1 s between pages, no typing, no forms.
Usage: uv run --env-file .env python scripts/explore.py --seeds URL [URL ...] --pages 200 --out data/explore/items.jsonl
"""

import argparse
import json
import logging
import random
import time
from collections import Counter, deque
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urldefrag, urlsplit
from urllib.robotparser import RobotFileParser

from evals.live_tasks import TASKS
from jev_ultrafast.actor import actor_request
from jev_ultrafast.instructions import instruction
from jev_ultrafast.model import action_space
from jev_ultrafast.policy import candidates_from
from jev_ultrafast.pruning import prune_actions
from jev_ultrafast.resolver import normalize
from jev_ultrafast.shortlister import shortlist
from jev_ultrafast.tactics import Step
from training.step_items import describe

log = logging.getLogger("explore")
EDITABLE = frozenset({"textbox", "searchbox", "combobox", "spinbutton"})


def _site(host: str) -> str:
    return ".".join(host.split(".")[-2:])


EXCLUDED_SITES = frozenset(_site(urlsplit(t.url).hostname or "") for t in TASKS.values())


def excluded(url: str) -> bool:
    return _site(urlsplit(url).hostname or "") in EXCLUDED_SITES


def _item(page: Mapping, gold, pool, rng: random.Random) -> dict | None:
    op = "TYPE_TEXT" if "TYPE_TEXT" in gold.ops and gold.role in EDITABLE else "CLICK"
    pool = [c for c in pool if op in c.ops]
    description = describe(gold.label, rng)
    value = rng.choice(page.get("title", "").split() or ["test"]) if op == "TYPE_TEXT" else ""
    step = Step(op, description, value, instruction("FILL" if op == "TYPE_TEXT" else "OPEN", description, value))
    chosen = shortlist(f"{step.instruction} {step.target_text}", [], pool)
    if len(chosen) < 2 or gold.id not in {c.id for c in chosen}:
        return None
    state, questions = actor_request(step, chosen, [])
    name = next(iter(questions))
    return {"source": "explore", "website": urlsplit(page["url"]).hostname, "task_id": page["url"],
            "state": state, "questions": questions, "gold_id": gold.id, "gold_op": op, "mode": "step",
            "gold": {name: {"probabilities": {c.id: float(c.id == gold.id) for c in chosen}}},
            "valid_gold": True, "drop_reason": None}


def items_from_page(page: Mapping, rng: random.Random, n: int = 8) -> list[dict]:
    elements, _, _ = action_space(prune_actions(page["actions"], ""))
    pool = candidates_from(elements)
    counts = Counter(normalize(c.label) for c in pool)
    golds = [c for c in pool if len(c.label.strip()) >= 2 and counts[normalize(c.label)] == 1]
    items = []
    for gold in rng.sample(golds, len(golds)):
        item = _item(page, gold, pool, rng)
        if item is not None:
            items.append(item)
        if len(items) >= n:
            break
    return items


def same_site_links(page: Mapping) -> list[str]:
    host = urlsplit(page["url"]).hostname
    links = [urldefrag(a.get("href") or "")[0] for a in page["actions"] if a.get("kind") == "click"]
    return list(dict.fromkeys(u for u in links if u.startswith("http") and urlsplit(u).hostname == host))


def _allowed(url: str, robots: dict[str, RobotFileParser]) -> bool:
    root = f"{urlsplit(url).scheme}://{urlsplit(url).netloc}"
    if root not in robots:
        robots[root] = RobotFileParser(root + "/robots.txt")
        try:
            robots[root].read()
        except OSError:
            log.warning("no robots.txt at %s; skipping the site", root)
            robots[root].disallow_all = True
    return robots[root].can_fetch("laya-browser-explorer", url)


def crawl(seeds: list[str], pages: int, out: Path, seed: int) -> Counter:
    from jev_ultrafast.browser import Browser

    rng, robots, stats = random.Random(seed), {}, Counter()
    queue, seen = deque(u for u in seeds if not excluded(u)), set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        while queue and stats["pages"] < pages:
            url = queue.popleft()
            if url in seen or excluded(url) or not _allowed(url, robots):
                continue
            seen.add(url)
            browser = Browser(url)
            try:
                page = browser.observe(screenshot=False)
            finally:
                browser.close()
            for item in items_from_page(page, rng):
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                stats["items"] += 1
            stats["pages"] += 1
            queue.extend(same_site_links(page))
            time.sleep(1.0)
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--pages", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("data/explore/items.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("explored: %s", dict(crawl(args.seeds, args.pages, args.out, args.seed)))


if __name__ == "__main__":
    main()
