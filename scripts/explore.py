"""Actor items from exploring public pages, with no LLM: the page itself supplies the label of every item.

Each visited page yields a few items: sample an element with a unique label, describe it with the serving
instruction template, shortlist with the serving query, and keep the element as gold. The live suite's sites are
excluded so the suite stays a fair test. Read-only: robots.txt respected (with the same user agent Chrome sends),
max(1 s, Crawl-delay) between pages, no typing, no forms, and action-like links (logout, delete, edit, tokens) skipped.
It never runs in your own Chrome: it needs a throwaway one given by --cdp-url or LAYA_EXPLORE_CDP_URL.
Usage: uv run --env-file .env python scripts/explore.py --cdp-url http://127.0.0.1:9333 --seeds URL [URL ...]
       --pages 200 --out data/explore/items.jsonl
"""

import argparse
import json
import logging
import os
import random
import re
import time
from collections import Counter, deque
from collections.abc import Mapping
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urldefrag, urlsplit
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

from evals.live_tasks import TASKS
from jev_ultrafast.actor import actor_request
from jev_ultrafast.instructions import instruction
from jev_ultrafast.model import action_space
from jev_ultrafast.policy import candidates_from
from jev_ultrafast.pruning import prune_actions
from jev_ultrafast.resolver import normalize
from jev_ultrafast.search import TOKEN_PARAM
from jev_ultrafast.shortlister import shortlist
from jev_ultrafast.tactics import Step
from training.step_items import describe

log = logging.getLogger("explore")
EDITABLE = frozenset({"textbox", "searchbox", "combobox", "spinbutton"})
ROBOTS_TOKEN = "laya-browser-explorer"
USER_AGENT = f"{ROBOTS_TOKEN}/0.1 (read-only research crawler)"
MIN_DELAY_SECONDS = 1.0
USER_CHROME_PORTS = frozenset({9222, 9223})  # browser-harness probes these for the user's own Chrome
_ACTION_PATH = re.compile(r"log-?out|sign-?out|unsubscribe|delete|(?:^|/)edit(?:/|$)", re.IGNORECASE)


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


def action_like(url: str) -> bool:
    """Links that may change state or carry credentials: never visited, even read-only."""
    parts = urlsplit(url)
    params = parse_qsl(parts.query, keep_blank_values=True)
    return bool(_ACTION_PATH.search(parts.path)) or any(
        name.lower() == "action" or TOKEN_PARAM.search(name) for name, _ in params)


def same_site_links(page: Mapping) -> list[str]:
    host = urlsplit(page["url"]).hostname
    links = [urldefrag(a.get("href") or "")[0] for a in page["actions"] if a.get("kind") == "click"]
    return list(dict.fromkeys(u for u in links if u.startswith("http") and urlsplit(u).hostname == host
                              and not action_like(u)))


def _root(url: str) -> str:
    return f"{urlsplit(url).scheme}://{urlsplit(url).netloc}"


def _read_robots(root: str) -> RobotFileParser:
    """robots.txt fetched with the crawler's own user agent (RobotFileParser.read would send Python's)."""
    parser = RobotFileParser(root + "/robots.txt")
    try:
        with urlopen(Request(root + "/robots.txt", headers={"User-Agent": USER_AGENT}), timeout=10) as response:
            parser.parse(response.read().decode("utf-8", errors="replace").splitlines())
    except HTTPError as exc:
        # As RobotFileParser.read: 401/403 forbid the site, any other 4xx means no robots.txt (allowed).
        if exc.code in (401, 403) or exc.code >= 500:
            log.warning("robots.txt at %s failed with HTTP %s; skipping the site", root, exc.code)
            parser.disallow_all = True
        else:
            parser.allow_all = True
    except (URLError, OSError) as exc:
        log.warning("no robots.txt at %s (%s); skipping the site", root, exc)
        parser.disallow_all = True
    return parser


def _allowed(url: str, robots: dict[str, RobotFileParser]) -> bool:
    root = _root(url)
    if root not in robots:
        robots[root] = _read_robots(root)
    return robots[root].can_fetch(ROBOTS_TOKEN, url)


def crawl_delay(url: str, robots: Mapping[str, RobotFileParser]) -> float:
    parser = robots.get(_root(url))
    delay = parser.crawl_delay(ROBOTS_TOKEN) if parser is not None else None
    return max(MIN_DELAY_SECONDS, float(delay or 0))


def throwaway_cdp_url(url: str | None) -> str:
    """The explorer's Chrome must be a throwaway one, never the user's: not the harness's default ports and not
    the BU_CDP_URL the user configured for their own runs."""
    parts = urlsplit(url or "")
    if parts.scheme not in ("http", "https", "ws", "wss") or not parts.hostname or parts.port is None:
        raise ValueError(f"Need a throwaway Chrome CDP URL like http://127.0.0.1:9333, got {url!r}")
    if parts.port in USER_CHROME_PORTS or url == os.environ.get("BU_CDP_URL"):
        raise ValueError(f"{url} may be your own Chrome; start a throwaway one on another port")
    return url


def _observe(url: str) -> Mapping:
    from jev_ultrafast.browser import Browser

    browser = Browser("about:blank")
    try:
        browser.call("Emulation.setUserAgentOverride", userAgent=USER_AGENT)
        browser.navigate(url)
        return browser.observe(screenshot=False)
    finally:
        browser.close()


def crawl(seeds: list[str], pages: int, out: Path, seed: int) -> Counter:
    rng, robots, stats = random.Random(seed), {}, Counter()
    queue, seen = deque(u for u in seeds if not excluded(u)), set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        while queue and stats["pages"] < pages:
            url = queue.popleft()
            if url in seen or excluded(url) or action_like(url) or not _allowed(url, robots):
                continue
            seen.add(url)
            page = _observe(url)
            for item in items_from_page(page, rng):
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                stats["items"] += 1
            stats["pages"] += 1
            queue.extend(same_site_links(page))
            time.sleep(crawl_delay(url, robots))
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cdp-url", default=os.environ.get("LAYA_EXPLORE_CDP_URL"),
                        help="CDP URL of a throwaway Chrome (or LAYA_EXPLORE_CDP_URL); never your own browser")
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--pages", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("data/explore/items.jsonl"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        cdp_url = throwaway_cdp_url(args.cdp_url)
    except ValueError as exc:
        parser.error(str(exc))
    # Set before browser_harness is imported: it reads both at import time. Its own daemon name keeps it off
    # any daemon already attached to the user's Chrome.
    os.environ.update(BU_CDP_URL=cdp_url, BU_NAME="laya-explore")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("explored: %s", dict(crawl(args.seeds, args.pages, args.out, args.seed)))


if __name__ == "__main__":
    main()
