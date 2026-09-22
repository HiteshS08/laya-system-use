"""Shortlister recall study on Mind2Web train shards: cache parsed steps once, then score ranking variants.

Sub-commands (all outputs go to the gitignored training/out/):
  cache   parse the shards once into a pickle of per-step data (goal, history, interactive pool with hints, gold)
  report  score every variant at K in {20, 30, 40, 60} (overall/CLICK x train/dev), reclaim result, JSON + table
"""

import argparse
import json
import logging
import math
import pickle
import random
import re
import unicodedata
from collections import Counter, deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, NamedTuple

from jev_ultrafast.candidates import Candidate
from jev_ultrafast.formatter import render_history_item
from jev_ultrafast.shortlister import _QUERY_STOP, _ranked_indices
from training import mind2web as m2w
from training.build_cases import is_dev_website

log = logging.getLogger("recall_study")

OUT_DIR = Path("training/out")
CACHE_PATH = OUT_DIR / "recall_cache.pkl"
REPORT_PATH = OUT_DIR / "recall_study.json"
KS = (20, 30, 40, 60)
DEV_MOD = 20
HINT_FIELDS = ("id", "name", "class", "title", "alt", "placeholder", "aria_label")
MAX_HINT = 300
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATORS = re.compile(r"[-_.:/]+")


class SCand(NamedTuple):
    """One pool element as cached: the neutral Candidate fields plus hint text and document position."""

    id: str
    label: str
    role: str
    value: str
    ops: tuple[str, ...]
    hints: str
    order: int

    def candidate(self) -> Candidate:
        return Candidate(self.id, self.label, self.role, self.value, frozenset(self.ops))


def split_identifier(text: str) -> str:
    """'mainNav-item_big' -> 'main Nav item big' so identifier pieces become matchable words."""
    return _SEPARATORS.sub(" ", _CAMEL.sub(" ", text))


def hint_text(attrs: dict[str, str]) -> str:
    parts = (split_identifier(attrs[f][:MAX_HINT]) for f in HINT_FIELDS if attrs.get(f))
    return " ".join(parts)


def _scand(cand: Candidate, raw_attrs: dict[str, str], order: dict[str, int]) -> SCand:
    return SCand(cand.id, cand.label, cand.role, cand.value, tuple(sorted(cand.ops)), hint_text(raw_attrs),
                 order.get(cand.id, len(order)))


def _raw_attrs(raw: dict) -> dict[str, str]:
    return {k: str(v) for k, v in json.loads(raw["attributes"]).items()}


def _nearest_interactive(node) -> tuple[Any, str, int] | None:
    """Nearest interactive descendant (breadth-first, document order on ties), else nearest interactive ancestor."""
    queue = deque((child, 1) for child in node if isinstance(child.tag, str))
    while queue:
        el, dist = queue.popleft()
        if m2w.is_interactive(el.tag, dict(el.attrib)):
            return el, "descendant", dist
        queue.extend((child, dist + 1) for child in el if isinstance(child.tag, str))
    for dist, el in enumerate(node.iterancestors(), start=1):
        if isinstance(el.tag, str) and m2w.is_interactive(el.tag, dict(el.attrib)):
            return el, "ancestor", dist
    return None


def _reclaim_record(node, gold_cand: Candidate, gold_tag: str, pool: dict[str, SCand],
                    index: dict, order: dict[str, int]) -> dict:
    """Where the non-interactive gold would map to; the mapped element is taken from the pool when it is in it."""
    record = {"orig_id": gold_cand.id, "orig_tag": gold_tag, "orig_label": gold_cand.label, "mapped": None,
              "kind": None, "distance": 0, "in_pool": False, "mapped_tag": None}
    found = _nearest_interactive(node) if node is not None else None
    if found is None:
        return record
    el, kind, dist = found
    bid = el.get("backend_node_id")
    if bid in pool:
        mapped = pool[bid]
    else:
        raw = {"tag": el.tag, "backend_node_id": bid, "attributes": json.dumps(dict(el.attrib))}
        cand, _ = m2w.to_candidate(raw, index)
        mapped = _scand(cand, {k: str(v) for k, v in el.attrib.items()}, order)
    return {**record, "mapped": tuple(mapped), "kind": kind, "distance": dist, "in_pool": bid in pool,
            "mapped_tag": el.tag}


def study_step(step: dict) -> dict:
    """parse_step's pool and gold as they were before Task 12, plus raw hints and the reclaim mapping."""
    op = m2w.OP_NAMES[step["operation"]["op"]]
    index = m2w.node_index(step["cleaned_html"])
    order = {bid: i for i, bid in enumerate(index)}
    parsed = [(raw, *m2w.to_candidate(raw, index)) for raw in [*step["pos_candidates"], *step["neg_candidates"]]]
    n_pos = len(step["pos_candidates"])
    pool = {c.id: _scand(c, _raw_attrs(raw), order) for raw, c, ok in parsed if ok}
    pos = parsed[:n_pos]
    gold = next((p for p in pos if p[2]), pos[0] if pos else None)
    record = {"pool": tuple(sorted(pool.values(), key=lambda s: s.order)), "op": op,
              "value": step["operation"].get("value") or "", "gold_id": gold[1].id if gold else None,
              "gold_label": gold[1].label if gold else "", "gold_interactive": bool(gold and gold[2]),
              "reclaim": None}
    if gold and not gold[2]:
        record["reclaim"] = _reclaim_record(index.get(gold[1].id), gold[1], gold[0]["tag"], pool, index, order)
    return record


def _check_against_parse_step(step: dict, record: dict) -> None:
    """The cached record is parse_step's result with the reclaim mapping held apart (parse_step applies it)."""
    parsed = m2w.parse_step(step)
    mapped = (record["reclaim"] or {}).get("mapped")
    pool_ids = {s.id for s in record["pool"]} | ({mapped[0]} if mapped else set())
    gold_id = mapped[0] if mapped else record["gold_id"]
    same_gold = (parsed.gold.id if parsed.gold else None) == gold_id
    if not (same_gold and {c.id for c in parsed.pool} == pool_ids
            and parsed.gold_interactive == (record["gold_interactive"] or bool(mapped))):
        raise RuntimeError(f"study_step disagrees with parse_step on action {step.get('action_uid')!r}")


def study_task(task: dict, dev: bool, verify: int = 0) -> Iterator[dict]:
    history: list[str] = []
    for i, step in enumerate(task["actions"]):
        record = study_step(step)
        if i < verify:
            _check_against_parse_step(step, record)
        gold_pool = next((s for s in record["pool"] if s.id == record["gold_id"]), None)
        valid = record["gold_id"] is not None and record["gold_interactive"] and gold_pool is not None \
            and record["op"] in gold_pool.ops
        yield {**record, "task_id": task["annotation_id"], "website": task["website"], "step": i, "dev": dev,
               "goal": task["confirmed_task"], "history": tuple(history), "valid_gold": valid}
        history.append(render_history_item(record["op"], record["gold_label"], record["value"]))


def build_cache(paths: Sequence[Path], out: Path, verify: int = 3) -> list[dict]:
    steps: list[dict] = []
    for path in paths:
        tasks = json.loads(Path(path).read_text())
        for task in tasks:
            steps.extend(study_task(task, is_dev_website(task["website"], DEV_MOD), verify))
        log.info("cached %s: %d steps so far", path.name, len(steps))
        del tasks
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:  # plain tuples: a pickled NamedTuple would be bound to the module it was written from
        pickle.dump([{**s, "pool": tuple(tuple(c) for c in s["pool"])} for s in steps], fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return steps


def load_cache(path: Path = CACHE_PATH) -> list[dict]:
    with path.open("rb") as fh:
        steps = pickle.load(fh)
    for step in steps:
        step["pool"] = tuple(SCand(*c) for c in step["pool"])
        if step["reclaim"] and step["reclaim"]["mapped"]:
            step["reclaim"]["mapped"] = SCand(*step["reclaim"]["mapped"])
    return steps


@dataclass(frozen=True)
class Variant:
    """One ranking recipe. All-defaults reproduces the production scorer's formula (V0)."""

    name: str
    hints: float = 0.0  # weight of matches on id/name/class/... words; 0 = ignore them (serve-time legal)
    idf: str = "inv"  # "inv" = 1/df (production) or "bm25" = ln(1 + (N - df + .5) / (df + .5))
    norm: str = "none"  # label-length normalisation: "none", "bm25" (uses b and k1) or "ratio" (score * avg_len / len)
    b: float = 0.75
    k1: float = 1.2
    bigram: float = 0.0  # weight of matching consecutive word pairs
    both_forms: bool = False  # a word matches on its raw form and again on its stem
    stemmer: str = "s"  # "s" = trailing-s only (production), "light" = also -ing/-ed/-es/-ly
    role_w: float = 0.0  # weight of the additive role prior
    pos_w: float = 0.0  # weight of the additive page-position prior


# Form fields > other controls > links, read off the per-role gold rates in role_lifts().
ROLE_PRIOR = {"textbox": 1.0, "searchbox": 1.0, "combobox": 1.0, "button": 0.5, "option": 0.5, "tab": 0.5,
              "checkbox": 0.5, "spinbutton": 0.5, "switch": 0.5}
_WORD = re.compile(r"[a-z0-9]{2,}")


@lru_cache(maxsize=None)
def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()))


def _stem(word: str, kind: str) -> str:
    if kind == "s":
        return word[:-1] if len(word) > 3 and word.endswith("s") else word
    for suffix in ("ing", "ed", "es", "ly", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


@lru_cache(maxsize=None)
def _terms(text: str, both: bool, kind: str) -> frozenset[str]:
    out: set[str] = set()
    for word in _tokens(text):
        out.add(_stem(word, kind))
        if both:
            out.add(word)
    return frozenset(out)


def _bigrams(text: str) -> frozenset[tuple[str, str]]:
    toks = _tokens(text)
    return frozenset(zip(toks, toks[1:], strict=False))


def _idf(df: int, n: int, kind: str) -> float:
    return 1.0 / df if kind == "inv" else math.log(1.0 + (n - df + 0.5) / (df + 0.5))


def _query(goal: str, history: Sequence[str], v: Variant) -> tuple[frozenset[str], frozenset[tuple[str, str]]]:
    words = _terms(" ".join([goal, *history]), v.both_forms, v.stemmer) - _QUERY_STOP
    pairs: set[tuple[str, str]] = set()
    for text in [goal, *history]:
        pairs |= _bigrams(text)
    return words, frozenset(p for p in pairs if not (set(p) & set(_QUERY_STOP)))


def _doc_terms(c: SCand, v: Variant) -> dict[str, float]:
    doc = {t: 1.0 for t in _terms(f"{c.label} {c.value}", v.both_forms, v.stemmer)}
    if v.hints:
        for t in _terms(c.hints, v.both_forms, v.stemmer):
            doc.setdefault(t, v.hints)
    return doc


def _lexical_scores(query: frozenset[str], docs: list[dict[str, float]], lens: list[int], v: Variant) -> list[float]:
    n = len(docs)
    df: Counter[str] = Counter(t for d in docs for t in d)
    avg = (sum(lens) / n) or 1.0
    scores = []
    for d, length in zip(docs, lens, strict=True):
        total = sum(w * _idf(df[t], n, v.idf) for t, w in d.items() if t in query)
        if v.norm == "bm25":
            total *= (v.k1 + 1.0) / (1.0 + v.k1 * (1.0 - v.b + v.b * length / avg))
        elif v.norm == "ratio":
            total *= avg / max(length, 1)
        scores.append(total)
    return scores


def _bigram_scores(pairs: frozenset[tuple[str, str]], cands: Sequence[SCand], v: Variant) -> list[float]:
    grams = [_bigrams(c.label) for c in cands]
    df: Counter[tuple[str, str]] = Counter(p for g in grams for p in g)
    return [v.bigram * sum(_idf(df[p], len(cands), v.idf) for p in g & pairs) for g in grams]


def rank_variant(v: Variant, goal: str, history: Sequence[str], cands: Sequence[SCand]) -> list[int]:
    query, pairs = _query(goal, history, v)
    docs = [_doc_terms(c, v) for c in cands]
    scores = _lexical_scores(query, docs, [len(_tokens(f"{c.label} {c.value}")) for c in cands], v)
    if v.bigram:
        scores = [s + b for s, b in zip(scores, _bigram_scores(pairs, cands, v), strict=True)]
    n = len(cands)
    for i, c in enumerate(cands):
        scores[i] += v.role_w * ROLE_PRIOR.get(c.role, 0.0) + v.pos_w * position_prior(i, n)
    return sorted(range(n), key=lambda i: (-scores[i], i))


def position_prior(i: int, n: int) -> float:
    """Favour elements earlier on the page (1.0 first, 0.0 last)."""
    return 1.0 - i / max(n - 1, 1)


def rank_production(goal: str, history: Sequence[str], cands: Sequence[SCand]) -> list[int]:
    """Whatever jev_ultrafast.shortlister ranks like today (V0 before Task 12, V3 after adoption)."""
    return _ranked_indices(goal, history, [c.candidate() for c in cands])


def evaluate(steps: Sequence[dict], ranker: Callable[[str, Sequence[str], Sequence[SCand]], list[int]]) -> list[tuple]:
    """(dev, gold_op, pool_size, gold_rank) for every valid-gold step; rank is 0-based within the gold op's pool."""
    out = []
    for s in steps:
        if not s["valid_gold"]:
            continue
        cands = [c for c in s["pool"] if s["op"] in c.ops]
        gold = next(i for i, c in enumerate(cands) if c.id == s["gold_id"])
        ranked = ranker(s["goal"], s["history"], cands)
        out.append((s["dev"], s["op"], len(cands), ranked.index(gold)))
    return out


def recall(results: Sequence[tuple], k: int, *, dev: bool, click_only: bool = False) -> float:
    chosen = [r for r in results if r[0] == dev and (r[1] == "CLICK" or not click_only)]
    return sum(n <= k or rank < k for _, _, n, rank in chosen) / len(chosen) if chosen else float("nan")


def _reclaim_pool(step: dict) -> tuple[list[SCand], int] | None:
    """Gold-op pool with the mapped element added (in document order) and the mapped element's index, or None."""
    rec = step["reclaim"]
    if rec is None or rec["mapped"] is None or step["op"] not in rec["mapped"].ops:
        return None
    mapped: SCand = rec["mapped"]
    pool = [c for c in step["pool"] if step["op"] in c.ops]
    if not rec["in_pool"]:
        pool = sorted([*pool, mapped], key=lambda c: c.order)
    return pool, next(i for i, c in enumerate(pool) if c.id == mapped.id)


def reclaim_ranks(steps: Sequence[dict], ranker) -> list[tuple]:
    """(dev, op, pool_size, gold_rank) for every step whose non-interactive gold maps to a usable element."""
    out = []
    for s in steps:
        got = _reclaim_pool(s) if s["gold_id"] and not s["gold_interactive"] else None
        if got:
            pool, gold = got
            out.append((s["dev"], s["op"], len(pool), ranker(s["goal"], s["history"], pool).index(gold)))
    return out


def _label(text: str, limit: int = 60) -> str:
    return " ".join(text.split())[:limit]


def _example(step: dict) -> dict:
    r = step["reclaim"]
    return {"goal": _label(step["goal"], 90), "op": step["op"],
            "original": f"<{r['orig_tag']}> {_label(r['orig_label'])!r}",
            "chosen": f"<{r['mapped_tag']}> {_label(r['mapped'].label)!r}",
            "how": f"{r['kind']} at distance {r['distance']}"}


def reclaim_examples(steps: Sequence[dict], n: int = 10, seed: int = 0) -> list[dict]:
    usable = [s for s in steps if s["gold_id"] and not s["gold_interactive"] and _reclaim_pool(s)]
    return [_example(s) for s in random.Random(seed).sample(usable, min(n, len(usable)))]


def reclaim_summary(steps: Sequence[dict], ranker) -> dict:
    touched = [s for s in steps if s["gold_id"] and not s["gold_interactive"]]
    mapped = [s for s in touched if s["reclaim"]["mapped"]]
    usable = [s for s in mapped if s["op"] in s["reclaim"]["mapped"].ops]
    ranks = reclaim_ranks(steps, ranker)
    return {
        "steps": len(steps), "gold_not_interactive": len(touched), "mapped": len(mapped), "op_compatible": len(usable),
        "kind": dict(Counter(s["reclaim"]["kind"] for s in usable)),
        "distance": dict(sorted(Counter(min(s["reclaim"]["distance"], 5) for s in usable).items())),
        "mapped_already_in_pool": sum(s["reclaim"]["in_pool"] for s in usable),
        "unmapped_original_tags": dict(Counter(
            s["reclaim"]["orig_tag"] for s in touched if not s["reclaim"]["mapped"]).most_common(8)),
        "recall": {k: {"train": recall(ranks, k, dev=False), "dev": recall(ranks, k, dev=True)} for k in KS},
    }


def role_lifts(steps: Sequence[dict]) -> dict[str, dict]:
    """Per role: share of pool candidates vs share of gold elements (train only) for the gold op's pool."""
    pool_n: Counter[str] = Counter()
    gold_n: Counter[str] = Counter()
    for s in steps:
        if s["dev"] or not s["valid_gold"]:
            continue
        for c in s["pool"]:
            if s["op"] in c.ops:
                pool_n[c.role] += 1
                gold_n[c.role] += c.id == s["gold_id"]
    return {r: {"pool": pool_n[r], "gold": gold_n[r], "gold_rate": round(gold_n[r] / pool_n[r], 4)}
            for r, _ in pool_n.most_common()}


def variant_recalls(steps: Sequence[dict], variants: Sequence[Variant | str]) -> dict[str, dict]:
    """name -> K -> {overall|click -> {train|dev -> recall}}; a plain string names the production ranker."""
    table: dict[str, dict] = {}
    for v in variants:
        name, ranker = (v.name, partial(rank_variant, v)) if isinstance(v, Variant) else (v, rank_production)
        results = evaluate(steps, ranker)
        table[name] = {k: {
            scope: {"train": recall(results, k, dev=False, click_only=click),
                    "dev": recall(results, k, dev=True, click_only=click)}
            for scope, click in (("overall", False), ("click", True))} for k in KS}
    return table


def format_table(table: dict[str, dict]) -> str:
    lines = [f"{'variant':<22}{'K':>3}  {'overall train':>13} {'overall dev':>11} {'CLICK train':>11} {'CLICK dev':>9}"]
    for name, by_k in table.items():
        for k, r in by_k.items():
            lines.append(f"{name:<22}{k:>3}  {r['overall']['train']:>13.3f} {r['overall']['dev']:>11.3f} "
                         f"{r['click']['train']:>11.3f} {r['click']['dev']:>9.3f}")
    return "\n".join(lines)


V3 = Variant("V3 ratio+role+pos", norm="ratio", role_w=0.4, pos_w=0.3)
VARIANTS = (
    Variant("V0"),
    Variant("V1 hints", hints=1.0),
    Variant("V1 hints x0.5", hints=0.5),
    Variant("V2 bm25 idf", idf="bm25"),
    Variant("V2 bm25 len-norm", norm="bm25", b=1.0),
    Variant("V2 ratio len-norm", norm="ratio"),
    Variant("V2 bigram", bigram=1.0),
    Variant("V2 raw+stem", both_forms=True),
    Variant("V2 light stem", stemmer="light"),
    Variant("V2 role prior", role_w=0.4),
    Variant("V2 position prior", pos_w=0.3),
    V3,
    Variant("V3 + bigram", norm="ratio", role_w=0.4, pos_w=0.3, bigram=1.0),
    Variant("V3 + raw+stem", norm="ratio", role_w=0.4, pos_w=0.3, both_forms=True),
    Variant("V3 + hints", norm="ratio", role_w=0.4, pos_w=0.3, hints=1.0),
    "production now",
)


def report(steps: Sequence[dict], out: Path = REPORT_PATH) -> dict:
    table = variant_recalls(steps, VARIANTS)
    rec = {"V0": reclaim_summary(steps, partial(rank_variant, VARIANTS[0])),
           "V3": reclaim_summary(steps, partial(rank_variant, V3))}
    result = {"variants": {n: {str(k): v for k, v in by_k.items()} for n, by_k in table.items()},
              "reclaim": rec, "reclaim_examples": reclaim_examples(steps), "role_lifts": role_lifts(steps)}
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(format_table(table))  # CLI report, so print rather than log
    print(json.dumps({k: result[k] for k in ("reclaim", "reclaim_examples")}, indent=2, ensure_ascii=False))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--input", nargs="+", type=Path, required=True)
    cache.add_argument("--out", type=Path, default=CACHE_PATH)
    sub.add_parser("report")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.cmd == "cache":
        build_cache(sorted(args.input), args.out)
    else:
        report(load_cache())


if __name__ == "__main__":
    main()
