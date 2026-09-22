"""Parse Mind2Web steps into neutral Candidates and Laya training cases."""

import json
import statistics
from collections import Counter, deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from jev_ultrafast.candidates import OPERATIONS, Candidate
from jev_ultrafast.formatter import build_request, render_history_item
from jev_ultrafast.shortlister import DEFAULT_K, rank_candidates, shortlist

INTERACTIVE_TAGS = frozenset({"a", "button", "input", "textarea", "select", "summary"})
# Mirrors the roles jev_ultrafast/snapshot.js recognises, so training candidates look like live ones.
ROLES = frozenset({
    "button", "link", "checkbox", "radio", "switch", "tab", "menuitem", "menuitemradio", "option", "gridcell",
    "combobox", "textbox", "searchbox", "spinbutton",
})
EDITABLE_ROLES = frozenset({"textbox", "searchbox", "spinbutton", "combobox"})
OP_NAMES = {"CLICK": "CLICK", "TYPE": "TYPE_TEXT", "SELECT": "SELECT"}
MAX_LABEL = 200
# The default parser silently stops nesting at 255 levels, which would drop labels on deep pages.
_PARSER = etree.HTMLParser(huge_tree=True)


def role_for(tag: str, attrs: dict[str, str]) -> str:
    explicit = attrs.get("role")
    if explicit in ROLES:
        return explicit
    if tag in {"button", "summary"}:
        return "button"
    if tag == "a":
        return "link"
    if tag == "select":
        return "combobox"
    if tag == "textarea":
        return "textbox"
    if tag == "input":
        kind = attrs.get("type", "text")
        if kind in {"checkbox", "radio"}:
            return kind
        if kind in {"button", "submit", "reset", "image"}:
            return "button"
        if kind == "search":
            return "searchbox"
        return "spinbutton" if kind == "number" else "textbox"
    return tag


def is_interactive(tag: str, attrs: dict[str, str]) -> bool:
    return tag in INTERACTIVE_TAGS or attrs.get("role") in ROLES or attrs.get("contenteditable") == "true"


def ops_for(tag: str, role: str) -> frozenset[str]:
    if tag == "select":
        return frozenset({"SELECT"})
    if tag == "textarea" or role in EDITABLE_ROLES:
        return frozenset({"TYPE_TEXT", "CLICK"})
    return frozenset({"CLICK"})


def label_for(node, attrs: dict[str, str]) -> str:
    text = " ".join("".join(node.itertext()).split()) if node is not None else ""
    for value in (attrs.get("aria_label"), text, attrs.get("alt"), attrs.get("title"), attrs.get("placeholder"),
                  attrs.get("value"), attrs.get("name")):
        if value:
            return value[:MAX_LABEL]
    return ""


def node_index(cleaned_html: str) -> dict:
    """backend_node_id -> element, in document order (dict order is the page order)."""
    root = etree.HTML(cleaned_html, parser=_PARSER)
    if root is None:
        return {}
    return {n.get("backend_node_id"): n for n in root.iter(tag=etree.Element) if n.get("backend_node_id")}


def to_candidate(raw: dict, index: dict) -> tuple[Candidate, bool]:
    attrs = {k: str(v) for k, v in json.loads(raw["attributes"]).items()}
    tag, bid = raw["tag"], raw["backend_node_id"]
    role = role_for(tag, attrs)
    ops = ops_for(tag, role)
    value = attrs.get("input_value", "")[:MAX_LABEL] if "TYPE_TEXT" in ops else ""
    # The live snapshot names an unnamed element by its role (`name(e) || rname`), so training must as well.
    label = label_for(index.get(bid), attrs) or role
    return Candidate(bid, label, role, value, ops), is_interactive(tag, attrs)


def _is_element(node) -> bool:
    return isinstance(node.tag, str)  # skips comments and processing instructions


def _is_interactive_element(node) -> bool:
    return _is_element(node) and is_interactive(node.tag, dict(node.attrib))


def nearest_interactive(node):
    """Nearest interactive descendant (breadth-first, page order on ties), else the nearest interactive ancestor."""
    queue = deque(child for child in node if _is_element(child))
    while queue:
        el = queue.popleft()
        if _is_interactive_element(el):
            return el
        queue.extend(child for child in el if _is_element(child))
    return next((el for el in node.iterancestors() if _is_interactive_element(el)), None)


def reclaim_gold(gold_id: str, index: dict, pool: dict[str, Candidate]) -> Candidate | None:
    """The interactive element a click on a non-interactive gold lands on (an icon in its button, a label's input)."""
    node = index.get(gold_id)
    target = nearest_interactive(node) if node is not None else None
    bid = target.get("backend_node_id") if target is not None else None
    if bid is None:
        return None
    if bid in pool:
        return pool[bid]
    raw = {"tag": target.tag, "backend_node_id": bid, "attributes": json.dumps(dict(target.attrib))}
    return to_candidate(raw, index)[0]


@dataclass(frozen=True)
class ParsedStep:
    pool: tuple[Candidate, ...]
    gold: Candidate | None
    gold_interactive: bool
    op: str
    value: str


def parse_step(step: dict) -> ParsedStep:
    raw_op = step["operation"]["op"]
    if raw_op not in OP_NAMES:
        raise ValueError(f"Unknown Mind2Web operation {raw_op!r} in action {step.get('action_uid')!r}")
    index = node_index(step["cleaned_html"])
    order = {bid: i for i, bid in enumerate(index)}
    pos = [to_candidate(raw, index) for raw in step["pos_candidates"]]
    neg = [to_candidate(raw, index) for raw in step["neg_candidates"]]
    pool = {c.id: c for c, ok in [*pos, *neg] if ok}
    gold = next(((c, ok) for c, ok in pos if ok), pos[0] if pos else None)
    if gold and not gold[1]:
        mapped = reclaim_gold(gold[0].id, index, pool)
        if mapped is not None:
            pool.setdefault(mapped.id, mapped)
            gold = (mapped, True)
    return ParsedStep(
        pool=tuple(sorted(pool.values(), key=lambda c: order.get(c.id, len(order)))),
        gold=gold[0] if gold else None,
        gold_interactive=bool(gold and gold[1]),
        op=OP_NAMES[raw_op],
        value=step["operation"].get("value") or "",
    )


def _drop_reason(parsed: ParsedStep) -> str | None:
    if parsed.gold is None:
        return "no_gold"
    if not parsed.gold_interactive:
        return "gold_not_interactive"
    if parsed.op not in parsed.gold.ops:
        return "op_not_allowed"
    return None


def _gold(op: str, gold_id: str, questions: dict) -> dict:
    gold = {}
    if "operation" in questions:
        gold["operation"] = {"probabilities": {o: float(o == op) for o in questions["operation"]["criteria"]}}
    target = f"{op.lower()}_target"
    if target in questions:
        gold[target] = {"probabilities": {i: float(i == gold_id) for i in questions[target]["criteria"]}}
    return gold


def _row(task: dict, index: int, parsed: ParsedStep, history: Sequence[str], k: int) -> dict:
    goal = task["confirmed_task"]
    reason = _drop_reason(parsed)
    valid = reason is None
    per_op = {op: [c for c in parsed.pool if op in c.ops] for op in OPERATIONS}
    by_op = {op: shortlist(goal, history, cands, k) for op, cands in per_op.items()}
    state, questions = build_request(goal, history, by_op)
    gold_id = parsed.gold.id if valid else None
    in_shortlist = valid and gold_id in {c.id for c in by_op[parsed.op]}
    if valid and not in_shortlist:
        reason = "gold_not_shortlisted"
    gold = _gold(parsed.op, gold_id, questions) if reason is None else {}
    if reason is None and not gold:
        reason = "trivial"
    return {
        "task_id": task["annotation_id"], "website": task["website"], "domain": task["domain"], "goal": goal,
        "step": index, "state": state, "questions": questions, "gold": gold, "gold_op": parsed.op,
        "gold_id": gold_id, "valid_gold": valid, "gold_in_shortlist": in_shortlist, "drop_reason": reason,
        "ops_available": [op for op in OPERATIONS if per_op[op]],
        "sole": {op: cands[0].id for op, cands in by_op.items() if len(cands) == 1},
        "top1_by_op": {op: rank_candidates(goal, history, cands)[0].id for op, cands in per_op.items() if cands},
        "pool_size": len(parsed.pool),
    }


def task_rows(task: dict, k: int = DEFAULT_K) -> Iterator[dict]:
    history: list[str] = []
    for i, step in enumerate(task["actions"]):
        parsed = parse_step(step)
        yield _row(task, i, parsed, history, k)
        history.append(render_history_item(parsed.op, parsed.gold.label if parsed.gold else "", parsed.value))


def iter_tasks(paths: Iterable[Path]) -> Iterator[dict]:
    for path in paths:
        tasks = json.loads(Path(path).read_text())
        yield from tasks
        del tasks


def _mean(values: Iterable[bool]) -> float | None:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else None


def summarize(rows: Sequence[dict]) -> dict:
    valid = [r for r in rows if r["valid_gold"]]
    return {
        "steps": len(rows),
        "valid_gold": len(valid),
        "usable_for_training": sum(r["drop_reason"] is None for r in rows),
        "drop_reasons": dict(Counter(r["drop_reason"] for r in rows if r["drop_reason"])),
        "recall_at_k": _mean(r["gold_in_shortlist"] for r in valid),
        "top1_given_op": _mean(r["top1_by_op"].get(r["gold_op"]) == r["gold_id"] for r in valid),
        "median_pool_size": statistics.median(r["pool_size"] for r in rows) if rows else None,
        "gold_ops": dict(Counter(r["gold_op"] for r in rows)),
    }
