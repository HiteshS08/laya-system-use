"""One prompt format for training and serving. Labels come first so Laya's per-option token budget keeps them."""

from collections.abc import Mapping, Sequence

from .candidates import OPERATIONS, Candidate

MAX_LABEL_CHARS = 70
MAX_VALUE_CHARS = 20
RECENT_ACTIONS = 3

OPERATION_HELP = {
    "CLICK": "click a link, button, option or other control",
    "TYPE_TEXT": "type text into a field",
    "SELECT": "choose a value in a dropdown",
}
# The question id is invisible to Laya, so the instruction must say which operation the options are for.
TARGET_INSTRUCTIONS = {
    "CLICK": "Which element should be clicked next to advance the goal?",
    "TYPE_TEXT": "Which field should be typed into next to advance the goal?",
    "SELECT": "Which dropdown should be set next to advance the goal?",
}


def _clean(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


def render_option(candidate: Candidate) -> str:
    label = _clean(candidate.label, MAX_LABEL_CHARS) or candidate.role or "unnamed"
    extras = [candidate.role] if candidate.role else []
    value = _clean(candidate.value, MAX_VALUE_CHARS)
    if value:
        extras.append(f"={value}")
    return f"{label} ({', '.join(extras)})" if extras else label


def render_history_item(op: str, label: str, value: str = "") -> str:
    suffix = f" = {_clean(value, MAX_LABEL_CHARS)}" if value else ""
    return f"{op} {_clean(label, MAX_LABEL_CHARS)}{suffix}"


def build_request(
    goal: str, history: Sequence[str], candidates_by_op: Mapping[str, Sequence[Candidate]]
) -> tuple[dict, dict]:
    state = {"goal": goal, "recent_actions": list(history)[-RECENT_ACTIONS:]}
    ops = [op for op in OPERATIONS if candidates_by_op.get(op)]
    questions: dict[str, dict] = {}
    if len(ops) > 1:
        questions["operation"] = {
            "type": "choice",
            "instructions": "Which operation advances the goal next?",
            "criteria": {op: OPERATION_HELP[op] for op in ops},
        }
    for op in ops:
        if len(candidates_by_op[op]) > 1:
            questions[f"{op.lower()}_target"] = {
                "type": "choice",
                "instructions": TARGET_INSTRUCTIONS[op],
                "criteria": {c.id: render_option(c) for c in candidates_by_op[op]},
            }
    return state, questions


def history_strings(history: Sequence[Mapping]) -> list[str]:
    """Jev history entries -> the same strings training used (see render_history_item)."""
    return [
        render_history_item(h.get("operation") or h["kind"].upper(), h["action"], h.get("text") or "")
        for h in history
    ]
