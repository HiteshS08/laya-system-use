"""The subgoal program a goal compiles into: a closed, line-based grammar the controller executes without an LLM."""

import re
from dataclasses import dataclass, replace

from .resolver import normalize

KINDS = ("FIND", "OPEN", "JUMP", "SCROLL", "FILL", "SELECT", "CLICK", "SUBMIT")
VALUE_KINDS = ("FILL", "SELECT")
MAX_SUBGOALS = 8
MAX_TARGET_CHARS = 80
MAX_VALUE_CHARS = 200
MAX_ORDINAL = 50
DONE_WHEN = "DONE_WHEN"

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```[A-Za-z]*")
_BULLET = re.compile(r"^(?:[-*]|\d+[.)])\s*")
_ORDINAL = re.compile(r"\s+@(\d+)$")


@dataclass(frozen=True)
class Subgoal:
    kind: str
    target: str
    value: str = ""
    ordinal: int = 0  # 1-based; 0 means no ordinal


@dataclass(frozen=True)
class Program:
    subgoals: tuple[Subgoal, ...]
    done_text: str = ""
    source: str = "compiler"  # compiler | cache | fallback


def _lines(text: str) -> list[str]:
    body = _FENCE.sub("\n", _THINK.sub("", text))
    lines = [_BULLET.sub("", line.strip()).strip() for line in body.splitlines()]
    return [line for line in lines if line]


def _checked(text: str, limit: int, what: str) -> str:
    if len(text) > limit:
        raise ValueError(f"Program {what} is longer than {limit} characters: {text[:40]!r}...")
    return text


def _subgoal(line: str) -> Subgoal:
    kind, _, rest = line.partition(" ")
    kind, rest = kind.upper(), rest.strip()
    if kind not in KINDS:
        raise ValueError(f"Unknown program step {kind!r}")
    value = ""
    if kind in VALUE_KINDS:
        rest, sep, value = rest.partition(" = ")
        rest, value = rest.strip(), value.strip()
        if not sep or not value:
            raise ValueError(f"{kind} needs 'field = value', got {line!r}")
    ordinal = 0
    match = _ORDINAL.search(rest)
    if match:
        ordinal, rest = int(match.group(1)), rest[: match.start()].strip()
        if not 1 <= ordinal <= MAX_ORDINAL:
            raise ValueError(f"Ordinal must be between 1 and {MAX_ORDINAL}, got {ordinal}")
    if not rest and kind != "SUBMIT":
        raise ValueError(f"{kind} needs a target")
    return Subgoal(kind, _checked(rest, MAX_TARGET_CHARS, "target"), _checked(value, MAX_VALUE_CHARS, "value"),
                   ordinal)


def parse_program(text: str) -> Program:
    lines = _lines(text)
    done_text = ""
    if lines and lines[-1].upper().startswith(DONE_WHEN):
        done_text = _checked(lines.pop()[len(DONE_WHEN):].strip(), MAX_TARGET_CHARS, "done text")
        if not done_text:
            raise ValueError("DONE_WHEN needs text")
    subgoals = tuple(_subgoal(line) for line in lines)
    if not subgoals:
        raise ValueError("Program has no steps")
    if len(subgoals) > MAX_SUBGOALS:
        raise ValueError(f"Program has more than {MAX_SUBGOALS} steps")
    return Program(subgoals, done_text)


def drop_redundant_opens(program: Program) -> Program:
    """Drop an OPEN/CLICK of the very name the preceding FIND reached: FIND already lands on that page."""
    kept: list[Subgoal] = []
    for sub in program.subgoals:
        previous = kept[-1] if kept else None
        if (sub.kind in ("OPEN", "CLICK") and not sub.ordinal and previous is not None and previous.kind == "FIND"
                and normalize(sub.target) == normalize(previous.target)):
            continue
        kept.append(sub)
    return replace(program, subgoals=tuple(kept))


def render_program(program: Program) -> str:
    lines = []
    for s in program.subgoals:
        line = f"{s.kind} {s.target}".rstrip()
        if s.ordinal:
            line += f" @{s.ordinal}"
        if s.kind in VALUE_KINDS:
            line += f" = {s.value}"
        lines.append(line)
    if program.done_text:
        lines.append(f"{DONE_WHEN} {program.done_text}")
    return "\n".join(lines)


def fallback_program(goal: str) -> Program:
    """Goal-mode Laya on the raw goal: the baseline behaviour, used when the goal cannot be compiled."""
    return Program((Subgoal("DO", goal),), source="fallback")
