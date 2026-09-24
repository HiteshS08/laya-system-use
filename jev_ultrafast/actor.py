"""The fast actor answers one question: which element does this step mean. The step instruction is its goal."""

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .candidates import Candidate
from .formatter import RECENT_ACTIONS, TARGET_INSTRUCTIONS, render_option
from .planner import PlanStep
from .policy import _valid, candidates_from
from .shortlister import shortlist

Predict = Callable[[dict, dict], dict]


@dataclass(frozen=True)
class ActorPick:
    index: str | None
    confidence: float
    ranked: tuple[tuple[str, float], ...]
    request: dict
    latency_ms: int
    model: str


def _context_enabled() -> bool:
    return os.environ.get("ACTOR_CONTEXT") == "1"


def actor_request(step: PlanStep, candidates: Sequence[Candidate], recent: Sequence[str], *,
                  context: bool = False) -> tuple[dict, dict]:
    state = {"goal": step.instruction, "recent_actions": list(recent)[-RECENT_ACTIONS:]}
    question = {"type": "choice", "instructions": TARGET_INSTRUCTIONS[step.operation],
                "criteria": {c.id: render_option(c, with_context=context) for c in candidates}}
    return state, {f"{step.operation.lower()}_target": question}


def pick_target(step: PlanStep, elements: Sequence[Mapping], recent: Sequence[str], *, predict: Predict) -> ActorPick:
    started = time.perf_counter()
    pool = candidates_from(elements)
    if not pool:
        return ActorPick(None, 0.0, (), {}, 0, "none")
    chosen = shortlist(f"{step.instruction} {step.target_text}", [], pool)
    if len(chosen) == 1:
        # A lone field is almost always the one meant; a lone link may be unrelated, so the planner confirms it.
        trust = 0.0 if step.operation == "CLICK" else 1.0
        return ActorPick(chosen[0].id, trust, ((chosen[0].id, trust),), {}, 0, "sole")
    state, questions = actor_request(step, chosen, recent, context=_context_enabled())
    name = next(iter(questions))
    info = predict(state, questions)
    answer = _valid(info.get("answers", {}).get(name, {}), [c.id for c in chosen], name)
    probabilities = answer["probabilities"]
    ranked = tuple(sorted(probabilities.items(), key=lambda kv: -kv[1]))
    return ActorPick(answer["choice"], probabilities[answer["choice"]], ranked,
                     {"state": state, "questions": questions},
                     round((time.perf_counter() - started) * 1000), info.get("model", "laya"))
