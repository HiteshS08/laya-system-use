"""Resolver first (exact, free), then the fast actor, then the planner picks among the actor's top five."""

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .actor import ActorPick, Predict, pick_target
from .planner import PlanStep, element_line
from .resolver import resolve

TOP_N = 5
DEFAULT_TAU = 0.5
Pick = Callable[[PlanStep, Sequence[tuple[str, str]], str], str | None]


@dataclass(frozen=True)
class Routed:
    index: str | None
    route: str
    confidence: float
    actor: ActorPick | None = None


def tau() -> float:
    return float(os.environ.get("ACTOR_TAU", DEFAULT_TAU))


def route(step: PlanStep, elements: Sequence[Mapping], recent: Sequence[str], goal: str, *,
          predict: Predict, pick: Pick) -> Routed:
    pool = [e for e in elements if step.operation in e["operations"]]
    if not pool:
        return Routed(None, "no_candidates", 0.0)
    element, how = resolve(step.target_text, pool)
    if element is not None and how is not None:
        return Routed(element["index"], how, 1.0)
    actor = pick_target(step, pool, recent, predict=predict)
    if actor.index is not None and actor.confidence >= tau():
        return Routed(actor.index, "actor", actor.confidence, actor)
    by_index = {e["index"]: e for e in pool}
    options = [(i, element_line(by_index[i])) for i, _ in actor.ranked[:TOP_N]]
    chosen = pick(step, options, goal) if options else None
    if chosen is None:
        return Routed(None, "planner_none", actor.confidence, actor)
    return Routed(chosen, "planner_pick", actor.confidence, actor)
