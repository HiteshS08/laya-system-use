"""One run's planner-driven control: step queue, memory, and decisions the Agent loop can execute."""

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict

from . import policy
from .formatter import history_strings
from .memory import StepMemory
from .model import action_space
from .planner import Plan, PlanStep, pick, plan
from .pruning import prune_actions
from .resolver import normalize, resolve
from .router import Routed, route
from .textmodel import choose_option

log = logging.getLogger("pilot")
STALL_ACTIONS = 4
MAX_PLANS_PER_DECISION = 2
TOOL_OPERATIONS = ("SCROLL_TO_TEXT", "GOTO")


class Pilot:
    def __init__(self, goal: str, *, plan_fn: Callable = plan, pick_fn: Callable = pick,
                 predict: Callable = policy.laya_predict, fallback: Callable = policy.decide) -> None:
        self.goal = goal
        self.memory = StepMemory()
        self.plans: list[dict] = []
        self._plan, self._pick, self._predict, self._fallback = plan_fn, pick_fn, predict, fallback
        self._queue: tuple[PlanStep, ...] = ()
        self._plan_url: str | None = None
        self._planned_at = 0  # attempts known when the current queue was planned
        self._completed: list[str] = []
        self._completed_at: list[int] = []  # len(completed) after each executed action
        self._pending: tuple[int, PlanStep] | None = None  # (history index, step) of the last decision
        self._unroutable: dict[str, list[str]] = {}
        self._planner_failed: set[str] = set()  # urls where the planner raised; skip straight to fallback there

    def decide(self, page: Mapping, history: Sequence[Mapping]) -> dict:
        started = time.perf_counter()
        self._requeue_unexecuted(len(history))
        self._absorb(history)
        if self._stalled():
            return self._stop("BLOCKED", "", started)
        elements, targets, _ = action_space(prune_actions(page["actions"], self.goal))
        for _ in range(MAX_PLANS_PER_DECISION):
            if self._needs_plan(page):
                if page["url"] in self._planner_failed:
                    break
                outcome = self._replan(page, elements)
                if outcome is None:
                    break
                if outcome.status != "continue":
                    return self._stop(outcome.status.upper(), outcome.evidence, started)
            step, self._queue = self._queue[0], self._queue[1:]
            decision = self._step_decision(step, page, elements, targets, history, started)
            if decision is not None:
                self._pending = (len(history), step)
                return decision
            self._note_unroutable(page["url"], step)
            self._queue = ()
        return self._fall_back(page, history)

    def _requeue_unexecuted(self, history_len: int) -> None:
        # The previous decision was never executed (no history entry appended for it): put it back.
        if self._pending is not None and self._pending[0] == history_len:
            _, step = self._pending
            self._queue = (step, *self._queue)
            self._pending = None

    def _note_unroutable(self, url: str, step: PlanStep) -> None:
        note = f"{step.operation} {step.target_text} (no matching element)"
        notes = self._unroutable.setdefault(url, [])
        if note not in notes:
            notes.append(note)

    def _absorb(self, history: Sequence[Mapping]) -> None:
        known = len(self.memory.attempts)
        self.memory.sync(history)
        if self._pending and self._pending[0] < len(history):
            index, step = self._pending
            if self.memory.attempts[index].outcome != "no_change" or \
                    self.memory.attempts[index].operation == "TYPE_TEXT":
                self._completed.append(step.instruction)
            self._pending = None
        self._completed_at.extend([len(self._completed)] * (len(self.memory.attempts) - known))

    def _stalled(self) -> bool:
        if self.memory.streak_without_url_change() < STALL_ACTIONS or len(self._completed_at) < STALL_ACTIONS:
            return False
        before = self._completed_at[-STALL_ACTIONS - 1] if len(self._completed_at) > STALL_ACTIONS else 0
        return self._completed_at[-1] == before

    def _needs_plan(self, page: Mapping) -> bool:
        new_failure = self.memory.last_failed() and len(self.memory.attempts) > self._planned_at
        return not self._queue or page["url"] != self._plan_url or new_failure

    def _replan(self, page: Mapping, elements: Sequence[Mapping]) -> Plan | None:
        failed = [*self.memory.failed(page["url"]), *self._unroutable.get(page["url"], [])]
        try:
            result = self._plan(self.goal, page, elements, list(self._completed), failed)
        except (ValueError, RuntimeError) as exc:
            log.warning("planner failed on %s: %s", page["url"], exc)
            self.plans.append({"url": page["url"], "error": str(exc)})
            self._planner_failed.add(page["url"])
            return None
        self.plans.append({"url": page["url"], "status": result.status, "evidence": result.evidence,
                           "steps": [asdict(s) for s in result.steps], "latency_ms": result.latency_ms,
                           "request_chars": result.request_chars})
        self._queue, self._plan_url, self._planned_at = result.steps, page["url"], len(self.memory.attempts)
        return result

    def _usable(self, page: Mapping, elements: Sequence[Mapping], operation: str) -> list[Mapping]:
        excluded = self.memory.excluded(page["url"])
        return [e for e in elements if (operation, normalize(e["label"])) not in excluded]

    def _step_decision(self, step: PlanStep, page: Mapping, elements: Sequence[Mapping], targets: Mapping,
                       history: Sequence[Mapping], started: float) -> dict | None:
        if step.operation in TOOL_OPERATIONS:
            tool = {"operation": step.operation, "arg": step.target_text}
            return self._decision("TOOL", step, Routed(None, "planner", 1.0), started, tool=tool)
        usable = self._usable(page, elements, step.operation)
        recent = history_strings(history)
        routed = route(step, usable, recent, self.goal, predict=self._predict, pick=self._pick)
        if routed.index is None:
            return None
        key = routed.index
        if step.operation == "SELECT":
            key = self._select_key(next(e for e in elements if e["index"] == routed.index), step)
        return self._decision(targets[step.operation][key]["id"], step, routed, started)

    def _select_key(self, element: Mapping, step: PlanStep) -> str:
        options = element["options"]
        labels = [o["label"].split(" → ", 1)[-1] for o in options]
        match, _ = resolve(step.value, [{"index": str(i), "label": label} for i, label in enumerate(labels)])
        position = int(match["index"]) if match else labels.index(choose_option(self.goal, element["label"], labels))
        return options[position]["index"]

    def _decision(self, choice: str, step: PlanStep, routed: Routed, started: float, *,
                  tool: dict | None = None) -> dict:
        actor = routed.actor
        return {
            "choice": choice, "operation": step.operation, "target": routed.index,
            "confidence": routed.confidence, "probabilities": {choice: routed.confidence},
            "operation_probabilities": {step.operation: 1.0},
            "target_probabilities": dict(actor.ranked) if actor else {}, "target_confidence": routed.confidence,
            "raw_answers": {}, "model": actor.model if actor else routed.route, "usage": {},
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "request": actor.request if actor else {}, "actor_ms": actor.latency_ms if actor else 0,
            "route": routed.route, "instruction": step.instruction,
            "value": step.value or None, "tool": tool, "evidence": "",
        }

    def _stop(self, kind: str, evidence: str, started: float) -> dict:
        step = PlanStep(kind, "", "", "")
        return {**self._decision(kind, step, Routed(None, "planner", 1.0), started), "evidence": evidence}

    def _fall_back(self, page: Mapping, history: Sequence[Mapping]) -> dict:
        decision = self._fallback(page, self.goal, history)
        self._pending = None
        return {**decision, "route": "planner_fallback", "instruction": None, "value": None, "tool": None,
                "evidence": ""}
