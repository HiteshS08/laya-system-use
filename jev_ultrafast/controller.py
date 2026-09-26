"""Program backend: execute a compiled subgoal program with deterministic tactics and the Laya actor.

Per action there is no LLM call: completion is checked on the page, tactics pick the step, and grounding goes
preset element -> ordinal group -> exact/fuzzy label -> Laya. The only LLM call is the compile, once per task.
"""

import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from urllib.parse import urldefrag, urlsplit

from . import policy
from .actor import ActorPick, pick_target
from .checks import fragment_names, heading_in_view, satisfied, text_shown, title_subject
from .compiler import compile_goal, default_cache
from .formatter import history_strings
from .groups import ordinal_pick
from .memory import EDITABLE_ROLES
from .model import action_space
from .program import Program, Subgoal, render_program
from .pruning import prune_actions
from .resolver import normalize, resolve
from .search import SearchTemplates, learn_template
from .tactics import Progress, Step, matching_option, next_step

log = logging.getLogger("controller")
MAX_ACTIONS_PER_SUBGOAL = 4
MAX_MISSES_PER_SUBGOAL = 2
TOOL_OPERATIONS = ("SCROLL_TO_TEXT", "GOTO", "SUBMIT", "FRAGMENT")
PAGE_KINDS = ("FIND", "JUMP", "SCROLL")  # kinds whose completion the page itself shows


@dataclass(frozen=True)
class _Pending:
    history_len: int
    subgoal: int
    step: Step
    index: str | None
    label: str


def _missed(entry: Mapping) -> bool:
    """No visible effect. Typing and focus clicks on fields act inside the field, so they never count as misses."""
    if entry.get("operation") == "SUBMIT":
        return not entry.get("page_changed")
    if entry.get("tool_ok") is not None:
        return entry["tool_ok"] is False
    if entry.get("page_changed") or entry.get("url") != entry.get("url_before"):
        return False
    return entry.get("operation") != "TYPE_TEXT" and not (
        entry.get("operation") == "CLICK" and entry.get("role") in EDITABLE_ROLES)


def _effects(step: Step) -> dict:
    flags = {"TYPE_TEXT": "typed", "SUBMIT": "submitted", "GOTO": "searched", "SCROLL_TO_TEXT": "scrolled"}
    purposes = {"open_search": "opened_search", "open": "opened", "pick": "picked", "submit": "submitted"}
    effects = {}
    if step.operation in flags:
        effects[flags[step.operation]] = True
    if step.purpose in purposes:
        effects[purposes[step.purpose]] = True
    if step.purpose == "act" and step.operation in ("CLICK", "SELECT", "SUBMIT"):
        effects["acted"] = True
    return effects


def _document(url: str) -> str:
    return urldefrag(url)[0]


def _evidence(sub: Subgoal, page: Mapping) -> str:
    if sub.kind == "FIND":
        return f'title shows "{title_subject(page.get("title", ""))}"'
    if sub.kind in ("JUMP", "SCROLL") and fragment_names(page["url"], sub.target):
        return f"url fragment #{urldefrag(page['url'])[1]}"
    if sub.kind in ("JUMP", "SCROLL") and heading_in_view(page, sub.target):
        return f'heading "{sub.target}" in view'
    return f"{sub.kind} {sub.target}".strip() + " done"


class Controller:
    def __init__(self, goal: str, program: Program, *, compile_meta: Mapping | None = None,
                 predict: Callable | None = None, fallback: Callable | None = None,
                 discover: Callable[[str], str | None] | None = None,
                 templates: SearchTemplates | None = None) -> None:
        self.goal, self.program = goal, program
        self._predict = predict or policy.laya_predict
        self._fallback = fallback or policy.decide
        self._discover = discover or (lambda url: None)
        self._templates = templates if templates is not None else SearchTemplates()
        self.plans: list[dict] = [{"program": render_program(program), "source": program.source,
                                   **(compile_meta or {})}]
        self.trace: list[dict] = []
        self.actor_calls = 0
        self._current = 0
        self._progress = Progress()
        self._excluded: set[tuple[str, str]] = set()  # (document url, normalized label) that had no effect
        self._start_url: str | None = None
        self._pending: _Pending | None = None
        self._asked_hosts: set[str] = set()

    @classmethod
    def from_goal(cls, goal: str, *, compile_fn: Callable | None = None, **deps) -> "Controller":
        cache = default_cache() if os.environ.get("LAYA_PROGRAM_CACHE", "1") != "0" else None
        program, meta = (compile_fn or compile_goal)(goal, cache=cache)
        log.info("compiled goal (%s): %s", meta.get("source"), render_program(program).replace("\n", " | "))
        return cls(goal, program, compile_meta=meta, **deps)

    def decide(self, page: Mapping, history: Sequence[Mapping]) -> dict:
        started = time.perf_counter()
        self._absorb(history)
        if self._start_url is None:
            self._start_url = page["url"]
        elements, targets, _ = action_space(prune_actions(page["actions"], self.goal))
        stop = self._advance(page, elements, started)
        if stop is not None:
            return stop
        sub = self.program.subgoals[self._current]
        if sub.kind == "DO":
            return self._goal_mode(page, history, started)
        p = self._progress
        if p.misses >= MAX_MISSES_PER_SUBGOAL or p.actions >= MAX_ACTIONS_PER_SUBGOAL:
            return self._stop("BLOCKED", f"{sub.kind} {sub.target}: {p.actions} actions, {p.misses} without effect",
                              started)
        here = _document(page["url"])
        usable = [e for e in elements if (here, normalize(e["label"])) not in self._excluded]
        template = self._template(page["url"]) if sub.kind == "FIND" else None
        step = next_step(sub, page, usable, p, template)
        if step is None:
            return self._stop("BLOCKED", f"{sub.kind} {sub.target}: no step left on this page", started)
        decision = self._act(step, usable, elements, targets, history, started)
        if decision is None:
            return self._stop("BLOCKED", f"{sub.kind} {sub.target}: no element for {step.instruction}", started)
        label = next((e["label"] for e in elements if e["index"] == decision["target"]), "")
        self._pending = _Pending(len(history), self._current, step, decision["target"], label)
        self.trace.append({"subgoal": self._current, "kind": sub.kind, "route": decision["route"],
                           "step": asdict(step), "progress": asdict(p)})
        return decision

    # -- bookkeeping ---------------------------------------------------------------------------------------------

    def _absorb(self, history: Sequence[Mapping]) -> None:
        """Credit the last decision's outcome; a decision the Agent never executed leaves no trace."""
        pending, self._pending = self._pending, None
        if pending is None or len(history) <= pending.history_len or pending.subgoal != self._current:
            return
        entry = history[pending.history_len]
        p = self._progress
        if _missed(entry):
            if pending.index is not None:
                self._excluded.add((_document(entry.get("url_before") or ""), normalize(pending.label)))
            self._progress = replace(p, actions=p.actions + 1, misses=p.misses + 1)
            return
        self._progress = replace(p, actions=p.actions + 1, **_effects(pending.step))
        if pending.step.purpose in ("search", "submit") and pending.step.operation != "TYPE_TEXT":
            template = learn_template(entry.get("url") or "", self.program.subgoals[self._current].target,
                                      entry.get("url_before") or "")
            if template:
                self._templates.put(entry["url"], template)

    def _complete(self, sub: Subgoal, page: Mapping, elements: Sequence[Mapping]) -> bool:
        p = self._progress
        if sub.kind in PAGE_KINDS:
            done = bool(satisfied(sub, page, self._start_url or page["url"]))
            return done or (sub.kind in ("SCROLL", "JUMP") and p.scrolled)
        if sub.kind == "FILL":
            return p.picked or (p.typed and matching_option(elements, sub.value) is None)
        if sub.kind == "SELECT":
            return p.picked or p.acted
        return sub.kind != "DO" and p.acted

    def _advance(self, page: Mapping, elements: Sequence[Mapping], started: float) -> dict | None:
        subgoals = self.program.subgoals
        if self.program.done_text and text_shown(page, self.program.done_text):
            return self._stop("DONE", f'page shows "{self.program.done_text}"', started)
        while self._current < len(subgoals) and self._complete(subgoals[self._current], page, elements):
            log.info("subgoal %d done: %s %s", self._current, subgoals[self._current].kind,
                     subgoals[self._current].target)
            self._current += 1
            self._progress, self._excluded, self._start_url = Progress(), set(), page["url"]
        if self._current >= len(subgoals):
            return self._stop("DONE", _evidence(subgoals[-1], page), started)
        last = subgoals[-1]
        if last.kind in PAGE_KINDS and satisfied(last, page, page["url"]):
            return self._stop("DONE", _evidence(last, page), started)
        return None

    def _template(self, url: str) -> str | None:
        template = self._templates.get(url)
        host = urlsplit(url).hostname or ""
        if template is None and host not in self._asked_hosts:
            self._asked_hosts.add(host)
            template = self._discover(url)
            if template:
                self._templates.put(url, template)
        return template

    # -- grounding -----------------------------------------------------------------------------------------------

    def _ground(self, step: Step, usable: Sequence[Mapping],
                history: Sequence[Mapping]) -> tuple[str | None, str, ActorPick | None]:
        pool = [e for e in usable if step.operation in e["operations"] and
                (not step.roles or e.get("role") in step.roles)]
        if step.index is not None:
            return (step.index if any(e["index"] == step.index for e in pool) else None), "tactic", None
        if not pool:
            return None, "no_candidates", None
        if step.ordinal:
            element = ordinal_pick(pool, step.target_text, step.ordinal,
                                   anchor=lambda: self._anchor(step, pool, history))
            return (element["index"] if element else None), "ordinal", None
        element, how = resolve(step.target_text, pool)
        if element is not None:
            return element["index"], how, None
        picked = self._pick(step, pool, history)
        return picked.index, "actor", picked

    def _pick(self, step: Step, pool: Sequence[Mapping], history: Sequence[Mapping]) -> ActorPick:
        picked = pick_target(step, pool, history_strings(history), predict=self._predict)
        if picked.model not in ("sole", "none"):
            self.actor_calls += 1
        return picked

    def _anchor(self, step: Step, pool: Sequence[Mapping], history: Sequence[Mapping]) -> Mapping | None:
        """An element of the wanted kind, when no group's labels share a word with the ordinal target."""
        element, _ = resolve(step.target_text, pool)
        if element is not None:
            return element
        index = self._pick(step, pool, history).index
        return next((e for e in pool if e["index"] == index), None)

    def _act(self, step: Step, usable: Sequence[Mapping], elements: Sequence[Mapping], targets: Mapping,
             history: Sequence[Mapping], started: float) -> dict | None:
        if step.operation in TOOL_OPERATIONS:
            tool = {"operation": step.operation, "arg": step.target_text, "templates": list(step.templates)}
            return self._decision("TOOL", step, None, "tactic", 1.0, None, started, tool=tool)
        index, how, actor = self._ground(step, usable, history)
        if index is None:
            return None
        key = index
        if step.operation == "SELECT":
            element = next(e for e in elements if e["index"] == index)
            labels = [{"index": o["index"], "label": o["label"].split(" → ", 1)[-1]} for o in element["options"]]
            match, _ = resolve(step.value, labels)
            if match is None:
                return None
            key = match["index"]
        confidence = actor.confidence if actor else 1.0
        return self._decision(targets[step.operation][key]["id"], step, index, how, confidence, actor, started)

    # -- decisions -----------------------------------------------------------------------------------------------

    def _decision(self, choice: str, step: Step, index: str | None, how: str, confidence: float,
                  actor: ActorPick | None, started: float, *, tool: dict | None = None) -> dict:
        return {
            "choice": choice, "operation": step.operation, "target": index,
            "confidence": confidence, "probabilities": {choice: 1.0},
            "operation_probabilities": {step.operation: 1.0},
            "target_probabilities": dict(actor.ranked) if actor else {}, "target_confidence": confidence,
            "raw_answers": {}, "model": actor.model if actor else how, "usage": {},
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "request": actor.request if actor else {}, "actor_ms": actor.latency_ms if actor else 0,
            "route": how, "instruction": step.instruction, "value": step.value or None, "tool": tool,
            "evidence": "",
        }

    def _stop(self, kind: str, evidence: str, started: float) -> dict:
        log.info("%s: %s", kind, evidence)
        self.trace.append({"subgoal": self._current, "kind": kind, "route": "controller", "step": None,
                           "progress": asdict(self._progress), "evidence": evidence})
        step = Step(kind, "", "", "")
        return {**self._decision(kind, step, None, "controller", 1.0, None, started), "evidence": evidence}

    def _goal_mode(self, page: Mapping, history: Sequence[Mapping], started: float) -> dict:
        decision = self._fallback(page, self.goal, history)
        self.actor_calls += 1
        self.trace.append({"subgoal": self._current, "kind": "DO", "route": "goal_fallback", "step": None,
                           "progress": asdict(self._progress)})
        return {**decision, "route": "goal_fallback", "instruction": None, "value": None, "tool": None,
                "evidence": "", "latency_ms": decision.get("latency_ms", round((time.perf_counter() - started)
                                                                               * 1000))}
