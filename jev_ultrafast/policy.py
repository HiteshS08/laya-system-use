"""Laya policy: shortlist observed elements, ask the fine-tuned model, map the answer to an executable action id."""

import os
import time
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache

from .candidates import OPERATIONS, Candidate
from .formatter import build_request, history_strings
from .model import action_space, validate_choice
from .shortlister import shortlist
from .textmodel import choose_option

Predict = Callable[[dict, dict], dict]


@lru_cache(maxsize=1)
def _agent():
    import laya

    checkpoint = os.environ.get("LAYA_CHECKPOINT")
    if not checkpoint:
        raise RuntimeError("POLICY_BACKEND=laya needs LAYA_CHECKPOINT (a directory or Hugging Face id).")
    return laya.load(checkpoint)


def laya_predict(state: dict, questions: dict) -> dict:
    return _agent().system_one(state, questions)


def candidates_from(elements: Sequence[Mapping]) -> list[Candidate]:
    return [
        Candidate(e["index"], e["label"], e.get("role", ""), str(e.get("value") or ""), frozenset(e["operations"]))
        for e in elements
    ]


def _valid(answer: Mapping, ids: Sequence[str], name: str) -> Mapping:
    try:
        return validate_choice(answer, ids)
    except ValueError as exc:
        raise ValueError(f"Invalid Laya answer for {name}; no action executed.") from exc


def _sure(choice: str) -> dict:
    return {"choice": choice, "probabilities": {choice: 1.0}, "confidence": 1.0}


def _control(controls: Mapping[str, Mapping], history: Sequence[Mapping], started: float) -> dict:
    last = history[-1].get("operation") if history else None
    if "WAIT" in controls and last != "WAIT":
        operation = "WAIT"
    elif "SCROLL_DOWN" in controls:
        operation = "SCROLL_DOWN"
    else:
        operation = "BLOCKED"
    choice = controls[operation]["id"] if operation in controls else operation
    return _result(choice, operation, None, _sure(operation), None, {}, {}, {"state": {}, "questions": {}}, started)


def _result(choice, operation, target, op_answer, target_answer, raw, model_info, request, started, probabilities=None):
    return {
        "choice": choice, "operation": operation, "target": target,
        "confidence": op_answer["confidence"],
        "probabilities": probabilities or {choice: 1.0},
        "operation_probabilities": op_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "raw_answers": raw, "model": model_info.get("model", "laya"), "usage": model_info.get("usage", {}),
        "latency_ms": round((time.perf_counter() - started) * 1000), "request": request,
    }


def _pick_target(answers: Mapping, operation: str, candidates: Sequence[Candidate]) -> Mapping:
    if len(candidates) == 1:
        return _sure(candidates[0].id)
    name = f"{operation.lower()}_target"
    return _valid(answers.get(name, {}), [c.id for c in candidates], name)


def _select_action(goal, element, pick_option) -> str:
    options = element["options"]
    labels = [o["label"].split(" → ", 1)[-1] for o in options]
    picked = pick_option(goal, element["label"], labels)
    return options[labels.index(picked)]["index"]


def decide(state: Mapping, goal: str, history: Sequence[Mapping], *, predict: Predict = laya_predict,
           pick_option: Callable[[str, str, Sequence[str]], str] = choose_option) -> dict:
    started = time.perf_counter()
    elements, targets, controls = action_space(state["actions"])
    past = history_strings(history)
    everything = candidates_from(elements)
    by_op = {op: shortlist(goal, past, [c for c in everything if op in c.ops]) for op in OPERATIONS}
    ops = [op for op in OPERATIONS if by_op[op]]
    if not ops:
        return _control(controls, history, started)
    request_state, questions = build_request(goal, past, by_op)
    info = predict(request_state, questions) if questions else {}
    answers = info.get("answers", {})
    op_answer = _valid(answers.get("operation", {}), ops, "operation") if len(ops) > 1 else _sure(ops[0])
    operation = op_answer["choice"]
    target_answer = _pick_target(answers, operation, by_op[operation])
    target = target_answer["choice"]
    if operation == "SELECT":
        key = _select_action(goal, elements[int(target) - 1], pick_option)
        probabilities = {targets[operation][key]["id"]: target_answer["probabilities"][target]}
    else:
        key = target
        probabilities = {targets[operation][t]["id"]: p for t, p in target_answer["probabilities"].items()}
    choice = targets[operation][key]["id"]
    request = {"state": request_state, "questions": questions}
    return _result(choice, operation, target, op_answer, target_answer, answers, info, request, started, probabilities)
