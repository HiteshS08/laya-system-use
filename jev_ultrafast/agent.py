"""The complete agent loop. Typed choices, observable state, bounded execution."""

import base64
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

from .browser import Browser, StalePage
from .controller import Controller
from .model import action_space, choose, field_context, field_text, valid_text_value
from .pilot import Pilot
from .questions import MAX_STEPS
from .search import default_templates, discover
from .tools import run_tool
from .verifier import completion_verdict

log = logging.getLogger("agent")


def _ms_since(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _blocked_by_repeated_no_change(history: list[dict]) -> bool:
    """Three actions in a row with no visible effect (and no deliberate wait) means the run is stuck.

    Applies equally to element actions and tool actions (SCROLL_TO_TEXT/GOTO): a tool that keeps reporting
    tool_ok without ever changing the page is just as stuck as a click that keeps doing nothing.
    """
    repeated = history[-3:]
    return len(repeated) == 3 and all(h["page_changed"] is False and h["kind"] != "wait" for h in repeated)


class Agent:
    def __init__(self, url, goals, *, record_dir=None, screenshots=False):
        task = goals.strip() if isinstance(goals, str) else "\n".join(goals).strip()
        if not task:
            raise ValueError("Supply a task")
        plan = [task]
        self.pending_text = None
        backend = os.environ.get("POLICY_BACKEND")
        self.pilot = Pilot(task) if backend == "planner" else None
        self.browser = Browser(url)
        self.record_dir = Path(record_dir) if record_dir else None
        self.screenshots = screenshots or bool(record_dir)
        try:
            if backend == "program":
                # The only LLM call of the run (none when the goal's program is cached).
                self.pilot = Controller.from_goal(task, discover=lambda page: discover(self.browser, page),
                                                  templates=default_templates())
            page = self.browser.observe(screenshot=self.screenshots)
        except Exception:
            self.browser.close()
            raise
        self.state = dict(
            browser=self.browser,
            goal="\n".join(plan),
            page=page,
            decision=None,
            history=[],
            status="ready",
            plan=plan,
            plan_index=0,
            decisions=[],
            text_calls=[],
            elapsed_ms=0,
            started_at=None,
            record=bool(self.record_dir),
        )
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            (self.record_dir / "000000.jpg").write_bytes(base64.b64decode(page["screenshot"]))

    def snapshot(self):
        return {
            **{k: v for k, v in self.state.items() if k != "browser"},
            "elements": action_space(self.state["page"]["actions"])[0],
        }

    def command(self, name, body=None):
        body = body or {}
        state = self.state
        if name == "tick":
            try:
                self.command("predict", {})
                return self.command("act", {"fingerprint": state["page"]["fingerprint"]})
            except StalePage:
                state["decision"] = None
                state["status"] = "ready"
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
        elif name == "predict":
            if not state["browser"]:
                raise ValueError("Start a demo first")
            if state["started_at"] is None:
                state["started_at"] = time.perf_counter()
            if not state["browser"].fresh(state["page"]):
                state["page"] = state["browser"].observe(screenshot=self.screenshots)
            state["decision"] = None
            if state["status"] in {"done", "blocked"}:
                raise ValueError("This run has stopped. Start a fresh demo.")
            if len(state["decisions"]) >= MAX_STEPS * 2:
                raise ValueError("Reached the demo's model-call budget")
            state["decision"] = (self.pilot.decide(state["page"], state["history"]) if self.pilot
                                 else choose(state["page"], state["goal"], state["history"]))
            state["decisions"].append(
                {
                    **state["decision"],
                    "fingerprint": state["page"]["fingerprint"],
                    "elapsed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                }
            )
            state["status"] = "predicted"
        elif name == "act":
            decision, page = state["decision"], state["page"]
            if not decision or body.get("fingerprint") != page["fingerprint"]:
                raise ValueError("Observe and choose before acting")
            # Consume once, before any mutation or model call. A retry cannot double-click.
            state["decision"] = None
            selected = decision["choice"]
            if selected in {"DONE", "BLOCKED"}:
                if not state["browser"].fresh(page):
                    state["status"] = "ready"
                    raise StalePage("Page changed since the decision. Choose again.")
                state["status"] = "done" if selected == "DONE" else "blocked"
                state["plan_index"] = int(selected == "DONE")
                state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
                return self.snapshot()
            if decision.get("tool"):
                return self._act_tool(decision, page)
            action = next(a for a in page["actions"] if a["id"] == selected)
            if len(state["history"]) >= MAX_STEPS:
                state["status"] = "blocked"
                raise ValueError(f"Stopped at the {MAX_STEPS}-action demo budget")
            text, helper = None, None
            if action["kind"] == "fill" and decision.get("value") is not None:
                text = valid_text_value(decision["value"])
            elif action["kind"] == "fill":
                if not state["browser"].fresh(page):
                    raise StalePage("Page changed before text generation. Choose again.")
                context = field_context(state["goal"], action, page, state["history"])
                if self.pending_text and self.pending_text[0] == context:
                    _, text, helper = self.pending_text
                else:
                    text, helper = field_text(context)
                    self.pending_text = (context, text, helper)
                    state["text_calls"].append({**helper, "field": action["label"], "value": text})
            # Browser.act checks freshness immediately before input, including after text generation.
            act_started = time.perf_counter()
            state["browser"].act(action, page, text=text)
            act_ms = _ms_since(act_started)
            self.pending_text = None
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            # Record execution before observing. A stale post-action observation must not erase the action.
            state["history"].append(
                {
                    "step": len(state["history"]) + 1,
                    "action": action["label"],
                    "kind": action["kind"],
                    "role": action.get("role"),
                    "choice": selected,
                    "probability": decision["probabilities"][selected],
                    "confidence": decision["confidence"],
                    "latency_ms": decision["latency_ms"],
                    "text": text,
                    "text_helper": helper["model"] if helper else None,
                    "text_latency_ms": helper["latency_ms"] if helper else 0,
                    "operation": decision["operation"],
                    "target": decision["target"],
                    "page_changed": None,
                    "url": page["url"],
                    "url_before": page["url"],
                    "route": decision.get("route"),
                    "instruction": decision.get("instruction"),
                    "usage": decision["usage"],
                    "executed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
                    "elapsed_ms": state["elapsed_ms"],
                    "act_ms": act_ms,
                }
            )
            observe_started = time.perf_counter()
            state["page"] = state["browser"].observe(screenshot=self.screenshots)
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            state["history"][-1].update(
                page_changed=state["page"]["fingerprint"] != page["fingerprint"],
                url=state["page"]["url"],
                elapsed_ms=state["elapsed_ms"],
                observe_ms=_ms_since(observe_started),
            )
            if state["record"]:
                (self.record_dir / f"{state['elapsed_ms']:06d}.jpg").write_bytes(
                    base64.b64decode(state["page"]["screenshot"])
                )
            state["status"] = "blocked" if _blocked_by_repeated_no_change(state["history"]) else "ready"
            if state["status"] == "ready" and os.environ.get("POLICY_BACKEND") == "laya":
                verdict = completion_verdict(state)
                if verdict:
                    state["verdicts"] = [*state.get("verdicts", []), asdict(verdict)]
                    if verdict.done:
                        state["status"] = "done"
                        state["plan_index"] = 1
                        state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
        else:
            raise ValueError("Unknown command")
        return self.snapshot()

    def _act_tool(self, decision, page):
        state = self.state
        if len(state["history"]) >= MAX_STEPS:
            state["status"] = "blocked"
            raise ValueError(f"Stopped at the {MAX_STEPS}-action demo budget")
        tool = decision["tool"]
        act_started = time.perf_counter()
        try:
            ran = run_tool(state["browser"], tool["operation"], tool["arg"], templates=tool.get("templates", ()))
        except ValueError as exc:
            log.warning("tool rejected: %s", exc)
            ran = False
        act_ms = _ms_since(act_started)
        state["history"].append({
            "step": len(state["history"]) + 1, "action": tool["arg"], "kind": "tool", "choice": "TOOL",
            "probability": 1.0, "confidence": decision["confidence"], "latency_ms": decision["latency_ms"],
            "text": None, "text_helper": None, "text_latency_ms": 0, "operation": tool["operation"],
            "target": None, "tool_ok": ran, "page_changed": False, "url": page["url"], "url_before": page["url"],
            "route": decision.get("route"), "instruction": decision.get("instruction"),
            "usage": {}, "executed_ms": round((time.perf_counter() - state["started_at"]) * 1000),
            "act_ms": act_ms, "observe_ms": 0,
        })
        if ran:
            observe_started = time.perf_counter()
            state["page"] = state["browser"].observe(screenshot=self.screenshots)
            state["history"][-1].update(page_changed=state["page"]["fingerprint"] != page["fingerprint"],
                                        url=state["page"]["url"], observe_ms=_ms_since(observe_started))
        state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
        state["history"][-1]["elapsed_ms"] = state["elapsed_ms"]
        state["status"] = "blocked" if _blocked_by_repeated_no_change(state["history"]) else "ready"
        return self.snapshot()

    def run(self):
        while self.state["status"] not in {"done", "blocked"}:
            yield self.command("tick")

    def close(self):
        self.browser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
