"""The program backend end to end: real Agent, real Browser and snapshot.js, on local pages. Fake compiler and actor.

Needs a throwaway Chrome on BU_CDP_URL; skipped otherwise.
"""

import re
from pathlib import Path

import pytest

from tests.test_snapshot_live import _chrome_up

SITE = Path(__file__).parent / "fixtures" / "site"
pytestmark = pytest.mark.skipif(not _chrome_up(), reason="needs a throwaway Chrome on BU_CDP_URL")


def lexical_predict(state, questions):
    """Stands in for Laya: the option sharing the most words with the instruction wins."""
    words = set(re.findall(r"[a-z0-9]+", state["goal"].lower()))
    name, question = next(iter(questions.items()))
    scores = {i: len(words & set(re.findall(r"[a-z0-9]+", text.lower()))) for i, text in question["criteria"].items()}
    best = max(scores, key=lambda i: (scores[i], -int(i)))
    probs = {i: (0.9 if i == best else 0.1 / max(len(scores) - 1, 1)) for i in scores}
    return {"answers": {name: {"choice": best, "confidence": 0.9, "probabilities": probs}}}


def run(monkeypatch, tmp_path, start, program_text, steps=8):
    from jev_ultrafast import agent as loop
    from jev_ultrafast import controller as ctl
    from jev_ultrafast.program import parse_program

    monkeypatch.setenv("POLICY_BACKEND", "program")
    monkeypatch.setenv("LAYA_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(ctl, "compile_goal", lambda goal, **kw: (parse_program(program_text), {"attempts": 1}))
    monkeypatch.setattr(ctl.policy, "laya_predict", lexical_predict)
    with loop.Agent((SITE / start).as_uri(), "fixture goal") as agent:
        for _ in range(steps):
            agent.command("tick")
            if agent.state["status"] in {"done", "blocked"}:
                break
        return agent.state


def test_find_by_typing_into_search_then_opening_the_result(monkeypatch, tmp_path):
    state = run(monkeypatch, tmp_path, "home.html", "FIND Ada Lovelace")
    assert state["status"] == "done" and state["page"]["url"].endswith("ada.html")
    assert [h["operation"] for h in state["history"]] == ["TYPE_TEXT", "SUBMIT", "CLICK"]


def test_find_then_follow_an_on_page_link(monkeypatch, tmp_path):
    state = run(monkeypatch, tmp_path, "ada.html", "FIND Ada Lovelace\nFIND Charles Babbage")
    assert state["status"] == "done" and state["page"]["url"].endswith("babbage.html")
    assert len(state["history"]) == 1


def test_jump_sets_the_fragment(monkeypatch, tmp_path):
    state = run(monkeypatch, tmp_path, "ada.html", "JUMP References")
    assert state["status"] == "done" and state["page"]["url"].endswith("#References")


def test_scroll_until_heading_visible(monkeypatch, tmp_path):
    state = run(monkeypatch, tmp_path, "ada.html", "SCROLL External links")
    assert state["status"] == "done" and state["page"]["scroll"]["y"] > 1000


def test_open_the_nth_repeated_item(monkeypatch, tmp_path):
    state = run(monkeypatch, tmp_path, "list.html", "OPEN comments @2")
    assert state["status"] == "done" and state["page"]["url"].endswith("item2.html")


def test_fill_with_suggestion_then_submit(monkeypatch, tmp_path):
    state = run(monkeypatch, tmp_path, "form.html",
                "FILL Where to? = London\nCLICK Search\nDONE_WHEN Results for London")
    assert state["status"] == "done" and state["page"]["url"].endswith("done.html")
