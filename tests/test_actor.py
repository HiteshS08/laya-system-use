import pytest

from jev_ultrafast.actor import actor_request, pick_target
from jev_ultrafast.candidates import Candidate
from jev_ultrafast.planner import PlanStep

STEP = PlanStep("CLICK", "8 References", "", "Click the 8 References link in the contents.")


def el(index, label, ops=("CLICK",), **extra):
    return {"index": str(index), "label": label, "role": "link", "operations": list(ops), "value": "", **extra}


def predictor(choice, probs):
    def predict(state, questions):
        predict.seen = (state, questions)
        return {"answers": {next(iter(questions)): {"choice": choice, "confidence": probs[choice],
                                                    "probabilities": probs}}, "model": "laya-test"}
    return predict


def test_request_has_one_target_question_and_the_step_as_goal():
    cands = [Candidate("1", "Toggle References subsection", "button"), Candidate("2", "8 References", "link")]
    state, questions = actor_request(STEP, cands, ["CLICK a", "CLICK b", "CLICK c", "CLICK d"])
    assert state == {"goal": STEP.instruction, "recent_actions": ["CLICK b", "CLICK c", "CLICK d"]}
    assert list(questions) == ["click_target"]
    assert questions["click_target"]["criteria"] == {"1": "Toggle References subsection (button)",
                                                     "2": "8 References (link)"}


def test_pick_returns_choice_confidence_and_ranking():
    elements = [el(1, "Toggle References subsection"), el(2, "8 References")]
    p = predictor("2", {"1": 0.2, "2": 0.8})
    pick = pick_target(STEP, elements, [], predict=p)
    assert (pick.index, pick.confidence, pick.ranked[0]) == ("2", 0.8, ("2", 0.8))
    assert "operation" not in p.seen[1] and pick.model == "laya-test"


def test_single_field_needs_no_model_call():
    def never(*_):
        raise AssertionError("model must not be called")
    typing = PlanStep("TYPE_TEXT", "search box", "Ada", "Type Ada into the search box.")
    pick = pick_target(typing, [el(2, "Search Wikipedia", ("TYPE_TEXT",))], [], predict=never)
    assert (pick.index, pick.confidence) == ("2", 1.0)


def test_single_click_candidate_is_offered_but_not_trusted():
    def never(*_):
        raise AssertionError("model must not be called")
    pick = pick_target(STEP, [el(2, "Donate")], [], predict=never)
    assert (pick.index, pick.confidence, pick.ranked) == ("2", 0.0, (("2", 0.0),))


def test_no_candidate_gives_no_pick():
    assert pick_target(STEP, [], [], predict=predictor("1", {"1": 1.0})).index is None


def test_answer_outside_the_offer_is_rejected():
    elements = [el(1, "A"), el(2, "B")]
    with pytest.raises(ValueError, match="Invalid Laya answer"):
        pick_target(STEP, elements, [], predict=predictor("9", {"9": 1.0}))


def test_context_rendering_follows_the_switch(monkeypatch):
    elements = [el(1, "Mary Mallon", landmark="main", section="From today's featured article"), el(2, "Read")]
    p = predictor("1", {"1": 0.9, "2": 0.1})
    pick_target(STEP, elements, [], predict=p)
    assert p.seen[1]["click_target"]["criteria"]["1"] == "Mary Mallon (link)"
    monkeypatch.setenv("ACTOR_CONTEXT", "1")
    pick_target(STEP, elements, [], predict=p)
    assert p.seen[1]["click_target"]["criteria"]["1"] == "Mary Mallon (link) [main > From today's featured article]"
