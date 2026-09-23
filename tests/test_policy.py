from unittest.mock import Mock

import pytest

from jev_ultrafast import model, policy
from jev_ultrafast.formatter import history_strings

EL = [
    {"id": "e1", "kind": "fill", "label": "Search", "role": "textbox", "value": "", "node": 10},
    {"id": "e2", "kind": "click", "label": "Open Search", "role": "textbox", "value": "", "node": 10},
    {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
]
CONTROLS = [{"id": "wait", "kind": "wait", "label": "Wait"},
            {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560}]


def page(*actions):
    return {"url": "https://x.test", "title": "T", "text": "t", "actions": [*actions, *CONTROLS]}


def answer(options, selected):
    others = [o for o in options if o != selected]
    probabilities = {selected: 0.9, **{o: 0.1 / len(others) for o in others}}
    return {"choice": selected, "confidence": 0.9, "probabilities": probabilities}


def predictor(**picks):
    """picks: operation="CLICK", click_target="2" ... Questions without a pick get their first option."""
    def predict(state, questions):
        predict.seen = (state, questions)
        answers = {qid: answer(list(q["criteria"]), picks.get(qid, next(iter(q["criteria"]))))
                   for qid, q in questions.items()}
        return {"answers": answers, "model": "laya-test", "usage": {"input_tokens": 1}}

    return predict


def test_click_maps_the_shortlisted_target_back_to_an_action_id():
    d = policy.decide(page(*EL), "search", [], predict=predictor(operation="CLICK", click_target="2"))
    assert (d["choice"], d["operation"], d["target"]) == ("e3", "CLICK", "2")
    assert "e3" in d["probabilities"] and d["model"] == "laya-test"


def test_lone_text_field_needs_no_target_question():
    p = predictor(operation="TYPE_TEXT")
    d = policy.decide(page(*EL), "search", [], predict=p)
    assert (d["choice"], d["operation"], d["target"]) == ("e1", "TYPE_TEXT", "1")
    assert "type_text_target" not in p.seen[1]


def test_single_candidate_skips_the_model_entirely():
    predict = Mock()
    d = policy.decide(page(EL[2]), "go", [], predict=predict)
    predict.assert_not_called()
    assert d["choice"] == "e3" and d["confidence"] == 1.0


def test_invalid_answer_is_rejected():
    bad = Mock(return_value={"answers": {"operation": answer(["CLICK", "TYPE_TEXT"], "CLICK"),
                                         "click_target": {"choice": "999"}}, "model": "m", "usage": {}})
    with pytest.raises(ValueError, match="Invalid Laya answer"):
        policy.decide(page(*EL), "search", [], predict=bad)


def test_missing_operation_answer_is_rejected_when_the_question_was_asked():
    # EL's e1/e2 share one node, so element "1" supports both TYPE_TEXT and CLICK, and element
    # "2" (e3) supports CLICK too: by_op["CLICK"] = ["1", "2"], by_op["TYPE_TEXT"] = ["1"], so
    # the operation question IS asked (len(ops) == 2 > 1). The mocked response is well-formed
    # for click_target (a valid answer over ids "1"/"2") but omits the "operation" key
    # entirely, simulating a malformed/truncated Laya reply. This must raise, not silently fall
    # back to the first operation ("CLICK") with confidence 1.0 — if it fell back, click_target
    # validation would succeed too (both "1" and "2" are valid CLICK targets), so only
    # validating the operation answer itself can catch this bug.
    missing = Mock(return_value={"answers": {"click_target": answer(["1", "2"], "2")},
                                  "model": "m", "usage": {}})
    with pytest.raises(ValueError, match="Invalid Laya answer"):
        policy.decide(page(*EL), "search", [], predict=missing)


def test_select_uses_laya_for_the_dropdown_and_the_text_model_for_the_option():
    select = [
        {"id": "s1", "kind": "select", "label": "Sort → Price", "role": "combobox", "value": "price",
         "current_value": "Relevance", "node": 30},
        {"id": "s2", "kind": "select", "label": "Sort → Rating", "role": "combobox", "value": "rating",
         "current_value": "Relevance", "node": 30},
    ]
    clicks = [{"id": "c1", "kind": "click", "label": "A", "role": "link", "value": "", "node": 1},
              {"id": "c2", "kind": "click", "label": "B", "role": "link", "value": "", "node": 2}]
    pick = Mock(return_value="Rating")
    d = policy.decide(page(*clicks, *select), "best rated", [], predict=predictor(operation="SELECT"),
                      pick_option=pick)
    assert (d["choice"], d["operation"], d["target"]) == ("s2", "SELECT", "3")
    assert pick.call_args.args == ("best rated", "Sort", ["Price", "Rating"])


def test_history_is_rendered_into_the_request_state():
    hist = [{"action": "Search", "kind": "fill", "operation": "TYPE_TEXT", "text": "nfl"}]
    p = predictor(operation="CLICK", click_target="1")
    policy.decide(page(*EL), "search", hist, predict=p)
    assert p.seen[0]["recent_actions"] == ["TYPE_TEXT Search = nfl"]
    assert history_strings(hist) == ["TYPE_TEXT Search = nfl"]


def test_no_elements_falls_back_to_wait_then_scroll_then_blocked():
    only_controls = {"url": "u", "title": "t", "text": "", "actions": list(CONTROLS)}
    assert policy.decide(only_controls, "g", [])["choice"] == "wait"
    assert policy.decide(only_controls, "g", [{"action": "Wait", "kind": "wait", "operation": "WAIT"}])["choice"] \
        == "scroll_down"
    nothing = {"url": "u", "title": "t", "text": "", "actions": []}
    assert policy.decide(nothing, "g", [])["choice"] == "BLOCKED"


def test_choose_dispatches_to_the_policy_when_selected(monkeypatch):
    monkeypatch.setenv("POLICY_BACKEND", "laya")
    monkeypatch.setattr(policy, "decide", Mock(return_value={"choice": "sentinel"}))
    assert model.choose(page(*EL), "g", [])["choice"] == "sentinel"


def test_missing_checkpoint_fails_with_the_variable_name(monkeypatch):
    monkeypatch.delenv("LAYA_CHECKPOINT", raising=False)
    policy._agent.cache_clear()
    with pytest.raises(RuntimeError, match="LAYA_CHECKPOINT"):
        policy._agent()
