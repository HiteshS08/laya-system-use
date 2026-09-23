import json
import logging
from unittest.mock import Mock

import pytest

from jev_ultrafast.verifier import Verdict, completion_verdict, verify_done


def complete(output):
    return Mock(return_value=(output, {"latency_ms": 7}))


def state(page_changed=True):
    history = [{"action": "Go", "kind": "click", "operation": "CLICK", "text": None, "page_changed": page_changed}]
    return {"goal": "g", "page": {"text": "results"}, "history": history}


def test_verify_done_accepts_a_valid_verdict():
    verdict = verify_done("goal", "text", ["a"], complete=complete({"done": True, "reason": "ok"}))
    assert verdict == Verdict(True, "ok", 7)


@pytest.mark.parametrize(
    "output",
    [{"done": "yes", "reason": "x"}, {"done": True}, {"done": True, "reason": "x", "extra": 1}, {"reason": "x"}],
)
def test_verify_done_rejects_invalid_verdicts(output):
    with pytest.raises(ValueError, match="invalid verdict"):
        verify_done("g", "t", [], complete=complete(output))


def test_verify_done_truncates_page_text_and_history():
    c = complete({"done": False, "reason": "no"})
    verify_done("g", "x" * 10000, ["1", "2", "3", "4"], complete=c)
    payload = json.loads(c.call_args.args[1])
    assert len(payload["page_text"]) == 4000 and payload["recent_actions"] == ["2", "3", "4"]


def test_no_verdict_without_history_or_page_change():
    verify = Mock()
    assert completion_verdict({"goal": "g", "page": {"text": ""}, "history": []}, verify=verify) is None
    assert completion_verdict(state(page_changed=False), verify=verify) is None
    verify.assert_not_called()


def test_verdict_uses_goal_page_text_and_rendered_history():
    verify = Mock(return_value=Verdict(True, "ok", 1))
    assert completion_verdict(state(), verify=verify).done is True
    verify.assert_called_once_with("g", "results", ["CLICK Go"])


def test_a_failing_verifier_is_logged_and_treated_as_not_done(caplog):
    verify = Mock(side_effect=ValueError("bad json"))
    with caplog.at_level(logging.WARNING, logger="verifier"):
        assert completion_verdict(state(), verify=verify) is None
    assert "bad json" in caplog.text
