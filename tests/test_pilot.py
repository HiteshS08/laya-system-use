from unittest.mock import Mock

from jev_ultrafast.pilot import Pilot
from jev_ultrafast.planner import Plan, PlanStep

URL = "https://example.test/"
ACTIONS = [
    {"id": "e1", "kind": "fill", "label": "Search", "role": "searchbox", "value": "", "node": 10},
    {"id": "e2", "kind": "click", "label": "Open Search", "role": "searchbox", "value": "", "node": 10},
    {"id": "e3", "kind": "click", "label": "Go", "role": "button", "value": "", "node": 20},
    {"id": "e4", "kind": "click", "label": "View source", "role": "link", "node": 30,
     "href": "https://example.test/w/index.php?title=X&action=edit"},
    {"id": "wait", "kind": "wait", "label": "Wait"},
]


def page(url=URL, text="Search the site"):
    return {"url": url, "title": "Example", "text": text, "outline": "", "actions": ACTIONS}


def cont(*steps):
    return Plan("continue", "", tuple(steps), latency_ms=5, request_chars=100)


CLICK_GO = PlanStep("CLICK", "Go", "", "Click the Go button.")
TYPE_ADA = PlanStep("TYPE_TEXT", "Search", "Ada Lovelace", 'Type "Ada Lovelace" into Search.')


def done_entry(label, op="CLICK", before=URL, after=URL, changed=True, text=None):
    return {"operation": op, "kind": "click", "action": label, "text": text,
            "url_before": before, "url": after, "page_changed": changed}


def pilot(plans, **kw):
    plan_fn = Mock(side_effect=plans)
    kw.setdefault("predict", Mock(side_effect=AssertionError("actor not expected")))
    kw.setdefault("pick_fn", Mock(side_effect=AssertionError("pick not expected")))
    kw.setdefault("fallback", Mock(side_effect=AssertionError("fallback not expected")))
    return Pilot("Find Ada Lovelace", plan_fn=plan_fn, **kw), plan_fn


def test_steps_are_queued_and_executed_without_replanning():
    p, plan_fn = pilot([cont(TYPE_ADA, CLICK_GO)])
    d1 = p.decide(page(), [])
    assert (d1["choice"], d1["operation"], d1["value"], d1["route"]) == ("e1", "TYPE_TEXT", "Ada Lovelace", "resolver")
    d2 = p.decide(page(), [done_entry("Search", op="TYPE_TEXT", changed=False, text="Ada Lovelace")])
    assert (d2["choice"], d2["route"]) == ("e3", "resolver")
    assert plan_fn.call_count == 1 and d2["probabilities"] == {"e3": 1.0}


def test_url_change_discards_queued_steps_and_replans():
    p, plan_fn = pilot([cont(CLICK_GO, TYPE_ADA), cont(CLICK_GO)])
    p.decide(page(), [])
    d = p.decide(page(url=URL + "results"), [done_entry("Go", after=URL + "results")])
    assert plan_fn.call_count == 2 and d["choice"] == "e3"
    assert plan_fn.call_args.args[4] == []  # no failures to report
    assert plan_fn.call_args.args[3] == ["Click the Go button."]  # completed steps


def test_step_with_no_supporting_element_is_reported_and_replanned_then_falls_back():
    select = PlanStep("SELECT", "Country", "India", "Select India in Country.")
    fallback = Mock(return_value={"choice": "wait", "operation": "WAIT", "probabilities": {"wait": 1.0}})
    p, plan_fn = pilot([cont(select), cont(select)], fallback=fallback)
    d = p.decide(page(), [])
    assert plan_fn.call_count == 2
    assert "SELECT Country (no matching element)" in plan_fn.call_args.args[4]
    assert d["route"] == "planner_fallback" and d["choice"] == "wait"


def test_done_and_blocked_stop_the_run():
    p, _ = pilot([Plan("done", "Search the site", ())])
    d = p.decide(page(), [])
    assert (d["choice"], d["evidence"], d["route"]) == ("DONE", "Search the site", "planner")
    p, _ = pilot([Plan("blocked", "", ())])
    assert p.decide(page(), [])["choice"] == "BLOCKED"


def test_planner_failure_falls_back_to_the_actor_only_policy():
    fallback = Mock(return_value={"choice": "e3", "operation": "CLICK", "probabilities": {"e3": 0.7}})
    p, _ = pilot([ValueError("no valid plan")], fallback=fallback)
    d = p.decide(page(), [])
    assert d["route"] == "planner_fallback" and d["choice"] == "e3"
    assert p.plans[-1]["error"] == "no valid plan"


def test_failed_action_is_excluded_and_triggers_a_replan():
    # After Go failed, only "Open Search" supports CLICK; the planner declines it, so Go is reported and replanned.
    p, plan_fn = pilot([cont(CLICK_GO), cont(CLICK_GO), cont(TYPE_ADA)], pick_fn=Mock(return_value=None))
    p.decide(page(), [])
    d = p.decide(page(), [done_entry("Go", changed=False)])
    assert plan_fn.call_count == 3  # replanned after the failure; Go is excluded, so replanned again
    assert "CLICK Go (no effect)" in plan_fn.call_args.args[4]
    assert d["choice"] == "e1"


def test_detour_links_are_never_offered(monkeypatch):
    monkeypatch.setenv("ACTOR_TAU", "0.99")
    view_source = PlanStep("CLICK", "View source", "", "Click View source.")
    fallback = Mock(return_value={"choice": "wait", "operation": "WAIT", "probabilities": {"wait": 1.0}})
    predict = Mock(return_value={"answers": {"click_target": {
        "choice": "1", "confidence": 0.5, "probabilities": {"1": 0.5, "2": 0.5}}}})
    p, _ = pilot([cont(view_source), cont(view_source)], fallback=fallback, predict=predict,
                 pick_fn=Mock(return_value=None))
    assert p.decide(page(), [])["route"] == "planner_fallback"
    for call in predict.call_args_list:
        labels = call.args[1]["click_target"]["criteria"].values()
        assert not any(label.startswith("View source") for label in labels)


def test_tool_steps_become_tool_decisions():
    scroll = PlanStep("SCROLL_TO_TEXT", "External links", "", "Scroll to the External links heading.")
    p, _ = pilot([cont(scroll)])
    d = p.decide(page(), [])
    assert (d["choice"], d["tool"]) == ("TOOL", {"operation": "SCROLL_TO_TEXT", "arg": "External links"})


def test_unsure_actor_defers_to_planner_pick(monkeypatch):
    monkeypatch.setenv("ACTOR_TAU", "0.9")
    vague = PlanStep("CLICK", "the button", "", "Click the button that submits.")
    predict = Mock(return_value={"answers": {"click_target": {
        "choice": "2", "confidence": 0.6, "probabilities": {"1": 0.4, "2": 0.6}}}})
    p, _ = pilot([cont(vague)], predict=predict, pick_fn=Mock(return_value="1"))
    d = p.decide(page(), [])
    assert (d["route"], d["choice"]) == ("planner_pick", "e2")


def test_stall_without_progress_blocks():
    p, _ = pilot([cont(CLICK_GO)] * 10)
    history = [done_entry(f"Thing {i}", changed=False) for i in range(4)]
    assert p.decide(page(), history)["choice"] == "BLOCKED"
