from unittest.mock import Mock

from jev_ultrafast import tools
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


def done_entry(label, op="CLICK", before=URL, after=URL, changed=True, text=None, role=None, tool_ok=None):
    e = {"operation": op, "kind": "tool" if tool_ok is not None else "click", "action": label, "text": text,
         "url_before": before, "url": after, "page_changed": changed}
    if role is not None:
        e["role"] = role
    if tool_ok is not None:
        e["tool_ok"] = tool_ok
    return e


def pilot(plans, **kw):
    plan_fn = Mock(side_effect=plans)
    kw.setdefault("predict", Mock(side_effect=AssertionError("actor not expected")))
    kw.setdefault("pick_fn", Mock(side_effect=AssertionError("pick not expected")))
    kw.setdefault("fallback", Mock(side_effect=AssertionError("fallback not expected")))
    kw.setdefault("search_query_fn", Mock(side_effect=AssertionError("search must not be called")))
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


def test_unexecuted_step_is_requeued_not_lost():
    # StalePage: the Agent can re-decide without ever appending the previous decision to history.
    p, plan_fn = pilot([cont(TYPE_ADA, CLICK_GO)])
    d1 = p.decide(page(), [])
    d2 = p.decide(page(), [])
    assert d1["choice"] == "e1" and d2["choice"] == "e1"
    assert plan_fn.call_count == 1


def test_progress_within_the_stall_window_is_not_blocked():
    # 4 actions where the first one completed a step: not a stall, spec 5.6's window is 4 actions wide.
    actions = [
        {"id": "e1", "kind": "click", "label": "Go", "role": "button", "node": 1},
        {"id": "e2", "kind": "click", "label": "Next", "role": "button", "node": 2},
        {"id": "e3", "kind": "click", "label": "Continue", "role": "button", "node": 3},
        {"id": "e4", "kind": "click", "label": "Submit", "role": "button", "node": 4},
    ]
    pg = {"url": URL, "title": "Example", "text": "Search the site", "outline": "", "actions": actions}
    go = PlanStep("CLICK", "Go", "", "Click Go.")
    nxt = PlanStep("CLICK", "Next", "", "Click Next.")
    keep_going = PlanStep("CLICK", "Continue", "", "Click Continue.")
    submit = PlanStep("CLICK", "Submit", "", "Click Submit.")
    p, _ = pilot([cont(go), cont(nxt), cont(keep_going), cont(submit), Plan("done", "Search the site", ())])
    p.decide(pg, [])
    p.decide(pg, [done_entry("Go", changed=True)])
    p.decide(pg, [done_entry("Go", changed=True), done_entry("Next", changed=False)])
    p.decide(pg, [done_entry("Go", changed=True), done_entry("Next", changed=False),
                  done_entry("Continue", changed=False)])
    d = p.decide(pg, [done_entry("Go", changed=True), done_entry("Next", changed=False),
                       done_entry("Continue", changed=False), done_entry("Submit", changed=False)])
    assert d["choice"] != "BLOCKED"


def test_planner_failure_is_remembered_per_url():
    fallback = Mock(return_value={"choice": "wait", "operation": "WAIT", "probabilities": {"wait": 1.0}})
    p, plan_fn = pilot([ValueError("no valid plan"), cont(CLICK_GO)], fallback=fallback)
    d1 = p.decide(page(), [])
    d2 = p.decide(page(), [])
    assert plan_fn.call_count == 1
    assert d1["route"] == "planner_fallback" and d2["route"] == "planner_fallback"
    d3 = p.decide(page(url=URL + "other"), [])
    assert plan_fn.call_count == 2 and d3["choice"] == "e3"


def test_unroutable_notes_are_not_duplicated():
    select = PlanStep("SELECT", "Country", "India", "Select India in Country.")
    fallback = Mock(return_value={"choice": "wait", "operation": "WAIT", "probabilities": {"wait": 1.0}})
    p, plan_fn = pilot([cont(select), cont(select), cont(CLICK_GO)], fallback=fallback)
    p.decide(page(), [])
    d = p.decide(page(), [])
    assert plan_fn.call_count == 3
    assert plan_fn.call_args.args[4].count("SELECT Country (no matching element)") == 1
    assert d["choice"] == "e3"


def test_failed_tool_step_is_excluded_and_replanned():
    scroll = PlanStep("SCROLL_TO_TEXT", "External links", "", "Scroll to the External links heading.")
    history = [done_entry("External links", op="SCROLL_TO_TEXT", changed=False, tool_ok=False)]
    p, plan_fn = pilot([cont(scroll), cont(CLICK_GO)])
    d = p.decide(page(), history)
    assert plan_fn.call_count == 2
    assert "SCROLL_TO_TEXT External links (no effect)" in plan_fn.call_args_list[0].args[4]
    assert "SCROLL_TO_TEXT External links (already tried)" in plan_fn.call_args_list[1].args[4]
    assert d["choice"] == "e3"


def test_focus_click_then_queued_type_step_needs_no_replan():
    click_search = PlanStep("CLICK", "Search", "", "Click Search.")
    type_search = PlanStep("TYPE_TEXT", "Search", "Ada Lovelace", 'Type "Ada Lovelace" into Search.')
    p, plan_fn = pilot([cont(click_search, type_search)])
    d1 = p.decide(page(), [])
    assert d1["choice"] == "e2"
    history = [done_entry("Open Search", op="CLICK", role="searchbox", changed=False)]
    d2 = p.decide(page(), history)
    assert (d2["choice"], d2["value"]) == ("e1", "Ada Lovelace")
    assert plan_fn.call_count == 1


def test_planner_failure_on_a_search_site_falls_back_to_a_search_goto_once():
    search_fn = Mock(return_value="Ada Lovelace")
    fallback = Mock(return_value={"choice": "e3", "operation": "CLICK", "probabilities": {"e3": 0.7}})
    p, plan_fn = pilot([ValueError("no valid plan")], fallback=fallback, search_query_fn=search_fn)
    wiki = page(url="https://en.wikipedia.org/wiki/Main_Page")
    d1 = p.decide(wiki, [])
    assert d1["choice"] == "TOOL" and d1["route"] == "search_fallback"
    assert tools.is_allowed_goto(d1["tool"]["arg"])
    search_fn.assert_called_once_with("Find Ada Lovelace")
    d2 = p.decide(wiki, [])
    assert d2["route"] == "planner_fallback" and d2["choice"] == "e3"
    search_fn.assert_called_once()  # a second planner failure does not search again
    assert plan_fn.call_count == 1


def test_search_fallback_with_no_query_falls_through_to_the_actor_only_fallback():
    search_fn = Mock(return_value=None)
    fallback = Mock(return_value={"choice": "e3", "operation": "CLICK", "probabilities": {"e3": 0.7}})
    p, _ = pilot([ValueError("no valid plan")], fallback=fallback, search_query_fn=search_fn)
    d = p.decide(page(url="https://en.wikipedia.org/wiki/Main_Page"), [])
    search_fn.assert_called_once_with("Find Ada Lovelace")
    assert d["route"] == "planner_fallback" and d["choice"] == "e3"


def test_planner_failure_without_a_search_template_uses_the_actor_only_fallback():
    fallback = Mock(return_value={"choice": "e3", "operation": "CLICK", "probabilities": {"e3": 0.7}})
    p, plan_fn = pilot([ValueError("no valid plan")], fallback=fallback)
    d = p.decide(page(), [])  # example.test has no registered search template
    assert d["route"] == "planner_fallback" and d["choice"] == "e3"


def test_search_query_is_asked_at_most_once_per_url_even_when_it_finds_nothing():
    search_fn = Mock(return_value=None)
    fallback = Mock(return_value={"choice": "e3", "operation": "CLICK", "probabilities": {"e3": 0.7}})
    p, plan_fn = pilot([ValueError("no valid plan")], fallback=fallback, search_query_fn=search_fn)
    wiki = page(url="https://en.wikipedia.org/wiki/Main_Page")
    p.decide(wiki, [])
    p.decide(wiki, [])
    search_fn.assert_called_once_with("Find Ada Lovelace")


def test_found_scroll_and_focus_click_count_as_completed_steps():
    scroll = PlanStep("SCROLL_TO_TEXT", "External links", "", "Scroll to the External links heading.")
    click_search = PlanStep("CLICK", "Search", "", "Click Search.")
    p, plan_fn = pilot([cont(scroll), cont(click_search), cont(CLICK_GO)])
    p.decide(page(), [])
    p.decide(page(), [done_entry("External links", op="SCROLL_TO_TEXT", changed=False, tool_ok=True)])
    history = [done_entry("External links", op="SCROLL_TO_TEXT", changed=False, tool_ok=True),
               done_entry("Open Search", op="CLICK", role="searchbox", changed=False)]
    p.decide(page(), history)
    assert plan_fn.call_args.args[3] == ["Scroll to the External links heading.", "Click Search."]


def test_any_goto_decision_marks_the_search_fallback_as_already_used():
    goto_step = PlanStep("GOTO", "https://en.wikipedia.org/w/index.php?search=Ada+Lovelace&title=Special%3ASearch&go=Go",
                          "", "Search the site for Ada Lovelace.")
    search_fn = Mock(side_effect=AssertionError("search must not be called"))
    fallback = Mock(return_value={"choice": "e3", "operation": "CLICK", "probabilities": {"e3": 0.7}})
    p, plan_fn = pilot([cont(goto_step), ValueError("no valid plan")], fallback=fallback, search_query_fn=search_fn)
    d1 = p.decide(page(url="https://en.wikipedia.org/wiki/Main_Page"), [])
    assert d1["tool"]["operation"] == "GOTO"
    d2 = p.decide(page(url="https://en.wikipedia.org/wiki/Ada_Lovelace"), [])
    assert d2["route"] == "planner_fallback" and d2["choice"] == "e3"
