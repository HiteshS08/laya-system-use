from unittest.mock import Mock

from jev_ultrafast.controller import Controller
from jev_ultrafast.program import Program, Subgoal, fallback_program, parse_program
from jev_ultrafast.search import SearchTemplates

HOME = "https://w.test/wiki/Main_Page"
ADA = "https://w.test/wiki/Ada_Lovelace"
T = "https://w.test/w/index.php?search={q}"


def act(i, label, kind="click", role="link", href=""):
    return {"id": f"e{i}", "kind": kind, "label": label, "role": role, "node": i, "value": "", "href": href}


BASE = [act(1, "Search Wikipedia", "fill", "searchbox"), act(2, "Open Search Wikipedia", "click", "searchbox"),
        act(3, "Issues 5"), act(4, "Code"), act(5, "Charles Babbage", href="https://w.test/wiki/Charles_Babbage")]


def page(url=HOME, title="Main Page", actions=BASE, text="", headings=()):
    return {"url": url, "title": title, "text": text, "headings": list(headings), "actions": list(actions)}


def entry(label, op="CLICK", before=HOME, after=HOME, changed=True, kind="click", role="link", **extra):
    return {"operation": op, "kind": kind, "action": label, "role": role, "url_before": before, "url": after,
            "page_changed": changed, **extra}


def ranked(*preferred):
    """A fake actor: full distribution over the offered ids, highest for the earliest preferred id offered."""
    def predict(state, questions):
        name, question = next(iter(questions.items()))
        ids = list(question["criteria"])
        order = [i for i in preferred if i in ids] + [i for i in ids if i not in preferred]
        weights = {i: 2.0 ** -rank for rank, i in enumerate(order)}
        total = sum(weights.values())
        probs = {i: w / total for i, w in weights.items()}
        return {"answers": {name: {"choice": order[0], "confidence": probs[order[0]], "probabilities": probs}}}
    return Mock(side_effect=predict)


def controller(text, **kw):
    kw.setdefault("predict", Mock(side_effect=AssertionError("actor not expected")))
    kw.setdefault("fallback", Mock(side_effect=AssertionError("fallback not expected")))
    kw.setdefault("discover", Mock(return_value=None))
    kw.setdefault("templates", SearchTemplates())
    program = text if isinstance(text, Program) else parse_program(text)
    return Controller("goal", program, **kw)


def test_find_via_discovered_template_then_done_when_the_page_is_about_it():
    c = controller("FIND Ada Lovelace", discover=Mock(return_value=T))
    d = c.decide(page(), [])
    assert (d["choice"], d["tool"]["operation"], d["route"]) == ("TOOL", "GOTO", "tactic")
    assert d["tool"]["templates"] == [T]
    history = [entry(d["tool"]["arg"], op="GOTO", kind="tool", after=ADA, tool_ok=True)]
    d = c.decide(page(url=ADA, title="Ada Lovelace - Wikipedia"), history)
    assert d["choice"] == "DONE" and "Ada Lovelace" in d["evidence"]
    assert c.actor_calls == 0 and len(c.plans) == 1


def test_find_via_search_box_learns_the_template():
    templates = SearchTemplates()
    c = controller("FIND Ada Lovelace", templates=templates, predict=ranked("5"))
    d1 = c.decide(page(), [])
    assert (d1["choice"], d1["value"]) == ("e1", "Ada Lovelace")
    h = [entry("Search Wikipedia", op="TYPE_TEXT", kind="fill", role="searchbox", changed=False, text="Ada Lovelace")]
    d2 = c.decide(page(), h)
    assert d2["tool"]["operation"] == "SUBMIT"
    results = "https://w.test/w/index.php?search=Ada+Lovelace&title=Special:Search"
    h.append(entry("", op="SUBMIT", kind="tool", after=results, tool_ok=True))
    d3 = c.decide(page(url=results, title="Search results"), h)
    assert templates.get(HOME) == "https://w.test/w/index.php?search={q}&title=Special:Search"
    assert d3["route"] == "actor" and d3["instruction"] == "Click the search result for Ada Lovelace."


def test_actor_grounds_descriptions_and_a_no_effect_element_is_excluded():
    predict = ranked("3", "4")
    c = controller("OPEN Issues tab", predict=predict)
    d1 = c.decide(page(), [])
    assert (d1["choice"], d1["route"]) == ("e3", "actor")
    assert predict.call_args.args[0]["goal"] == "Click the Issues tab."
    d2 = c.decide(page(), [entry("Issues 5", changed=False)])
    assert d2["choice"] == "e4" and c.actor_calls == 2
    assert "3" not in predict.call_args.args[1]["click_target"]["criteria"]


def test_ordinal_route_needs_no_model():
    rows = [act(1, "comments"), act(2, "Story one"), act(3, "48 comments"), act(4, "Story two"),
            act(5, "3 comments")]
    d = controller("OPEN comments @2").decide(page(actions=rows), [])
    assert (d["choice"], d["route"]) == ("e5", "ordinal")


def test_fill_uses_the_program_value_then_picks_the_suggestion_then_moves_on():
    field = [act(1, "Where to?", "fill", "combobox"), act(2, "Search", role="button")]
    c = controller("FILL Where to? = London\nCLICK Search")
    d1 = c.decide(page(actions=field), [])
    assert (d1["choice"], d1["operation"], d1["value"], d1["route"]) == ("e1", "TYPE_TEXT", "London", "resolver")
    h = [entry("Where to?", op="TYPE_TEXT", kind="fill", role="combobox", changed=True, text="London")]
    with_options = field + [act(3, "London, United Kingdom", role="option")]
    d2 = c.decide(page(actions=with_options), h)
    assert (d2["choice"], d2["route"]) == ("e3", "tactic")
    h.append(entry("London, United Kingdom", role="option"))
    d3 = c.decide(page(actions=field), h)
    assert d3["choice"] == "e2"
    h.append(entry("Search", role="button", after="https://w.test/results"))
    assert c.decide(page(url="https://w.test/results", actions=field), h)["choice"] == "DONE"


def test_native_select_maps_the_value_to_an_option():
    select = [{"id": "e1", "kind": "select", "label": "Size → Small", "role": "combobox", "node": 1, "value": "s"},
              {"id": "e2", "kind": "select", "label": "Size → Medium", "role": "combobox", "node": 1, "value": "m"}]
    d = controller("SELECT Size = Medium").decide(page(actions=select), [])
    assert (d["choice"], d["operation"]) == ("e2", "SELECT")


def test_blocked_after_two_misses():
    c = controller("OPEN Issues tab", predict=ranked("3", "4"))
    c.decide(page(), [])
    c.decide(page(), [entry("Issues 5", changed=False)])
    d = c.decide(page(), [entry("Issues 5", changed=False), entry("Code", changed=False)])
    assert d["choice"] == "BLOCKED" and "Issues tab" in d["evidence"]


def test_a_step_with_no_element_blocks():
    d = controller("FILL Where to? = London").decide(page(actions=[act(1, "Home")]), [])
    assert d["choice"] == "BLOCKED"


def test_done_text_and_an_already_satisfied_goal_stop_at_once():
    c = controller(Program((Subgoal("OPEN", "Issues"),), done_text="3 open issues"))
    assert c.decide(page(text="There are 3 open issues"), [])["choice"] == "DONE"
    c = controller("FIND Ada Lovelace")
    assert c.decide(page(url=ADA, title="Ada Lovelace - Wikipedia"), [])["choice"] == "DONE"


def test_scroll_completes_once_the_tool_found_the_text():
    c = controller("SCROLL External links")
    d = c.decide(page(), [])
    assert d["tool"] == {"operation": "SCROLL_TO_TEXT", "arg": "External links", "templates": []}
    h = [entry("External links", op="SCROLL_TO_TEXT", kind="tool", changed=True, tool_ok=True)]
    assert c.decide(page(), h)["choice"] == "DONE"


def test_unexecuted_decision_is_not_counted_as_progress():
    c = controller("FIND Ada Lovelace")
    first = c.decide(page(), [])
    again = c.decide(page(), [])  # the Agent raised StalePage and asked again with the same history
    assert first["choice"] == again["choice"] == "e1"


def test_do_fallback_delegates_to_the_goal_mode_policy():
    fallback = Mock(return_value={"choice": "e4", "operation": "CLICK", "probabilities": {"e4": 0.8}})
    c = controller(fallback_program("Buy milk"), fallback=fallback)
    d = c.decide(page(), [])
    assert (d["choice"], d["route"]) == ("e4", "goal_fallback") and fallback.call_args.args[1] == "goal"


def test_from_goal_compiles_once_and_records_it():
    compile_fn = Mock(return_value=(parse_program("FIND Ada"), {"source": "compiler", "latency_ms": 7,
                                                                "attempts": 1}))
    c = Controller.from_goal("Find Ada", compile_fn=compile_fn, predict=Mock(), fallback=Mock(),
                             discover=Mock(return_value=None), templates=SearchTemplates())
    assert compile_fn.call_count == 1 and c.plans[0]["latency_ms"] == 7 and c.plans[0]["program"] == "FIND Ada"


def test_decisions_have_the_shape_the_agent_needs():
    c = controller("FIND Ada Lovelace")
    d = c.decide(page(), [])
    for key in ("choice", "operation", "target", "confidence", "probabilities", "latency_ms", "usage", "route",
                "instruction", "value", "tool", "evidence", "actor_ms"):
        assert key in d
    assert d["probabilities"][d["choice"]] == 1.0
    assert c.trace[-1]["kind"] == "FIND" and c.trace[-1]["route"] == "tactic"
