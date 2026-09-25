from unittest.mock import Mock

from jev_ultrafast.planner import PlanStep
from jev_ultrafast.router import route


def el(index, label, ops=("CLICK",)):
    return {"index": str(index), "label": label, "role": "link", "operations": list(ops), "value": ""}


ELEMENTS = [el(1, "comments"), el(2, "48 comments"), el(3, "17 comments"), el(4, "Search", ("TYPE_TEXT",))]


def predictor(choice, probs):
    return lambda state, questions: {"answers": {next(iter(questions)): {
        "choice": choice, "confidence": probs[choice], "probabilities": probs}}}


def never(*_):
    raise AssertionError("must not be called")


def test_unique_label_resolves_without_models():
    r = route(PlanStep("CLICK", "48 comments", "", "Click 48 comments."), ELEMENTS, [], "g",
              predict=never, pick=never)
    assert (r.index, r.route, r.confidence) == ("2", "resolver", 1.0)


def test_operation_filters_candidates():
    r = route(PlanStep("TYPE_TEXT", "Search", "Ada", "Type Ada into Search."), ELEMENTS, [], "g",
              predict=never, pick=never)
    assert (r.index, r.route) == ("4", "resolver")
    none = route(PlanStep("SELECT", "Country", "India", "Select India."), ELEMENTS, [], "g",
                 predict=never, pick=never)
    assert (none.index, none.route) == (None, "no_candidates")


def test_confident_actor_is_accepted():
    s = PlanStep("CLICK", "comments of the second story", "", "Click the comments link of the second story.")
    r = route(s, ELEMENTS, [], "g", predict=predictor("3", {"1": 0.1, "2": 0.1, "3": 0.8}), pick=never)
    assert (r.index, r.route, r.confidence) == ("3", "actor", 0.8)


def test_unsure_actor_hands_top_options_to_the_planner(monkeypatch):
    monkeypatch.setenv("ACTOR_TAU", "0.6")
    s = PlanStep("CLICK", "comments of the second story", "", "Click the comments link of the second story.")
    pick = Mock(return_value="3")
    r = route(s, ELEMENTS, [], "Open the second story's comments", predict=predictor(
        "1", {"1": 0.5, "2": 0.2, "3": 0.3}), pick=pick)
    assert (r.index, r.route) == ("3", "planner_pick")
    options = pick.call_args.args[1]
    assert [i for i, _ in options] == ["1", "3", "2"] and options[0][1].startswith("1 | comments")


def test_empty_target_text_routes_to_the_sole_field_without_a_model_call():
    field = el(4, "Search", ("TYPE_TEXT",))
    r = route(PlanStep("TYPE_TEXT", "", "Ada Lovelace", "Type Ada Lovelace into Search."), [field], [], "g",
              predict=never, pick=never)
    assert (r.index, r.confidence) == ("4", 1.0)


def test_planner_rejecting_all_options_gives_no_index(monkeypatch):
    monkeypatch.setenv("ACTOR_TAU", "0.9")
    s = PlanStep("CLICK", "comments of a story", "", "Click the comments link of a story.")
    r = route(s, ELEMENTS, [], "g", predict=predictor("1", {"1": 0.4, "2": 0.3, "3": 0.3}),
              pick=Mock(return_value=None))
    assert (r.index, r.route) == (None, "planner_none")
