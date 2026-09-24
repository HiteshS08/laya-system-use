from jev_ultrafast.candidates import Candidate
from jev_ultrafast.formatter import build_request, context_text, render_history_item, render_option


def test_context_text_joins_landmark_section_and_row_with_limits():
    assert context_text("main", "From today's featured article") == "main > From today's featured article"
    assert context_text("nav") == "nav"
    assert context_text("", "", "") == ""
    long = context_text("main", "S" * 100, "R" * 100)
    assert long == "main > " + "S" * 40 + " · row: " + "R" * 60


def test_render_option_appends_context_only_when_asked():
    c = Candidate("1", "Mary Mallon", "link", context="main > From today's featured article")
    assert render_option(c) == "Mary Mallon (link)"
    assert render_option(c, with_context=True) == "Mary Mallon (link) [main > From today's featured article]"
    assert render_option(Candidate("2", "Go", "button"), with_context=True) == "Go (button)"


def cand(i, label="Go", role="button", value="", ops=("CLICK",)):
    return Candidate(str(i), label, role, value, frozenset(ops))


def test_render_option_puts_label_first_with_role_and_value():
    assert render_option(cand(1, "Search", "textbox", "foo")) == "Search (textbox, =foo)"


def test_render_option_truncates_and_collapses_whitespace():
    text = render_option(cand(1, "  a\n b  " + "x" * 200))
    assert text.startswith("a b xxx") and len(text) < 100


def test_render_option_falls_back_to_role_when_unnamed():
    assert render_option(cand(1, "", "button")) == "button (button)"


def test_render_history_item():
    assert render_history_item("CLICK", "NFL") == "CLICK NFL"
    assert render_history_item("TYPE_TEXT", "Search", "abc") == "TYPE_TEXT Search = abc"


def test_build_request_asks_only_multi_option_questions():
    by_op = {"CLICK": [cand(1), cand(2)], "TYPE_TEXT": [cand(3, ops=("TYPE_TEXT",))], "SELECT": []}
    state, questions = build_request("find nfl", [], by_op)
    assert list(questions) == ["operation", "click_target"]
    assert list(questions["operation"]["criteria"]) == ["CLICK", "TYPE_TEXT"]
    assert questions["click_target"]["criteria"] == {"1": "Go (button)", "2": "Go (button)"}
    assert state == {"goal": "find nfl", "recent_actions": []}


def test_build_request_single_operation_has_no_operation_question():
    _, questions = build_request("g", [], {"CLICK": [cand(1), cand(2)]})
    assert list(questions) == ["click_target"]


def test_build_request_keeps_last_three_actions_and_does_not_mutate_input():
    history = ["a", "b", "c", "d"]
    state, _ = build_request("g", history, {})
    assert state["recent_actions"] == ["b", "c", "d"]
    assert history == ["a", "b", "c", "d"]


def test_target_instruction_names_the_operation():
    by_op = {"CLICK": [cand(1), cand(2)], "TYPE_TEXT": [cand(3, ops=("TYPE_TEXT",)), cand(4, ops=("TYPE_TEXT",))]}
    _, questions = build_request("g", [], by_op)
    assert "clicked" in questions["click_target"]["instructions"]
    assert "typed into" in questions["type_text_target"]["instructions"]
