import json
from unittest.mock import Mock

import pytest

from jev_ultrafast import planner
from jev_ultrafast.planner import PlanStep

PAGE = {"url": "https://en.wikipedia.org/wiki/Main_Page", "title": "Wikipedia, the free encyclopedia",
        "text": "From today's featured article\nMary Mallon was an Irish-born cook.", "outline": "Main Page"}


def el(index, label, ops=("CLICK",), **extra):
    return {"index": str(index), "label": label, "role": "link", "operations": list(ops), "value": "", **extra}


ELEMENTS = [el(1, "Search Wikipedia", ("TYPE_TEXT", "CLICK"), role="searchbox", landmark="header"),
            el(2, "Mary Mallon", landmark="main", section="From today's featured article", in_viewport=True),
            el(3, "Official archive", landmark="main", section="External links", row_text="Archive Official archive",
               in_viewport=False)]


def step(**kw):
    base = {"operation": "CLICK", "target_text": "Mary Mallon", "value": "",
            "instruction": "Click the Mary Mallon link in today's featured article."}
    return {**base, **kw}


def test_element_line_shows_context_and_offscreen():
    assert planner.element_line(ELEMENTS[2]) == (
        "3 | Official archive | link | CLICK | main > External links | row: Archive Official archive | offscreen")


def test_view_limits_elements_and_offers_the_search_template():
    view = planner.planner_view("Open the Mary Mallon article", PAGE, ELEMENTS, ["a"], ["CLICK Read (no effect)"])
    assert view["search_url_template"].startswith("https://en.wikipedia.org/w/index.php?search={q}")
    assert len(view["elements"]) == 3 and view["failed_attempts"] == ["CLICK Read (no effect)"]
    many = [el(i, f"Link {i}") for i in range(1, 101)]
    assert len(planner.planner_view("goal", PAGE, many, [], [])["elements"]) == planner.MAX_ELEMENTS
    assert "search_url_template" not in planner.planner_view("g", {**PAGE, "url": "https://x.test/"}, [], [], [])


def test_parse_valid_continue_plan():
    p = planner.parse_plan({"status": "continue", "evidence": "", "steps": [step()]}, PAGE)
    assert p.status == "continue" and p.steps == (PlanStep("CLICK", "Mary Mallon", "",
                                                            "Click the Mary Mallon link in today's featured article."),)


def test_done_needs_a_quote_the_page_shows_regardless_of_case_accents_and_spacing():
    ok = planner.parse_plan({"status": "done", "evidence": "mary  MALLON was an irish-born cook", "steps": []}, PAGE)
    assert ok.status == "done"
    accented = {**PAGE, "text": "Kurt Gödel was a logician."}
    assert planner.parse_plan({"status": "done", "evidence": "Kurt Godel was", "steps": []}, accented).status == "done"
    with pytest.raises(ValueError, match="evidence"):
        planner.parse_plan({"status": "done", "evidence": "Open the Mary Mallon article", "steps": []}, PAGE)


@pytest.mark.parametrize("bad", [
    {"status": "maybe", "evidence": "", "steps": []},
    {"status": "continue", "evidence": "", "steps": []},
    {"status": "continue", "evidence": "", "steps": [step(operation="HOVER")]},
    {"status": "continue", "evidence": "", "steps": [step(target_text="")]},
    {"status": "continue", "evidence": "", "steps": [step(operation="TYPE_TEXT", value="")]},
    {"status": "continue", "evidence": "", "steps": [step(operation="TYPE_TEXT", value="false")]},
    {"status": "continue", "evidence": "", "steps": [step(operation="GOTO", target_text="https://evil.test/")]},
    {"status": "continue", "evidence": "", "steps": [step(instruction=7)]},
])
def test_invalid_plans_are_rejected(bad):
    with pytest.raises(ValueError):
        planner.parse_plan(bad, PAGE)


def test_long_instruction_is_cut_and_steps_capped_at_three():
    long = " ".join(["word"] * 30)
    p = planner.parse_plan({"status": "continue", "evidence": "", "steps": [step(instruction=long)] * 5}, PAGE)
    assert len(p.steps) == 3 and len(p.steps[0].instruction.split()) == planner.INSTRUCTION_WORDS


def test_plan_retries_once_on_an_invalid_plan_then_succeeds():
    outputs = [({"status": "continue", "evidence": "", "steps": []}, {}),
               ({"status": "continue", "evidence": "", "steps": [step()]}, {})]
    complete = Mock(side_effect=outputs)
    p = planner.plan("Open today's featured article.", PAGE, ELEMENTS, [], [], complete=complete)
    assert p.steps[0].target_text == "Mary Mallon" and complete.call_count == 2
    assert p.request_chars > 0
    assert complete.call_args.kwargs["extra"] == planner.DISABLE_THINKING
    payload = json.loads(complete.call_args_list[0].args[1])
    assert payload["goal"] == "Open today's featured article."


def test_plan_sends_the_validation_error_as_correction_on_retry():
    outputs = [({"status": "continue", "evidence": "", "steps": []}, {}),
               ({"status": "continue", "evidence": "", "steps": [step()]}, {})]
    complete = Mock(side_effect=outputs)
    p = planner.plan("Open today's featured article.", PAGE, ELEMENTS, [], [], complete=complete)
    assert p.steps[0].target_text == "Mary Mallon"
    first_message = complete.call_args_list[0].args[1]
    second_message = complete.call_args_list[1].args[1]
    assert second_message.startswith(first_message)
    assert "Planner said continue but gave no steps" in second_message


def test_plan_gives_up_after_two_invalid_plans():
    complete = Mock(return_value=({"status": "continue", "evidence": "", "steps": []}, {}))
    with pytest.raises(ValueError, match="no valid plan"):
        planner.plan("g", PAGE, ELEMENTS, [], [], complete=complete)


def test_request_fits_the_prompt_budget_on_a_large_page():
    many = [el(i, "L" * 70, landmark="main", section="S" * 60, row_text="R" * 90) for i in range(1, 121)]
    big = {**PAGE, "text": "x" * 6000, "outline": " | ".join(["Heading"] * 200)}
    payload = json.dumps(planner.planner_view("goal " * 20, big, many, ["done step"] * 12, ["failed"] * 12),
                         ensure_ascii=False, separators=(",", ":"))
    # ~4 characters per token: system prompt + payload must stay within ~1,200 tokens.
    assert len(planner.PLANNER_SYSTEM) + len(payload) <= 4800


def test_plan_records_prompt_tokens_and_sends_compact_json():
    complete = Mock(return_value=({"status": "continue", "evidence": "", "steps": [step()]},
                                  {"usage": {"prompt_tokens": 812}}))
    p = planner.plan("Open today's featured article.", PAGE, ELEMENTS, [], [], complete=complete)
    assert p.prompt_tokens == 812
    sent = complete.call_args.args[1]
    assert ", " not in sent[:40] and '": ' not in sent
    assert complete.call_args.kwargs["max_tokens"] == 256


def test_pick_returns_an_offered_id_or_none():
    s = PlanStep("CLICK", "comments", "", "Click the comments link of the second story.")
    options = [("5", "5 | comments | link | CLICK | nav"), ("16", "16 | 17 comments | link | CLICK | main")]
    assert planner.pick(s, options, "g", complete=Mock(return_value=({"option": "16"}, {}))) == "16"
    assert planner.pick(s, options, "g", complete=Mock(return_value=({"option": None}, {}))) is None
    assert planner.pick(s, options, "g", complete=Mock(return_value=({"option": "99"}, {}))) is None
