import pytest

from jev_ultrafast.program import (
    Program,
    Subgoal,
    drop_redundant_opens,
    fallback_program,
    parse_program,
    render_program,
)


def test_parses_kinds_values_ordinals_and_done_text():
    p = parse_program("FIND Alan Turing\nOPEN comments @2\nFILL Where to? = London\nSUBMIT\nDONE_WHEN Search results")
    assert p.subgoals == (Subgoal("FIND", "Alan Turing"), Subgoal("OPEN", "comments", ordinal=2),
                          Subgoal("FILL", "Where to?", "London"), Subgoal("SUBMIT", ""))
    assert p.done_text == "Search results" and p.source == "compiler"


def test_tolerates_numbering_bullets_fences_and_think_blocks():
    raw = "<think>plan</think>```\n1. FIND Ada Lovelace\n- JUMP References\n```"
    assert [s.kind for s in parse_program(raw).subgoals] == ["FIND", "JUMP"]


@pytest.mark.parametrize("raw", ["", "GO somewhere", "FILL Name", "FIND", "OPEN x @0", "DO anything",
                                 "\n".join(["FIND a"] * 9), "FIND " + "x" * 81, "DONE_WHEN only"])
def test_rejects_invalid_programs(raw):
    with pytest.raises(ValueError):
        parse_program(raw)


def test_render_round_trips():
    p = parse_program("FIND Ada\nOPEN comments @3\nSELECT Size = M\nSUBMIT Search\nDONE_WHEN Added")
    assert parse_program(render_program(p)) == p


def test_fallback_is_a_single_do_subgoal():
    assert fallback_program("Buy milk") == Program((Subgoal("DO", "Buy milk"),), source="fallback")


def test_an_open_of_what_find_just_reached_is_dropped():
    program = drop_redundant_opens(parse_program("FIND Ada Lovelace\nOPEN ada lovelace.\nCLICK Ada Lovelace"))
    assert [(s.kind, s.target) for s in program.subgoals] == [("FIND", "Ada Lovelace")]


def test_other_steps_after_find_are_kept():
    text = "FIND Ada Lovelace\nOPEN Talk\nFIND Charles Babbage\nOPEN History @2"
    program = drop_redundant_opens(parse_program(text))
    assert render_program(program) == text
    twice = parse_program("OPEN Ada\nOPEN Ada")
    assert drop_redundant_opens(twice) == twice
