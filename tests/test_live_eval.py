import json
from types import SimpleNamespace

from scripts.live_eval import _jsonable, pilot_fields, write_record


def test_jsonable_turns_sets_and_frozensets_into_sorted_lists():
    assert _jsonable({3, 1, 2}) == [1, 2, 3]
    assert _jsonable(frozenset({"b", "a"})) == ["a", "b"]


def test_write_record_serialises_frozenset_and_set_valued_trace_fields(tmp_path):
    record = {"task": "t", "trace": [{"step": {"roles": frozenset({"link", "button"})}}], "tags": {"y", "x"}}
    write_record(record, tmp_path, "t")
    written = json.loads((tmp_path / "t.json").read_text())
    assert written["trace"][0]["step"]["roles"] == ["button", "link"]
    assert written["tags"] == ["x", "y"]


def test_write_record_survives_an_unserialisable_field_and_logs_a_minimal_record(tmp_path):
    class Unserialisable:
        pass

    record = {"task": "bad", "status": "done", "unserialisable": Unserialisable()}
    write_record(record, tmp_path, "bad")
    written = json.loads((tmp_path / "bad.json").read_text())
    assert written["task"] == "bad"
    assert written["status"] == "error"
    assert "not JSON serializable" in written["error"]


def test_pilot_fields_for_the_program_backend():
    pilot = SimpleNamespace(plans=[{"program": "FIND Ada", "latency_ms": 5, "attempts": 1, "source": "compiler"}],
                            trace=[{"route": "resolver"}], actor_calls=2)
    fields = pilot_fields(pilot)
    assert fields == {"planner_calls": pilot.plans, "program": "FIND Ada", "trace": pilot.trace,
                      "actor_calls": 2, "llm_calls": 1}


def test_cached_programs_cost_no_llm_call():
    pilot = SimpleNamespace(plans=[{"program": "FIND Ada", "attempts": 0, "source": "cache"}], trace=[],
                            actor_calls=0)
    assert pilot_fields(pilot)["llm_calls"] == 0


def test_pilot_fields_for_the_planner_backend_and_none():
    assert pilot_fields(SimpleNamespace(plans=[{"latency_ms": 1}, {"error": "x"}]))["llm_calls"] == 2
    assert pilot_fields(None) == {"planner_calls": [], "llm_calls": 0}
