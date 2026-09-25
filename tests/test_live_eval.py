from types import SimpleNamespace

from scripts.live_eval import pilot_fields


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
