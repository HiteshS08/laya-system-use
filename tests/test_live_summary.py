from scripts.live_summary import summarize


def rec(task, category, success, steps, seconds, decisions, planner_calls=()):
    return {"task": task, "category": category, "success": success, "steps": [{}] * steps, "seconds": seconds,
            "decisions": decisions, "planner_calls": list(planner_calls)}


def test_summary_counts_categories_routes_and_medians():
    records = [
        rec("a", "site_search", True, 2, 10.0,
            [{"route": "resolver", "latency_ms": 900, "actor_ms": 0},
             {"route": "actor", "latency_ms": 1200, "actor_ms": 800}],
            [{"latency_ms": 2000}, {"latency_ms": 2500}]),
        rec("b", "site_search", False, 4, 20.0, [{"route": "planner_pick", "latency_ms": 3000, "actor_ms": 700}],
            [{"error": "no valid plan"}]),
    ]
    s = summarize(records)
    assert (s["passed"], s["n"]) == (1, 2)
    assert s["by_category"] == {"site_search": [1, 2]}
    assert s["routes"] == {"resolver": 1, "actor": 1, "planner_pick": 1}
    assert s["median_decision_ms"] == 1200 and s["median_actor_ms"] == 750
    assert s["median_planner_ms"] == 2250 and s["wall_s_per_action"] == 5.0


def test_summary_reports_model_calls_and_timings():
    records = [
        {**rec("a", "x", True, 2, 6.0, []), "llm_calls": 1, "actor_calls": 1,
         "steps": [{"observe_ms": 300, "act_ms": 100}, {"observe_ms": 500, "act_ms": 300}]},
        {**rec("b", "x", False, 1, 2.0, []), "llm_calls": 0, "actor_calls": 3,
         "steps": [{"observe_ms": 400, "act_ms": 200}]},
    ]
    s = summarize(records)
    assert s["llm_calls_per_task"] == 0.5 and s["actor_calls_per_task"] == 2.0
    assert s["median_observe_ms"] == 400 and s["median_act_ms"] == 200
    assert s["median_wall_s_per_action"] == 2.5


def test_llm_calls_fall_back_to_planner_call_records():
    records = [rec("a", "x", True, 1, 1.0, [], [{"latency_ms": 1}, {"latency_ms": 2}])]
    assert summarize(records)["llm_calls_per_task"] == 2.0
