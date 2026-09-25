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
