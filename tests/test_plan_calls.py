from scripts.count_plan_calls import SCENARIOS, report, run


def test_scripted_scenarios_average_at_most_one_planner_call_per_distinct_url():
    results = [run(s) for s in SCENARIOS]
    assert [r.outcome for r in results] == ["DONE"] * len(SCENARIOS)
    assert {r.name: r.plan_calls for r in results} == {
        "wiki_search_ada": 1, "flights_form": 1, "turing_external_links": 2, "mallon_to_typhoid": 2,
        "gh_issues_retry": 1, "ada_references": 2}
    summary = report(results)
    assert summary["calls_per_url"] <= 1 and summary["mean_scenario_calls_per_url"] <= 1
    assert sum(r.pick_calls for r in results) == 0
