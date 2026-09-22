from m2w_fixtures import make_step, make_task

from training import evaluate as ev
from training.mind2web import task_rows


def rows(k=20, goal=None):
    task = make_task(make_step("CLICK", "10"), make_step("TYPE", "20", "abc"))
    if goal:
        task["confirmed_task"] = goal
    return list(task_rows(task, k))


def perfect(row):
    return {"operation": row["gold_op"], "targets": {row["gold_op"]: row["gold_id"]}}


def test_perfect_predictor_scores_one():
    r = ev.evaluate_rows(rows(), perfect)
    assert r["op_acc"] == r["element_acc_given_op"] == r["step_success_scored"] == r["step_success_overall"] == 1.0


def test_ranker_baseline_gets_click_steps_only():
    # Goal avoids the word "Find": the shortlister's textbox prior would otherwise outrank the link for that goal.
    r = ev.evaluate_rows(rows(goal="NFL scores"), ev.ranker_predictor())
    assert r["op_acc"] == 0.5 and r["element_acc_given_op"] == 1.0 and r["step_success_scored"] == 0.5


def test_gold_dropped_by_shortlist_counts_as_overall_miss():
    r = ev.evaluate_rows(list(task_rows(make_task(make_step("CLICK", "30")), k=1)), perfect)
    assert r["scored"] == 0 and r["step_success_overall"] == 0.0 and r["step_success_scored"] is None


def test_predictor_errors_are_counted_and_scored_wrong():
    def boom(row):
        raise ValueError("options exceed head_max_len")

    r = ev.evaluate_rows(rows(), boom)
    assert r["errors"] == 2 and r["step_success_scored"] == 0.0


class FakeAgent:
    def system_one(self, state, questions):
        if not state["recent_actions"]:  # step 1
            return {"answers": {"operation": {"choice": "CLICK"}, "click_target": {"choice": "10"}}}
        return {"answers": {"operation": {"choice": "TYPE_TEXT"}, "click_target": {"choice": "30"}}}


def test_laya_predictor_uses_sole_candidate_when_no_target_question():
    r = ev.evaluate_rows(rows(), ev.laya_predictor(FakeAgent()))
    assert r["step_success_scored"] == 1.0
