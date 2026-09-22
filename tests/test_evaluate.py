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


def test_element_acc_overall_matches_step_success_overall_denominator():
    # Mirrors MindAct's Element Accuracy: correct / all steps with a resolvable gold (drop-reason-free set).
    r = ev.evaluate_rows(rows(), perfect)
    assert r["element_acc_overall"] == 1.0
    r2 = ev.evaluate_rows(list(task_rows(make_task(make_step("CLICK", "30")), k=1)), perfect)
    # shortlist missed the gold -> counts as wrong, same denominator as step_success_overall
    assert r2["element_acc_overall"] == 0.0


def test_trivial_steps_count_as_correct_for_macro_and_success_rate():
    # One candidate and nothing else: no question to ask. The deployed policy always gets it right (policy.py).
    step = make_step("CLICK", "10")
    step["neg_candidates"] = []
    trivial_rows = list(task_rows(make_task(step), k=20))
    assert trivial_rows[0]["drop_reason"] == "trivial"
    r = ev.evaluate_rows(trivial_rows, perfect)
    assert r["element_acc_macro"] == r["op_acc_macro"] == r["step_success_macro"] == r["success_rate"] == 1.0


def test_trivial_steps_are_forced_correct_even_if_the_predictor_gets_them_wrong():
    # jev_ultrafast/policy.py never calls the model for a single candidate, so a trivial step can't actually fail.
    step = make_step("CLICK", "10")
    step["neg_candidates"] = []
    trivial_rows = list(task_rows(make_task(step), k=20))

    def wrong(row):
        return {"operation": "TYPE_TEXT", "targets": {}}

    r = ev.evaluate_rows(trivial_rows, wrong)
    assert r["success_rate"] == 1.0


def test_hard_drop_reasons_fail_macro_and_success_rate_even_with_a_perfect_predictor():
    # gold_not_interactive (no reclaim target exists): the deployed agent has no way to reach this element at all.
    hard_rows = list(task_rows(make_task(make_step("CLICK", "40")), k=20))
    assert hard_rows[0]["drop_reason"] == "gold_not_interactive"
    r = ev.evaluate_rows(hard_rows, perfect)
    assert r["element_acc_macro"] == r["op_acc_macro"] == r["step_success_macro"] == r["success_rate"] == 0.0


def test_success_rate_requires_every_step_in_the_task_to_succeed():
    # Step 1 succeeds (perfect), step 2 is a hard failure -> the whole task fails, matching MindAct's task SR.
    task = make_task(make_step("CLICK", "10"), make_step("CLICK", "40"))
    r = ev.evaluate_rows(list(task_rows(task, k=20)), perfect)
    assert r["success_rate"] == 0.0
    assert 0.0 < r["step_success_macro"] < 1.0  # one of the two steps still succeeded


def test_macro_metrics_average_per_task_not_per_step():
    # Task A: 1 step, correct. Task B: 2 steps, both correct. Macro = mean(1.0, 1.0) = 1.0 regardless of step counts.
    task_a = make_task(make_step("CLICK", "10"))
    task_a["annotation_id"] = "A"
    task_b = make_task(make_step("CLICK", "10"), make_step("TYPE", "20", "x"))
    task_b["annotation_id"] = "B"
    rows_ = [*task_rows(task_a, k=20), *task_rows(task_b, k=20)]
    r = ev.evaluate_rows(rows_, perfect)
    assert r["element_acc_macro"] == r["success_rate"] == 1.0


def test_predictor_error_fails_macro_and_success_rate_for_that_task():
    def boom(row):
        raise ValueError("options exceed head_max_len")

    r = ev.evaluate_rows(rows(), boom)
    assert r["success_rate"] == 0.0 and r["step_success_macro"] == 0.0
