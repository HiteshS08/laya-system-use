import json

from scripts.compare_runs import compare, load_run, majority, mcnemar_exact, outcomes


def test_outcomes_and_majority():
    assert outcomes([{"task": "a", "success": True}, {"task": "b", "success": False}]) == {"a": True, "b": False}
    runs = [{"a": True, "b": False}, {"a": True, "b": True}, {"a": False, "b": False}]
    assert majority(runs) == {"a": True, "b": False}


def test_mcnemar_exact_two_sided():
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(0, 6) - 0.03125) < 1e-9
    assert abs(mcnemar_exact(1, 7) - 0.0703125) < 1e-9


def test_compare_counts_discordant_pairs_on_shared_tasks():
    a = {"t1": True, "t2": False, "t3": True, "t4": False}
    b = {"t1": True, "t2": True, "t3": False, "t4": True, "t5": True}
    c = compare(a, b)
    assert (c["n"], c["a_passed"], c["b_passed"], c["only_a"], c["only_b"]) == (4, 2, 3, 1, 2)
    assert c["p_value"] == mcnemar_exact(1, 2)


def test_load_run_reads_task_records(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"task": "a", "success": True}))
    (tmp_path / "b.json").write_text(json.dumps({"task": "b", "success": False}))
    assert load_run(tmp_path) == {"a": True, "b": False}
