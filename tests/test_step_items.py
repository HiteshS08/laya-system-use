import json
import random

import pytest

from jev_ultrafast.instructions import instruction
from tests.m2w_fixtures import make_step, make_task
from training.mind2web import parse_step
from training.step_items import describe, main, step_row


def test_describe_drops_at_most_one_word():
    rng = random.Random(0)
    outs = {describe("Find flights now", rng, drop_p=1.0) for _ in range(20)}
    assert all(len(o.split()) == 2 for o in outs) and describe("Go", rng, drop_p=1.0) == "Go"


def test_step_row_uses_the_serving_instruction_and_one_target_question():
    task = make_task(make_step("CLICK", gold="30"))
    row = step_row(task, 0, parse_step(task["actions"][0]), [], random.Random(1), goal_mode_p=0.0)
    assert row["state"]["goal"] == instruction("OPEN", "Search")
    assert list(row["questions"]) == ["click_target"]
    gold = row["gold"]["click_target"]["probabilities"]
    assert gold[row["gold_id"]] == 1.0 and row["mode"] == "step" and row["drop_reason"] is None


def test_goal_mode_rows_keep_the_task_goal():
    task = make_task(make_step("CLICK", gold="10"))
    row = step_row(task, 0, parse_step(task["actions"][0]), [], random.Random(1), goal_mode_p=1.0)
    assert row["state"]["goal"] == "Find NFL scores" and row["mode"] == "goal"


def test_unusable_steps_are_skipped():
    task = make_task(make_step("CLICK", gold="40"))  # a div, not interactive
    assert step_row(task, 0, parse_step(task["actions"][0]), [], random.Random(1)) is None
    lone_field = make_task(make_step("TYPE", gold="20", value="nfl"))  # one candidate: serving never asks the model
    assert step_row(lone_field, 0, parse_step(lone_field["actions"][0]), [], random.Random(1)) is None


def test_cli_writes_rows_and_refuses_test_splits(tmp_path):
    shard = tmp_path / "train_0.json"
    shard.write_text(json.dumps([make_task(make_step("CLICK", gold="10"), make_step("TYPE", gold="20", value="x"))]))
    out = tmp_path / "step.jsonl"
    main(["--input", str(shard), "--out", str(out), "--goal-mode-p", "0"])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["mode"] for r in rows] == ["step"]
    with pytest.raises(SystemExit):
        main(["--input", str(tmp_path / "test_task_1.json"), "--out", str(out)])
    with pytest.raises(SystemExit):
        main(["--input", str(tmp_path / "test" / "x.json"), "--out", str(out)])


def test_cli_without_a_dev_split_opens_no_extra_file(tmp_path, monkeypatch):
    import builtins

    shard = tmp_path / "train_0.json"
    shard.write_text(json.dumps([make_task(make_step("CLICK", gold="10"))]))
    real_open = builtins.open

    def guarded(file, *args, **kwargs):
        assert str(file) != "/dev/null", "no /dev/null (not portable)"
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", guarded)
    main(["--input", str(shard), "--out", str(tmp_path / "rows.jsonl"), "--goal-mode-p", "0"])
    assert (tmp_path / "rows.jsonl").read_text()
