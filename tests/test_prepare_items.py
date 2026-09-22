import json
import re

import pytest
from m2w_fixtures import make_step, make_task

from training import prepare_items as pi
from training.mind2web import task_rows


@pytest.fixture(scope="module")
def model_dir():
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            "convaiinnovations/laya", allow_patterns=["tokenizer/*", "rl_agent_config.json"], local_files_only=True
        )
    except (OSError, ValueError):
        pytest.skip("Laya tokenizer is not in the local Hugging Face cache")


@pytest.fixture(scope="module")
def assets(model_dir):
    return pi.load_tokenizer_and_cfg(model_dir, pi.MAX_LEN, pi.HEAD_MAX_LEN)


def rows():
    return list(task_rows(make_task(make_step("CLICK", "10"), make_step("TYPE", "20", "abc"))))


def test_items_follow_laya_format(assets):
    tok, cfg = assets
    items, stats = pi.build_items(tok, cfg, rows())
    assert stats["items"] == 3  # operation + click_target for step 1, operation for step 2
    for item in items:
        assert len(item["markers"]) == len(item["target"]) and abs(sum(item["target"]) - 1) < 1e-6
        assert item["target"][item["label"]] == 1.0
    assert items[1]["label"] == 0  # click_target: gold "10" is the first option


def test_rows_with_a_drop_reason_are_skipped(assets):
    tok, cfg = assets
    dropped = list(task_rows(make_task(make_step("CLICK", "40"))))
    items, stats = pi.build_items(tok, cfg, dropped)
    assert items == [] and stats["rows_skipped"] == 1


def test_items_that_overflow_the_budget_are_counted_not_kept(assets):
    tok, cfg = assets
    items, stats = pi.build_items(tok, {**cfg, "max_len": 8}, rows())
    assert items == [] and stats["items_dropped_overflow"] == 3


def write_cases(path, case_rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in case_rows))
    return path


def test_main_writes_items_and_matching_meta(assets, model_dir, tmp_path, monkeypatch):
    import huggingface_hub
    import torch

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *_a, **_k: model_dir)
    cases = write_cases(tmp_path / "cases.jsonl", rows())
    out = tmp_path / "train_items.pt"
    pi.main(["--cases", str(cases), "--out", str(out)])
    items = torch.load(out, weights_only=False)
    meta = json.loads((tmp_path / "train_items.meta.json").read_text())
    assert len(items) == meta["n_items"] == 3
    assert meta["max_len"] == 768 and meta["head_max_len"] == 448
    assert meta["stats"]["items"] == 3


def test_main_exits_when_no_items_can_be_built(assets, model_dir, tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *_a, **_k: model_dir)
    cases = write_cases(tmp_path / "dropped.jsonl", task_rows(make_task(make_step("CLICK", "40"))))
    out = tmp_path / "items.pt"
    with pytest.raises(SystemExit, match=re.escape(str(cases))):
        pi.main(["--cases", str(cases), "--out", str(out)])
    assert not out.exists()
