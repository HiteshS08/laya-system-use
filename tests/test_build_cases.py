import json

import pytest
from m2w_fixtures import make_step, make_task

from training import build_cases as bc
from training import fetch_data


def test_dev_website_is_deterministic_and_disabled_at_zero():
    assert bc.is_dev_website("site", 0) is False
    assert bc.is_dev_website("site", 20) == bc.is_dev_website("site", 20)


def test_build_writes_jsonl_and_summary(tmp_path):
    shard = tmp_path / "s.json"
    shard.write_text(json.dumps([make_task(make_step("CLICK", "10"), make_step("TYPE", "20", "x"))]))
    summary = bc.build([shard], tmp_path / "out", "train", k=20, dev_mod=0, limit_tasks=0)
    rows = [json.loads(line) for line in (tmp_path / "out" / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and summary["train"]["steps"] == 2
    assert (tmp_path / "out" / "train_summary.json").exists()


def test_limit_tasks_stops_after_n_tasks(tmp_path):
    shard = tmp_path / "s.json"
    shard.write_text(json.dumps([make_task(make_step()), make_task(make_step(), make_step("TYPE", "20", "x"))]))
    summary = bc.build([shard], tmp_path / "out", "train", k=20, dev_mod=0, limit_tasks=1)
    rows = (tmp_path / "out" / "train.jsonl").read_text().splitlines()
    assert len(rows) == 1 and summary["train"]["steps"] == 1


def test_dev_split_goes_to_its_own_file(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "is_dev_website", lambda website, mod: True)
    shard = tmp_path / "s.json"
    shard.write_text(json.dumps([make_task(make_step())]))
    bc.build([shard], tmp_path / "out", "train", k=20, dev_mod=20, limit_tasks=0)
    assert (tmp_path / "out" / "train_dev.jsonl").read_text().strip()
    assert (tmp_path / "out" / "train.jsonl").read_text() == ""


def test_fetch_test_extracts_and_finds_splits(tmp_path, monkeypatch):
    import zipfile

    import huggingface_hub

    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in fetch_data.TEST_SPLITS:
            zf.writestr(f"data/{name}/{name}_0.json", "[]")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(zip_path))
    found = fetch_data.fetch_test(tmp_path / "dest")
    assert set(found) == set(fetch_data.TEST_SPLITS) and all(found.values())


def test_fetch_test_reports_missing_splits(tmp_path, monkeypatch):
    import zipfile

    import huggingface_hub

    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "x")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(zip_path))
    with pytest.raises(FileNotFoundError, match="layout changed"):
        fetch_data.fetch_test(tmp_path / "dest")
