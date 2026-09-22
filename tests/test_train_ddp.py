import json
import math
import random

from training.train_ddp import fit_one_temp, fit_temperatures, write_config

CHOICE = 0
BUCKET_KEYS = {2: "choice:2", 4: "choice:3-5", 8: "choice:6-10", 15: "choice:11+"}


def overconfident_preds(k: int, n: int, qtype: int = CHOICE, seed: int = 0):
    """Peaked logits (margin 6) whose argmax is wrong a third of the time."""
    rng = random.Random(seed)
    preds = []
    for i in range(n):
        pick = rng.randrange(k)
        logits = [rng.gauss(0, 0.1) for _ in range(k)]
        logits[pick] += 6.0
        gold = pick if i % 3 else (pick + 1) % k
        preds.append((qtype, logits, [1.0 if j == gold else 0.0 for j in range(k)]))
    return preds


def test_overconfident_bucket_gets_temperature_above_one():
    _, by_bucket = fit_temperatures(overconfident_preds(15, 60))
    assert by_bucket["choice:11+"] > 1.0


def test_bucket_with_fewer_than_ten_samples_is_omitted():
    _, by_bucket = fit_temperatures(overconfident_preds(15, 30) + overconfident_preds(4, 9))
    assert "choice:11+" in by_bucket and "choice:3-5" not in by_bucket


def test_buckets_use_the_keys_laya_reads_at_inference():
    from laya.common import temp_bucket

    preds = [p for k in BUCKET_KEYS for p in overconfident_preds(k, 20)]
    _, by_bucket = fit_temperatures(preds)
    assert set(by_bucket) == set(BUCKET_KEYS.values())
    assert all(temp_bucket(CHOICE, k) == key for k, key in BUCKET_KEYS.items())


def test_only_fitted_buckets_are_returned_and_globals_fall_back():
    temps, by_bucket = fit_temperatures(overconfident_preds(15, 30) + overconfident_preds(8, 5))
    assert set(by_bucket) == {"choice:11+"}  # nothing carried over from the base config, no unfitted bucket
    assert temps[CHOICE] > 1.0 and temps[1] == 1.2 and temps[2] == 1.2


def test_non_finite_fit_falls_back_to_one():
    # torch.clamp(nan) is nan, json would emit a bare NaN and laya would turn it into T=1e-3
    for bad in (float("nan"), float("inf"), float("-inf")):
        sel = [([bad, 0.0, 0.0], [1.0, 0.0, 0.0])] + [([0.5, 0.1, 0.0], [1.0, 0.0, 0.0])] * 12
        assert fit_one_temp(sel) == 1.0


def test_finite_fit_is_unchanged_by_the_guard():
    temp = fit_one_temp([(z, t) for _, z, t in overconfident_preds(15, 60)])
    assert math.isfinite(temp) and temp > 1.0


BASE_CFG = {"max_len": 768, "head_max_len": 448, "temperature_by_options": {"choice:11+": 0.1006}}


def read_config(output_dir):
    return json.loads((output_dir / "rl_agent_config.json").read_text())


def test_write_config_records_calibration_and_budgets(tmp_path):
    write_config(BASE_CFG, tmp_path, [1.2, 1.2, 1.2], {"choice:2": 1.5})
    written = read_config(tmp_path)
    assert written["temperature"] == [1.2, 1.2, 1.2]
    assert written["temperature_by_options"] == {"choice:2": 1.5}
    assert written["fine_tuned"] is True and written["model_name"] == "laya-browser-mind2web"
    assert written["max_len"] == 768 and written["head_max_len"] == 448


def test_write_config_twice_keeps_only_the_fitted_values(tmp_path):
    write_config(BASE_CFG, tmp_path, [1.2, 1.2, 1.2], {"choice:2": 1.5})  # placeholder write
    write_config(BASE_CFG, tmp_path, [0.9, 1.2, 1.2], {"choice:11+": 2.5})  # fitted write
    written = read_config(tmp_path)
    assert written["temperature"] == [0.9, 1.2, 1.2]
    assert written["temperature_by_options"] == {"choice:11+": 2.5}  # neither "choice:2" nor the base 0.1006 survive


def test_write_config_does_not_mutate_the_input_config(tmp_path):
    before = json.loads(json.dumps(BASE_CFG))
    write_config(BASE_CFG, tmp_path, [1.0, 1.0, 1.0], {})
    assert BASE_CFG == before
