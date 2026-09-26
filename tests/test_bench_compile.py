from evals.heldout_goals import HELDOUT
from evals.live_tasks import TASKS
from scripts.bench_compile import bench


def test_bench_counts_valid_and_fallback_programs():
    outputs = iter(["FIND Ada", "nonsense", "still nonsense"])
    s = bench(["Find Ada", "Do x"], lambda system, user, **kw: (next(outputs), {"latency_ms": 10,
                                                                             "usage": {"completion_tokens": 4}}))
    assert (s["n"], s["valid"], s["fallback"]) == (2, 1, 1) and s["programs"]["Find Ada"] == "FIND Ada"
    assert s["median_completion_tokens"] == 6.0 and s["programs"]["Do x"] == "DO Do x"


def test_heldout_goals_are_new_sites():
    suite_hosts = {t.url.split("/")[2] for t in TASKS.values()}
    assert len(HELDOUT) == 12 and not {u.split("/")[2] for u, _ in HELDOUT} & suite_hosts
    assert len({g for _, g in HELDOUT}) == 12
