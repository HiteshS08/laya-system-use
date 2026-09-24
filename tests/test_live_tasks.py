"""Outcome checks for the live suite; no browser or model calls."""

import base64
from collections import Counter

from evals.live_tasks import TASKS, check_outcome
from scripts import live_eval


def test_fixed_url_tasks_require_the_requested_destination():
    task = TASKS["gh_issues"]
    assert check_outcome(task, "https://github.com/browser-use/browser-use/issues", "Issues")
    assert not check_outcome(task, "https://github.com/browser-use/browser-use", "Issues 146")
    assert check_outcome(TASKS["wiki_search_mallon"],
                         "https://en.wikipedia.org/wiki/Mary_Mallon#Biography", "Mary Mallon")
    assert not check_outcome(TASKS["turing_references"],
                             "https://en.wikipedia.org/wiki/Alan_Turing#External_links", "References")


def test_dynamic_link_must_match_the_starting_page_target():
    task = TASKS["hn_comments"]
    expected = "https://news.ycombinator.com/item?id=123"
    assert check_outcome(task, "https://news.ycombinator.com/item?id=123", "comments", expected)
    assert not check_outcome(task, "https://news.ycombinator.com/item?id=456", "comments", expected)
    assert not check_outcome(task, expected, "comments")


def test_flights_needs_observed_fields_matching_results_and_encoded_date():
    task = TASKS["flights_mumbai_delhi"]
    encoded = base64.urlsafe_b64encode(b"departing 2026-12-02").decode().rstrip("=")
    url = f"https://www.google.com/travel/flights/search?tfs={encoded}"
    actions = [
        {"label": "Change ticket type. One way", "value": "One way"},
        {"label": "1 passenger, change number of passengers.", "value": ""},
        {"label": "Change seating class. Economy", "value": "Economy"},
        {"label": "Where from?", "value": "Mumbai"},
        {"label": "Where to?", "value": "New Delhi"},
        {"label": "Departure", "value": "Wed, Dec 2"},
        {"label": "From 6810 Indian rupees. Leaves Mumbai on Wednesday, December 2. Select flight"},
    ]
    assert check_outcome(task, url, "Search results\n62 results returned.", final_actions=actions)
    assert not check_outcome(task, url, "Search results\n62 results returned.")
    assert not check_outcome(task, url, "Search results", final_actions=actions[:-1])
    assert not check_outcome(task, url.replace(encoded, ""), "Search results", final_actions=actions)
    wrong_destination = [a | {"value": "London"} if a["label"] == "Where to?" else a for a in actions]
    assert not check_outcome(task, url, "Search results", final_actions=wrong_destination)


def test_suite_has_the_planned_size_and_category_coverage():
    categories = Counter(task.category for task in TASKS.values())
    assert len(TASKS) == 25
    for name in ("site_search", "in_page_navigation", "disambiguation", "multi_field_form",
                 "below_fold", "multi_page_flow"):
        assert categories[name] >= 3


def test_multi_page_and_below_fold_checks_require_the_actual_path():
    task = TASKS["turing_to_award"]
    award = "https://en.wikipedia.org/wiki/Turing_Award"
    turing = "https://en.wikipedia.org/wiki/Alan_Turing"
    assert not check_outcome(task, award, "Turing Award")
    assert check_outcome(task, award, "Turing Award", visited_urls=(task.url, turing, award))

    task = TASKS["turing_external_links"]
    assert not check_outcome(task, task.url, "External links", scroll_y=0)
    assert check_outcome(task, task.url, "External links", scroll_y=1500)


def test_live_runner_records_independent_success_and_urls(monkeypatch, tmp_path):
    class FakeAgent:
        def __init__(self, url, goal):
            self.state = {
                "page": {"url": url, "title": "Repository", "text": "Code", "scroll": {"y": 0}},
                "status": "ready", "history": [], "decisions": [],
            }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def command(self, name):
            assert name == "tick"
            page = self.state["page"]
            final_url = page["url"] + "/issues"
            self.state["page"] = {**page, "url": final_url, "title": "Issues", "text": "Issues"}
            self.state["history"] = [{"operation": "CLICK", "action": "Issues", "latency_ms": 5,
                                      "page_changed": True, "url": final_url}]
            self.state["decisions"] = [{"choice": "e1", "operation": "CLICK", "confidence": 0.9,
                                        "target_confidence": 0.8, "latency_ms": 5}]
            self.state["status"] = "done"

    monkeypatch.setattr(live_eval, "Agent", FakeAgent)
    record = live_eval.run("gh_issues", TASKS["gh_issues"], tmp_path)
    assert record["success"] is True
    assert record["steps"][0]["url_before"] == TASKS["gh_issues"].url
    assert record["steps"][0]["url_after"].endswith("/issues")
    assert record["decisions"][0]["target_confidence"] == 0.8
    assert (tmp_path / "gh_issues.json").exists()
