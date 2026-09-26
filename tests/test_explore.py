import random
from urllib.robotparser import RobotFileParser

import pytest

from scripts.explore import (
    USER_AGENT,
    action_like,
    crawl_delay,
    excluded,
    items_from_page,
    main,
    same_site_links,
    throwaway_cdp_url,
)


def act(i, label, kind="click", role="link", href=""):
    return {"id": f"e{i}", "kind": kind, "label": label, "role": role, "node": i, "value": "", "href": href}


PAGE = {"url": "https://docs.example/", "title": "Docs home", "actions": [
    act(1, "Getting started", href="https://docs.example/start"), act(2, "API reference", href="https://docs.example/api"),
    act(3, "Search docs", "fill", "searchbox"), act(4, "Open Search docs", "click", "searchbox"),
    act(5, "Download", href="https://other.example/dl"), act(6, "Download", href="https://docs.example/dl"),
    {"id": "wait", "kind": "wait", "label": "Wait"}]}


def test_suite_sites_are_excluded():
    assert excluded("https://en.wikipedia.org/wiki/X") and excluded("https://de.wikipedia.org/wiki/X")
    assert excluded("https://news.ycombinator.com/") and not excluded("https://docs.example/")


def test_items_have_serving_shape_and_a_gold_among_the_options():
    items = items_from_page(PAGE, random.Random(0), n=4)
    assert 1 <= len(items) <= 4
    for item in items:
        (name, question), = item["questions"].items()
        assert name in {"click_target", "type_text_target"}
        assert item["gold_id"] in question["criteria"]
        assert item["gold"][name]["probabilities"][item["gold_id"]] == 1.0
        assert item["source"] == "explore" and item["website"] == "docs.example" and item["drop_reason"] is None


def test_duplicate_labels_are_never_gold():
    items = items_from_page(PAGE, random.Random(0), n=50)
    golds = {item["questions"][next(iter(item["questions"]))]["criteria"][item["gold_id"]] for item in items}
    assert not any(g.startswith("Download") for g in golds)


def test_same_site_links_skip_other_hosts_and_fragments():
    assert same_site_links(PAGE) == ["https://docs.example/start", "https://docs.example/api",
                                     "https://docs.example/dl"]


@pytest.mark.parametrize("url", [
    "https://docs.example/logout", "https://docs.example/account/sign-out?next=/",
    "https://docs.example/signout", "https://docs.example/list/unsubscribe/42",
    "https://docs.example/post/7/delete", "https://docs.example/index.php?action=edit&title=X",
    "https://docs.example/wiki/X/edit", "https://docs.example/page?sid=abc",
    "https://docs.example/page?csrfToken=abc", "https://docs.example/page?api_key=1",
])
def test_action_like_urls_are_skipped(url):
    assert action_like(url)
    page = {"url": "https://docs.example/", "actions": [act(1, "x", href=url)]}
    assert same_site_links(page) == []


@pytest.mark.parametrize("url", ["https://docs.example/start", "https://docs.example/editorial/today",
                                 "https://docs.example/search?q=delegates", "https://docs.example/?page=2"])
def test_plain_urls_are_kept(url):
    assert not action_like(url)


def robots(text):
    parser = RobotFileParser()
    parser.parse(text.splitlines())
    return {"https://docs.example": parser}


def test_crawl_delay_is_at_least_one_second_and_honours_robots():
    assert crawl_delay("https://docs.example/a", robots("User-agent: *\nDisallow:")) == 1.0
    assert crawl_delay("https://docs.example/a", robots("User-agent: *\nCrawl-delay: 5")) == 5.0
    assert crawl_delay("https://docs.example/a", robots("User-agent: *\nCrawl-delay: 0")) == 1.0


def test_the_robots_user_agent_is_the_one_chrome_sends():
    assert USER_AGENT.startswith("laya-browser-explorer")


def test_explore_refuses_to_run_without_a_throwaway_chrome(monkeypatch):
    monkeypatch.delenv("LAYA_EXPLORE_CDP_URL", raising=False)
    monkeypatch.setenv("BU_CDP_URL", "http://127.0.0.1:9444")
    with pytest.raises(SystemExit):
        main(["--seeds", "https://docs.example/"])
    for bad in (None, "", "http://127.0.0.1:9222", "http://localhost:9223", "http://127.0.0.1:9444", "file:///x"):
        with pytest.raises(ValueError):
            throwaway_cdp_url(bad)
    assert throwaway_cdp_url("http://127.0.0.1:9333") == "http://127.0.0.1:9333"
