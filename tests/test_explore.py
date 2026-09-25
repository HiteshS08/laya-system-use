import random

from scripts.explore import excluded, items_from_page, same_site_links


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
