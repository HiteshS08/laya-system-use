from jev_ultrafast.pruning import is_detour, prune_actions

WIKI = "https://en.wikipedia.org"


def link(i, label, href, node=None):
    return {"id": f"e{i}", "kind": "click", "label": label, "role": "link", "node": node or i, "href": href}


def test_editor_and_history_links_are_detours():
    assert is_detour(f"{WIKI}/w/index.php?title=Main_Page&action=edit", "Open today's featured article.")
    assert is_detour(f"{WIKI}/w/index.php?title=Main_Page&action=history", "Open the article")
    assert is_detour("https://github.com/a/b/edit/main/README.md", "Open the Issues tab")
    assert not is_detour(f"{WIKI}/wiki/Mary_Mallon", "Open today's featured article.")
    assert not is_detour("", "anything")


def test_goal_that_asks_for_editing_keeps_detours():
    assert not is_detour(f"{WIKI}/w/index.php?title=X&action=edit", "View the source of this page")
    assert not is_detour(f"{WIKI}/w/index.php?title=X&action=history", "Show the revision history")


def test_detour_links_are_removed_and_other_actions_kept():
    actions = [link(1, "View source", f"{WIKI}/w/index.php?title=Main_Page&action=edit"),
               link(2, "Mary Mallon", f"{WIKI}/wiki/Mary_Mallon"),
               {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560}]
    kept = prune_actions(actions, "Open today's featured article.")
    assert [a["id"] for a in kept] == ["e2", "scroll_down"]


def test_links_to_the_same_url_merge_into_the_first_with_aliases():
    actions = [link(1, "Mary Mallon", f"{WIKI}/wiki/Mary_Mallon"),
               link(2, "Full article...", f"{WIKI}/wiki/Mary_Mallon"),
               link(3, "Mary Mallon", f"{WIKI}/wiki/Mary_Mallon")]
    kept = prune_actions(actions, "Open today's featured article.")
    assert [a["id"] for a in kept] == ["e1"]
    assert kept[0]["aliases"] == ["Full article..."]


def test_fragments_are_distinct_targets_and_placeholder_hrefs_never_merge():
    actions = [link(1, "8 References", f"{WIKI}/wiki/Ada#References"),
               link(2, "10 External links", f"{WIKI}/wiki/Ada#External_links"),
               link(3, "Menu", f"{WIKI}/wiki/Ada#"), link(4, "More", f"{WIKI}/wiki/Ada#"),
               link(5, "Run", "javascript:void(0)"), link(6, "Stop", "javascript:void(0)")]
    assert [a["id"] for a in prune_actions(actions, "Jump to References")] == ["e1", "e2", "e3", "e4", "e5", "e6"]


def test_input_actions_are_not_mutated():
    actions = [link(1, "A", f"{WIKI}/wiki/A"), link(2, "B", f"{WIKI}/wiki/A")]
    prune_actions(actions, "goal")
    assert "aliases" not in actions[0]
