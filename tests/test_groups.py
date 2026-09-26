from jev_ultrafast.groups import ordinal_pick, shape


def el(i, label, landmark="main", role="link"):
    return {"index": str(i), "label": label, "role": role, "landmark": landmark, "operations": ["CLICK"]}


HN = [el(1, "comments", landmark="nav"), el(2, "Story one"), el(3, "48 comments"), el(4, "Story two"),
      el(5, "1 comment"), el(6, "Story three"), el(7, "120 comments"), el(8, "More")]


def test_shape_masks_numbers_and_plurals():
    assert shape("48 comments") == shape("1 comment") == "# comment"
    assert shape("Comments") == "comment"


def test_picks_the_nth_member_of_the_matching_repeated_group():
    assert [ordinal_pick(HN, "comments", n)["index"] for n in (1, 2, 3)] == ["3", "5", "7"]
    assert ordinal_pick(HN, "comments link", 2)["index"] == "5"


def test_returns_none_without_a_large_enough_matching_group():
    assert ordinal_pick(HN, "comments", 4) is None
    assert ordinal_pick(HN, "reviews", 1) is None


def test_elements_that_cannot_be_clicked_are_ignored():
    fields = [{**el(i, f"{i} comments"), "operations": ["TYPE_TEXT"]} for i in range(3)]
    assert ordinal_pick(fields, "comments", 1) is None


def hn_rows():
    """Hacker News front page shape: nav links, then per story a title link, a user link, 'hide', and a
    comments link reading 'N comments', '1 comment' or 'discuss'."""
    nav = [el("n1", "Hacker News", landmark="nav"), el("n2", "new", landmark="nav"),
           el("n3", "comments", landmark="nav"), el("n4", "ask", landmark="nav")]
    rows = []
    for i, (title, user, comments) in enumerate([("Rust 2.0 released", "alice", "312 comments"),
                                                  ("Show HN: A tiny database", "bob", "discuss"),
                                                  ("Why cats purr", "carol", "1 comment"),
                                                  ("Solar sails in practice", "dave", "57 comments")], 1):
        rows += [el(f"t{i}", title), el(f"u{i}", user), el(f"h{i}", "hide"), el(f"c{i}", comments)]
    return nav + rows


def test_hacker_news_comments_by_story_position():
    rows = hn_rows()
    for text in ("comments", "discussion", "comments page"):
        assert [ordinal_pick(rows, text, n)["index"] for n in (1, 2, 3)] == ["c1", "c2", "c3"], text
    assert ordinal_pick(rows, "comments", 5) is None


def test_discuss_has_the_comment_shape():
    assert shape("discuss") == shape("48 comments")


def test_no_word_match_falls_back_to_the_group_of_the_anchor_element():
    rows = hn_rows()
    assert ordinal_pick(rows, "the thing under each story", 3, anchor=lambda: rows[-1])["index"] == "c3"
    assert ordinal_pick(rows, "the thing under each story", 3, anchor=lambda: None) is None
    assert ordinal_pick(rows, "the thing under each story", 9, anchor=lambda: rows[-1]) is None


def test_a_resembling_repeated_group_is_used_when_nothing_else_matches():
    rows = [el(1, "Commentary (4)"), el(2, "Story"), el(3, "Commentary (9)"), el(4, "Commentary (1)")]
    assert ordinal_pick(rows, "comment", 2)["index"] == "3"
