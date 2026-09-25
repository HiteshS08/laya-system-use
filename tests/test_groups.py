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
