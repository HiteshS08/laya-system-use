from jev_ultrafast.planner import PlanStep
from scripts.bench_planner import first_step_ok


def el(index, label, **kw):
    return {"index": str(index), "label": label, "operations": ["CLICK"], **kw}


HN = [el(1, "comments", landmark="nav"), el(2, "Story A"), el(3, "48 comments"), el(4, "Story B"),
      el(5, "17 comments"), el(6, "Story C"), el(7, "discuss")]


def test_search_tasks_accept_goto_or_typing_the_name():
    ok = PlanStep("GOTO", "https://en.wikipedia.org/w/index.php?search=Ada+Lovelace&title=Special%3ASearch&go=Go",
                  "", "Go to the search results.")
    assert first_step_ok("wiki_search_ada", ok, None, [])
    typed = PlanStep("TYPE_TEXT", "Search Wikipedia", "Ada Lovelace", "Type it.")
    assert first_step_ok("wiki_search_ada", typed, None, [])
    assert not first_step_ok("wiki_search_ada", PlanStep("CLICK", "View history", "", "x"), None, [])


def test_hn_ordinal_comments_link():
    assert first_step_ok("hn_comments", PlanStep("CLICK", "48 comments", "", "x"), HN[2], HN)
    assert first_step_ok("hn_second_comments", PlanStep("CLICK", "17 comments", "", "x"), HN[4], HN)
    assert first_step_ok("hn_third_comments", PlanStep("CLICK", "discuss", "", "x"), HN[6], HN)
    assert not first_step_ok("hn_second_comments", PlanStep("CLICK", "comments", "", "x"), HN[0], HN)


def test_section_tasks_accept_toc_link_or_scroll_but_not_toggle():
    assert first_step_ok("ada_references", PlanStep("CLICK", "8 References", "", "x"), el(1, "8 References"), [])
    assert first_step_ok("ada_references", PlanStep("SCROLL_TO_TEXT", "References", "", "x"), None, [])
    toggle = el(2, "Toggle References subsection")
    assert not first_step_ok("ada_references", PlanStep("CLICK", toggle["label"], "", "x"), toggle, [])


def test_no_step_is_never_ok():
    assert not first_step_ok("gh_issues", None, None, [])
