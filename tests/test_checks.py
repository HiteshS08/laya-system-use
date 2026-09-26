from jev_ultrafast.checks import fragment_names, heading_in_view, is_about, satisfied, text_shown, title_subject
from jev_ultrafast.program import Subgoal

URL = "https://w.test/wiki/Alan_Turing"


def page(**kw):
    base = {"url": URL, "title": "Alan Turing - Wikipedia", "text": "Alan Turing\nEarly life",
            "headings": [{"text": "Alan Turing", "level": 1, "id": "firstHeading", "in_viewport": True},
                         {"text": "References", "level": 2, "id": "References", "in_viewport": False}]}
    return {**base, **kw}


def test_title_subject():
    assert title_subject("Alan Turing - Wikipedia") == "Alan Turing"
    assert title_subject("Issues · browser-use/browser-use") == "Issues"
    assert title_subject("Plain") == "Plain"


def test_is_about_uses_title_or_h1_and_ignores_case_accents_and_parentheticals():
    assert is_about(page(), "alan turing")
    python = page(title="Python (programming language) - Wikipedia", headings=[])
    assert is_about(python, "Python programming language") and is_about(python, "Python")
    assert is_about(page(title="Kurt Gödel - Wikipedia", headings=[]), "Kurt Godel")
    assert is_about(page(title="Search results", headings=[{"text": "Ada Lovelace", "level": 1,
                                                            "id": "", "in_viewport": True}]), "Ada Lovelace")
    assert not is_about(page(), "Alan Turing Institute")
    assert not is_about(page(title="Alan Turing Institute - Wikipedia", headings=[]), "Alan Turing")


def test_fragment_names_section():
    assert fragment_names(URL + "#External_links", "External links")
    assert fragment_names(URL + "#References", "references")
    assert not fragment_names(URL, "References")
    assert not fragment_names(URL + "#cite_note-5", "References")


def test_heading_in_view_and_text_shown():
    assert heading_in_view(page(), "Alan Turing") and not heading_in_view(page(), "References")
    assert text_shown(page(), "early LIFE") and text_shown(page(), "wikipedia") and not text_shown(page(), "")


def test_satisfied_by_kind():
    assert satisfied(Subgoal("FIND", "Alan Turing"), page(), "https://w.test/") is True
    assert satisfied(Subgoal("JUMP", "References"), page(url=URL + "#References"), URL) is True
    assert satisfied(Subgoal("JUMP", "References"), page(), URL) is False
    assert satisfied(Subgoal("SCROLL", "Alan Turing"), page(), URL) is True
    assert satisfied(Subgoal("OPEN", "Issues"), page(url="https://g.test/r/issues"), "https://g.test/r") is True
    assert satisfied(Subgoal("OPEN", "Issues"), page(url=URL + "#x"), URL) is False
    assert satisfied(Subgoal("FILL", "Where to?", "London"), page(), URL) is None


def test_pages_without_headings_still_work():
    assert not heading_in_view({"url": URL, "title": "", "text": ""}, "References")
    assert is_about({"url": URL, "title": "Alan Turing - Wikipedia", "text": ""}, "Alan Turing")


def test_heading_with_an_inline_edit_link_still_matches():
    edit = page(headings=[{"text": "References[edit]", "level": 2, "id": "References", "in_viewport": True}])
    assert heading_in_view(edit, "References")
