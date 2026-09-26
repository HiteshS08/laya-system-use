from unittest.mock import Mock

from jev_ultrafast.search import (
    SearchTemplates,
    discover,
    learn_template,
    matches_template,
    render,
    template_from_opensearch,
)

OSD = """<?xml version="1.0"?><OpenSearchDescription xmlns="http://a9.com/-/spec/opensearch/1.1/">
<Url type="application/x-suggestions+json" template="https://en.wikipedia.org/w/api.php?search={searchTerms}"/>
<Url type="text/html" method="get"
 template="https://en.wikipedia.org/w/index.php?title=Special:Search&amp;search={searchTerms}&amp;page={startPage?}"/>
</OpenSearchDescription>"""
WIKI = "https://en.wikipedia.org/wiki/Main_Page"
T = "https://en.wikipedia.org/w/index.php?title=Special:Search&search={q}"


def test_opensearch_html_template_for_the_same_host():
    assert template_from_opensearch(OSD, WIKI) == T
    assert template_from_opensearch(OSD, "https://evil.test/") is None
    assert template_from_opensearch("not xml", WIKI) is None
    assert template_from_opensearch(OSD.replace('type="text/html"', 'type="x"'), WIKI) is None


def test_opensearch_needs_an_http_template_on_the_page_host():
    for bad in ("javascript:alert(1)//{searchTerms}", "data:text/html,{searchTerms}", "file:///x?q={searchTerms}"):
        osd = OSD.replace("https://en.wikipedia.org/w/index.php?title=Special:Search&amp;search={searchTerms}"
                          "&amp;page={startPage?}", bad)
        assert template_from_opensearch(osd, "about:blank") is None, bad
        assert template_from_opensearch(osd, WIKI) is None, bad
    assert template_from_opensearch(OSD, "file:///wiki/Main_Page") is None


def test_opensearch_with_another_required_placeholder_is_ignored():
    osd = OSD.replace("{startPage?}", "{language}")
    assert template_from_opensearch(osd, WIKI) is None


EX = "https://ex.test/"


def test_learn_template_from_an_observed_search():
    assert learn_template("https://ex.test/find?q=ada+lovelace&lang=en", "Ada Lovelace", EX) == \
        "https://ex.test/find?q={q}&lang=en"
    assert learn_template("https://ex.test/wiki/Ada_Lovelace", "Ada Lovelace", EX) is None


def test_learn_template_refuses_a_landing_on_another_host():
    assert learn_template("https://victim.test/find?q=ada", "ada", EX) is None
    assert learn_template("https://ex.test.evil.test/find?q=ada", "ada", EX) is None


def test_learn_template_refuses_token_like_parameters():
    for name in ("sid", "SessionId", "token", "csrf_token", "auth", "api_key", "sig"):
        assert learn_template(f"https://ex.test/find?q=ada&{name}=abc", "ada", EX) is None, name
    long_value = "a1B2" * 10
    assert learn_template(f"https://ex.test/find?q=ada&v={long_value}", "ada", EX) is None


def test_learn_template_needs_http_and_a_host():
    assert learn_template("javascript:x?q=ada", "ada", "javascript:y") is None
    assert learn_template("file:///find?q=ada", "ada", "file:///") is None
    assert learn_template("https://u:p@ex.test/find?q=ada", "ada", EX) is None


def test_learn_template_drops_tracking_parameters():
    assert learn_template("https://ex.test/find?q=ada&utm_source=x&fbclid=y", "ada", EX) == \
        "https://ex.test/find?q={q}"


def test_render_and_match_are_host_bound():
    url = render(T, "Ada Lovelace")
    assert url == "https://en.wikipedia.org/w/index.php?title=Special:Search&search=Ada+Lovelace"
    assert matches_template(url, T)
    assert not matches_template("https://evil.test/w/index.php?title=Special:Search&search=x", T)
    assert not matches_template(url + "&x=1#frag", T)


def test_store_persists_by_host(tmp_path):
    SearchTemplates(tmp_path / "t.json").put(WIKI, T)
    assert SearchTemplates(tmp_path / "t.json").get("https://en.wikipedia.org/wiki/Other") == T
    assert SearchTemplates().get(WIKI) is None


OSD_HREF = "https://en.wikipedia.org/w/rest.php/v1/search"


def test_discover_reads_the_pages_own_description():
    browser = Mock(evaluate=Mock(return_value=OSD))
    assert discover(browser, {"url": WIKI, "opensearch": OSD_HREF}) == T
    assert OSD_HREF in browser.evaluate.call_args.args[0]
    assert discover(Mock(evaluate=Mock(return_value=None)), {"url": WIKI, "opensearch": OSD_HREF}) is None
    gone = Mock(evaluate=Mock(side_effect=RuntimeError("gone")))
    assert discover(gone, {"url": WIKI, "opensearch": OSD_HREF}) is None


def test_discover_skips_the_page_query_without_a_same_host_description_link():
    for href in ("", "https://evil.test/osd.xml", "javascript:alert(1)", None):
        browser = Mock(evaluate=Mock(return_value=OSD))
        assert discover(browser, {"url": WIKI, "opensearch": href}) is None
        assert not browser.evaluate.called


def test_template_store_writes_are_atomic(tmp_path, monkeypatch):
    import os

    store = SearchTemplates(tmp_path / "t.json")
    store.put(WIKI, T)
    before = (tmp_path / "t.json").read_text()
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    try:
        store.put("https://ex.test/", "https://ex.test/?q={q}")
    except OSError:
        pass
    assert (tmp_path / "t.json").read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == ["t.json"]
