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


def test_opensearch_with_another_required_placeholder_is_ignored():
    osd = OSD.replace("{startPage?}", "{language}")
    assert template_from_opensearch(osd, WIKI) is None


def test_learn_template_from_an_observed_search():
    assert learn_template("https://ex.test/find?q=ada+lovelace&lang=en", "Ada Lovelace") == \
        "https://ex.test/find?q={q}&lang=en"
    assert learn_template("https://ex.test/wiki/Ada_Lovelace", "Ada Lovelace") is None


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


def test_discover_reads_the_pages_own_description():
    assert discover(Mock(evaluate=Mock(return_value=OSD)), WIKI) == T
    assert discover(Mock(evaluate=Mock(return_value=None)), WIKI) is None
    assert discover(Mock(evaluate=Mock(side_effect=RuntimeError("gone"))), WIKI) is None
