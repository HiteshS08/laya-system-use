from unittest.mock import Mock

import pytest

from jev_ultrafast import tools
from jev_ultrafast.browser import Browser

WIKI_SEARCH = "https://en.wikipedia.org/w/index.php?search=Ada+Lovelace&title=Special%3ASearch&go=Go"


def test_search_template_by_host():
    assert tools.search_template("https://en.wikipedia.org/wiki/Main_Page").startswith("https://en.wikipedia.org/")
    assert tools.search_template("https://news.ycombinator.com/") is None


def test_goto_allows_only_rendered_registry_templates():
    assert tools.is_allowed_goto(WIKI_SEARCH)
    assert tools.is_allowed_goto("https://github.com/search?q=browser-use&type=repositories")
    assert not tools.is_allowed_goto("https://en.wikipedia.org/wiki/Ada_Lovelace")
    assert not tools.is_allowed_goto("https://evil.test/w/index.php?search=x&title=Special%3ASearch&go=Go")
    assert not tools.is_allowed_goto(WIKI_SEARCH + "&extra=1")
    assert not tools.is_allowed_goto("https://en.wikipedia.org/w/index.php?search=&title=Special%3ASearch&go=Go")


def test_run_tool_dispatches_and_rejects():
    browser = Mock(scroll_to_text=Mock(return_value=True))
    assert tools.run_tool(browser, "GOTO", WIKI_SEARCH) is True
    browser.navigate.assert_called_once_with(WIKI_SEARCH)
    assert tools.run_tool(browser, "SCROLL_TO_TEXT", "External links") is True
    with pytest.raises(ValueError, match="not a registered search URL"):
        tools.run_tool(browser, "GOTO", "https://en.wikipedia.org/wiki/X")
    with pytest.raises(ValueError, match="Unknown tool"):
        tools.run_tool(browser, "CLICK", "x")


def test_browser_scroll_to_text_passes_the_needle_as_json():
    b = Browser.__new__(Browser)
    b.evaluate = Mock(return_value=True)
    assert b.scroll_to_text('External "links"') is True
    assert '("External \\"links\\"")' in b.evaluate.call_args.args[0]


def test_browser_navigate_waits_for_load():
    b = Browser.__new__(Browser)
    b.call = Mock()
    b.evaluate = Mock(side_effect=["loading", "complete"])
    b.navigate("https://en.wikipedia.org/wiki/X")
    b.call.assert_called_once_with("Page.navigate", url="https://en.wikipedia.org/wiki/X")
    assert b.evaluate.call_count == 2
