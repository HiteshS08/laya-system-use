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


def test_goto_accepts_only_urls_rendered_from_a_handed_template():
    from jev_ultrafast.search import render

    t = "https://ex.test/find?q={q}"
    browser = Mock()
    assert tools.run_tool(browser, "GOTO", render(t, "ada"), templates=(t,)) is True
    browser.navigate.assert_called_once()
    with pytest.raises(ValueError):
        tools.run_tool(browser, "GOTO", "https://evil.test/find?q=ada", templates=(t,))


def test_submit_presses_enter():
    browser = Mock()
    assert tools.run_tool(browser, "SUBMIT", "") is True
    browser.press_enter.assert_called_once()


def test_press_enter_dispatches_key_events_then_waits_for_navigation(monkeypatch):
    from jev_ultrafast import browser as b

    br = b.Browser.__new__(b.Browser)
    br.call = Mock()
    urls = iter(["https://ex.test/", "https://ex.test/", "https://ex.test/results?q=a"])
    br.evaluate = Mock(side_effect=lambda expr, **kw: next(urls) if expr == "location.href" else "complete")
    monkeypatch.setattr(b.time, "sleep", Mock())
    br.press_enter()
    kinds = [c.kwargs.get("type") for c in br.call.call_args_list if c.args[0] == "Input.dispatchKeyEvent"]
    assert kinds == ["keyDown", "keyUp"]
    assert br.evaluate.call_args.args[0] == "document.readyState"


def test_evaluate_can_await_a_promise():
    b = Browser.__new__(Browser)
    b.call = Mock(return_value={"result": {"value": "xml"}})
    assert b.evaluate("fetch()", await_promise=True) == "xml"
    assert b.call.call_args.kwargs["awaitPromise"] is True


def test_fragment_sets_the_hash_as_json_data():
    browser = Mock(evaluate=Mock(return_value="#Refs"))
    assert tools.run_tool(browser, "FRAGMENT", 'Refs"; alert(1); "') is True
    assert browser.evaluate.call_args.args[0] == 'location.hash = "Refs\\"; alert(1); \\""'
