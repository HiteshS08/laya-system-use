"""Runs snapshot.js in real Chrome. Needs a throwaway Chrome on BU_CDP_URL; skipped otherwise."""

import os
from pathlib import Path

import httpx
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "context_page.html"


def _chrome_up() -> bool:
    url = os.environ.get("BU_CDP_URL")
    if not url:
        return False
    try:
        return httpx.get(url + "/json/version", timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(not _chrome_up(), reason="needs a throwaway Chrome on BU_CDP_URL")


@pytest.fixture(scope="module")
def observed():
    from jev_ultrafast.browser import Browser

    browser = Browser(FIXTURE.as_uri())
    try:
        yield browser, browser.observe(screenshot=False)
    finally:
        browser.close()


def by_label(page, label):
    return next(a for a in page["actions"] if a["label"] == label)


def test_offscreen_elements_are_captured_with_position(observed):
    _, page = observed
    archive = by_label(page, "Official archive")
    assert archive["in_viewport"] is False and archive["y"] > 780
    assert by_label(page, "Mary Mallon")["in_viewport"] is True


def test_context_fields(observed):
    _, page = observed
    assert by_label(page, "Home")["landmark"] == "header" and by_label(page, "Home")["section"] == ""
    assert by_label(page, "comments")["landmark"] == "nav"
    mallon = by_label(page, "Mary Mallon")
    assert (mallon["landmark"], mallon["section"]) == ("main", "From today's featured article")
    assert mallon["href"].endswith("/wiki/Mary_Mallon")
    assert by_label(page, "48 comments")["row_text"] == "Story one48 comments"
    assert by_label(page, "Official archive")["section"] == "External links"
    assert by_label(page, "Official archive")["row_text"] == "Archive Official archive"
    assert by_label(page, "About")["landmark"] == "footer"
    assert page["outline"] == "From today's featured article | External links"


def test_offscreen_target_is_scrolled_into_view_and_clicked(observed):
    browser, page = observed
    target = by_label(page, "Official archive")
    browser.act(target, page)
    assert browser.evaluate("location.hash") == "#archive"


def test_headings_and_opensearch_link(observed):
    _, page = observed
    headings = {h["text"]: h for h in page["headings"]}
    assert headings["From today's featured article"]["in_viewport"] is True
    assert headings["External links"]["in_viewport"] is False and headings["External links"]["id"] == "External_links"
    assert headings["External links"]["level"] == 2
    assert page["opensearch"].endswith("/opensearch.xml")
