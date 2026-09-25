"""Non-element actions: scroll to text, go to a rendered site-search URL, press Enter, go to an observed anchor."""

import json
import re
from collections.abc import Sequence
from urllib.parse import urlsplit

from .search import matches_template

# Sites with a stable GET search. The planner may only navigate to a rendering of one of these.
SEARCH_TEMPLATES = {
    "en.wikipedia.org": "https://en.wikipedia.org/w/index.php?search={q}&title=Special%3ASearch&go=Go",
    "github.com": "https://github.com/search?q={q}&type=repositories",
}
_ALLOWED = [re.compile(re.escape(t).replace(re.escape("{q}"), r"[^&#]+")) for t in SEARCH_TEMPLATES.values()]


def search_template(page_url: str) -> str | None:
    return SEARCH_TEMPLATES.get(urlsplit(page_url).hostname or "")


def is_allowed_goto(url: str) -> bool:
    return any(pattern.fullmatch(url) for pattern in _ALLOWED)


def run_tool(browser, operation: str, arg: str, *, templates: Sequence[str] = ()) -> bool:
    """GOTO only to a URL rendered from the static registry or from a template the controller handed over."""
    if operation == "GOTO":
        if not (is_allowed_goto(arg) or any(matches_template(arg, t) for t in templates)):
            raise ValueError(f"GOTO target is not a registered search URL: {arg!r}; not navigating.")
        browser.navigate(arg)
        return True
    if operation == "SCROLL_TO_TEXT":
        return browser.scroll_to_text(arg)
    if operation == "SUBMIT":
        browser.press_enter()
        return True
    if operation == "FRAGMENT":
        # The id comes from the observed page and is passed as a JSON string literal, never as code.
        browser.evaluate(f"location.hash = {json.dumps(arg)}")
        return True
    raise ValueError(f"Unknown tool operation {operation!r}")
