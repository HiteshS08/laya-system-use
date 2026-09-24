"""Non-element actions the planner may take: scroll to text, and go to a registered site-search URL."""

import re
from urllib.parse import urlsplit

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


def run_tool(browser, operation: str, arg: str) -> bool:
    if operation == "GOTO":
        if not is_allowed_goto(arg):
            raise ValueError(f"GOTO target is not a registered search URL: {arg!r}; not navigating.")
        browser.navigate(arg)
        return True
    if operation == "SCROLL_TO_TEXT":
        return browser.scroll_to_text(arg)
    raise ValueError(f"Unknown tool operation {operation!r}")
