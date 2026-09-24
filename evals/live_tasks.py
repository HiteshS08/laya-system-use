"""Public-site tasks with checks independent of the agent's DONE verdict."""

import base64
import re
from dataclasses import dataclass
from datetime import date
from typing import Sequence
from urllib.parse import parse_qs, urljoin, urlsplit, urlunsplit


@dataclass(frozen=True)
class LiveTask:
    url: str
    goal: str
    url_regex: str | None = None
    text_regex: str | None = None
    expected_link_script: str | None = None
    category: str = ""
    via_url_regex: str | None = None
    min_scroll_y: int = 0
    flight_date: str | None = None
    flight_origin_regex: str | None = None
    flight_destination_regex: str | None = None


TASKS = {
    "wiki_featured": LiveTask(
        "https://en.wikipedia.org/wiki/Main_Page",
        "Open today's featured article.",
        expected_link_script='document.querySelector("#mp-tfa p a[href*=\'/wiki/\']")?.href ?? null',
        category="disambiguation",
    ),
    "wiki_search": LiveTask(
        "https://en.wikipedia.org/wiki/Main_Page",
        "Find and open the Wikipedia article about Gödel's incompleteness theorems.",
        url_regex=r"^https://en\.wikipedia\.org/wiki/G%C3%B6del%27s_incompleteness_theorems$",
        category="site_search",
    ),
    "wiki_long_page": LiveTask(
        "https://en.wikipedia.org/wiki/Gödel%27s_incompleteness_theorems",
        "Open the Wikipedia article about Kurt Gödel, the logician who proved these theorems.",
        url_regex=r"^https://en\.wikipedia\.org/wiki/Kurt_G%C3%B6del$",
        category="below_fold",
    ),
    "hn_comments": LiveTask(
        "https://news.ycombinator.com/",
        "Open the comments page of the top story.",
        expected_link_script=(
            '(() => { const story = document.querySelector("tr.athing"); '
            'return story?.nextElementSibling?.querySelector("a[href^=\'item?id=\']")?.href ?? null; })()'
        ),
        category="disambiguation",
    ),
    "gh_issues": LiveTask(
        "https://github.com/browser-use/browser-use",
        "Open the Issues tab of this repository.",
        url_regex=r"^https://github\.com/browser-use/browser-use/issues(?:[/?#].*)?$",
        category="in_page_navigation",
    ),
    "flights": LiveTask(
        "https://www.google.com/travel/flights?hl=en",
        "Find one-way flights from Zurich to London on October 20, 2026, for one adult in economy. "
        "Stop when matching flight options are visible.",
        url_regex=r"^https://www\.google\.com/travel/flights/search(?:[?#].*)?$",
        category="multi_field_form",
        flight_date="2026-10-20",
        flight_origin_regex=r"(?:Zürich|Zurich)", flight_destination_regex=r"London",
    ),
}

TASKS.update({
    "wiki_search_turing": LiveTask(
        "https://en.wikipedia.org/wiki/Main_Page", "Find and open the article about Alan Turing.",
        url_regex=r"^https://en\.wikipedia\.org/wiki/Alan_Turing$", category="site_search",
    ),
    "wiki_search_ada": LiveTask(
        "https://en.wikipedia.org/wiki/Main_Page", "Find and open the article about Ada Lovelace.",
        url_regex=r"^https://en\.wikipedia\.org/wiki/Ada_Lovelace$", category="site_search",
    ),
    "wiki_search_python": LiveTask(
        "https://en.wikipedia.org/wiki/Main_Page", "Find and open the article about Python the programming language.",
        url_regex=r"^https://en\.wikipedia\.org/wiki/Python_(?:\(|%28)programming_language(?:\)|%29)$",
        category="site_search",
    ),
    "wiki_search_mallon": LiveTask(
        "https://en.wikipedia.org/wiki/Main_Page", "Find and open the article about Mary Mallon.",
        url_regex=r"^https://en\.wikipedia\.org/wiki/Mary_Mallon$", category="site_search",
    ),
    "turing_references": LiveTask(
        "https://en.wikipedia.org/wiki/Alan_Turing", "Jump to the References section of this article.",
        url_regex=r"^https://en\.wikipedia\.org/wiki/Alan_Turing#References$", category="in_page_navigation",
    ),
    "ada_references": LiveTask(
        "https://en.wikipedia.org/wiki/Ada_Lovelace", "Jump to the References section of this article.",
        url_regex=r"^https://en\.wikipedia\.org/wiki/Ada_Lovelace#References$", category="in_page_navigation",
    ),
    "python_references": LiveTask(
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "Jump to the References section of this article.",
        url_regex=r"^https://en\.wikipedia\.org/wiki/Python_(?:\(|%28)programming_language(?:\)|%29)#References$",
        category="in_page_navigation",
    ),
    "hn_second_comments": LiveTask(
        "https://news.ycombinator.com/", "Open the comments page of the second story, not the top story.",
        expected_link_script=(
            '(() => { const story = document.querySelectorAll("tr.athing")[1]; '
            'return story?.nextElementSibling?.querySelector("a[href^=\'item?id=\']")?.href ?? null; })()'
        ),
        category="disambiguation",
    ),
    "hn_third_comments": LiveTask(
        "https://news.ycombinator.com/", "Open the comments page of the third story, not the top story.",
        expected_link_script=(
            '(() => { const story = document.querySelectorAll("tr.athing")[2]; '
            'return story?.nextElementSibling?.querySelector("a[href^=\'item?id=\']")?.href ?? null; })()'
        ),
        category="disambiguation",
    ),
    "flights_paris_rome": LiveTask(
        "https://www.google.com/travel/flights?hl=en",
        "Find one-way flights from Paris to Rome on November 4, 2026, for one adult in economy. "
        "Stop when matching flight options are visible.",
        url_regex=r"^https://www\.google\.com/travel/flights/search(?:[?#].*)?$",
        category="multi_field_form",
        flight_date="2026-11-04",
        flight_origin_regex=r"Paris", flight_destination_regex=r"Rome",
    ),
    "flights_mumbai_delhi": LiveTask(
        "https://www.google.com/travel/flights?hl=en",
        "Find one-way flights from Mumbai to Delhi on December 2, 2026, for one adult in economy. "
        "Stop when matching flight options are visible.",
        url_regex=r"^https://www\.google\.com/travel/flights/search(?:[?#].*)?$",
        category="multi_field_form",
        flight_date="2026-12-02",
        flight_origin_regex=r"Mumbai", flight_destination_regex=r"(?:New )?Delhi",
    ),
})

for name, article in (
    ("turing_external_links", "Alan_Turing"),
    ("ada_external_links", "Ada_Lovelace"),
    ("python_external_links", "Python_(programming_language)"),
    ("godel_external_links", "Kurt_G%C3%B6del"),
):
    TASKS[name] = LiveTask(
        f"https://en.wikipedia.org/wiki/{article}",
        "Scroll down until the External links section heading is visible.",
        text_regex=r"(?m)^External links$", category="below_fold", min_scroll_y=1000,
    )

TASKS.update({
    "turing_to_award": LiveTask(
        "https://en.wikipedia.org/wiki/Main_Page",
        "Find Alan Turing's article, then open the Turing Award article from it.",
        url_regex=r"^https://en\.wikipedia\.org/wiki/Turing_Award$",
        via_url_regex=r"^https://en\.wikipedia\.org/wiki/Alan_Turing$", category="multi_page_flow",
    ),
    "ada_to_babbage": LiveTask(
        "https://en.wikipedia.org/wiki/Main_Page",
        "Find Ada Lovelace's article, then open the Charles Babbage article from it.",
        url_regex=r"^https://en\.wikipedia\.org/wiki/Charles_Babbage$",
        via_url_regex=r"^https://en\.wikipedia\.org/wiki/Ada_Lovelace$", category="multi_page_flow",
    ),
    "python_to_guido": LiveTask(
        "https://en.wikipedia.org/wiki/Main_Page",
        "Find the Python programming language article, then open the Guido van Rossum article from it.",
        url_regex=r"^https://en\.wikipedia\.org/wiki/Guido_van_Rossum$",
        via_url_regex=r"^https://en\.wikipedia\.org/wiki/Python_(?:\(|%28)programming_language(?:\)|%29)$",
        category="multi_page_flow",
    ),
    "mallon_to_typhoid": LiveTask(
        "https://en.wikipedia.org/wiki/Main_Page",
        "Find the Mary Mallon article, then open the typhoid fever article from it.",
        url_regex=r"^https://en\.wikipedia\.org/wiki/Typhoid_fever$",
        via_url_regex=r"^https://en\.wikipedia\.org/wiki/Mary_Mallon$", category="multi_page_flow",
    ),
})


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), parts.query, ""))


def matches_url(pattern: str, url: str) -> bool:
    """Keep fragments for section targets; ignore them for page destinations."""
    return bool(re.search(pattern, url if "#" in pattern else canonical_url(url)))


def check_outcome(task: LiveTask, final_url: str, final_text: str, expected_link: str | None = None,
                  *, visited_urls: tuple[str, ...] = (), scroll_y: int = 0,
                  final_actions: Sequence[dict[str, object]] = ()) -> bool:
    """A task passes only when every specified final-page condition holds."""
    if task.expected_link_script:
        if not expected_link:
            return False
        if canonical_url(final_url) != canonical_url(urljoin(task.url, expected_link)):
            return False
    if task.url_regex and not matches_url(task.url_regex, final_url):
        return False
    if task.text_regex and not re.search(task.text_regex, final_text):
        return False
    if task.flight_date:
        encoded = parse_qs(urlsplit(final_url).query).get("tfs", [""])[0]
        try:
            route = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        except ValueError:
            return False
        if task.flight_date.encode() not in route:
            return False
        trip_date = date.fromisoformat(task.flight_date)
        fields = {str(action.get("label", "")).strip(): str(action.get("value", ""))
                  for action in final_actions}
        departure = fields.get("Departure", "")
        if not (fields.get("Change ticket type. One way") == "One way"
                and any(label.startswith("1 passenger") for label in fields)
                and fields.get("Change seating class. Economy") == "Economy"
                and task.flight_origin_regex
                and re.fullmatch(task.flight_origin_regex, fields.get("Where from?", ""), re.IGNORECASE)
                and task.flight_destination_regex
                and re.fullmatch(task.flight_destination_regex, fields.get("Where to?", ""), re.IGNORECASE)
                and f"{trip_date:%b} {trip_date.day}" in departure):
            return False
        result_date = f"{trip_date:%B} {trip_date.day}"
        if "Search results" not in final_text or not any(
            "Select flight" in str(action.get("label", ""))
            and result_date in str(action.get("label", "")) for action in final_actions
        ):
            return False
    if task.via_url_regex and not any(matches_url(task.via_url_regex, url) for url in visited_urls):
        return False
    if scroll_y < task.min_scroll_y:
        return False
    return bool(task.expected_link_script or task.url_regex or task.text_regex or task.flight_date)
