"""Deterministic completion checks on an observed page. No model decides whether a subgoal is done."""

import re
from collections.abc import Mapping
from urllib.parse import unquote, urldefrag

from .program import Subgoal
from .resolver import normalize

_TITLE_SEPARATORS = re.compile(r"\s+[-–—|·]\s+")
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*$")
_BRACKETED = re.compile(r"\s*\[[^\]]*\]\s*$")  # "References[edit]" on skins that put the edit link inside


def title_subject(title: str) -> str:
    """'Alan Turing - Wikipedia' -> 'Alan Turing': the part before the site name."""
    return _TITLE_SEPARATORS.split(title.strip(), maxsplit=1)[0].strip()


def name_forms(text: str) -> set[str]:
    """A name and the same name without a trailing disambiguator: 'Python (programming language)' -> 'Python'."""
    return {f for f in (normalize(text), normalize(_PARENTHETICAL.sub("", text))) if f}


def _first_h1(page: Mapping) -> str:
    return next((h["text"] for h in page.get("headings", []) if h.get("level") == 1), "")


def is_about(page: Mapping, name: str) -> bool:
    want = normalize(name)
    if not want:
        return False
    return any(want in name_forms(text) for text in (title_subject(page.get("title", "")), _first_h1(page)) if text)


def fragment_names(url: str, section: str) -> bool:
    fragment = urldefrag(url)[1]
    want = normalize(section)
    return bool(fragment and want) and normalize(unquote(fragment).replace("_", " ")) == want


def heading_in_view(page: Mapping, text: str) -> bool:
    want = normalize(text)
    return bool(want) and any(h.get("in_viewport") and normalize(_BRACKETED.sub("", h.get("text", ""))) == want
                              for h in page.get("headings", []))


def text_shown(page: Mapping, text: str) -> bool:
    want = normalize(text)
    return bool(want) and (want in normalize(page.get("text", "")) or want in normalize(page.get("title", "")))


def _document(url: str) -> str:
    return urldefrag(url)[0].rstrip("/")


def satisfied(subgoal: Subgoal, page: Mapping, start_url: str) -> bool | None:
    """Whether the page shows the subgoal met; None when only execution can tell (FILL, SELECT, CLICK, SUBMIT)."""
    if subgoal.kind == "FIND":
        return is_about(page, subgoal.target)
    if subgoal.kind == "JUMP":
        return fragment_names(page["url"], subgoal.target) or heading_in_view(page, subgoal.target)
    if subgoal.kind == "SCROLL":
        return heading_in_view(page, subgoal.target)
    if subgoal.kind == "OPEN":
        return _document(page["url"]) != _document(start_url)
    return None
