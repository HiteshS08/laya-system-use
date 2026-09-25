"""Generic tactics: what concrete step moves a subgoal forward on this page. No site names, no model calls."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urldefrag

from .checks import fragment_names, name_forms
from .instructions import instruction
from .program import Subgoal
from .resolver import normalize, resolve
from .search import render

SEARCH_ROLES = frozenset({"searchbox", "combobox", "textbox"})
OPTION_ROLES = frozenset({"option", "menuitem", "menuitemradio", "gridcell"})


@dataclass(frozen=True)
class Step:
    operation: str  # CLICK | TYPE_TEXT | SELECT | SCROLL_TO_TEXT | GOTO | SUBMIT | FRAGMENT | DO
    target_text: str
    value: str
    instruction: str
    ordinal: int = 0
    index: str | None = None  # preset element chosen deterministically by the tactic
    roles: frozenset[str] = frozenset()
    templates: tuple[str, ...] = ()
    purpose: str = "act"  # act | search | submit | open_search | open | pick | result


@dataclass(frozen=True)
class Progress:
    actions: int = 0
    misses: int = 0
    typed: bool = False
    submitted: bool = False
    searched: bool = False
    opened_search: bool = False
    opened: bool = False
    picked: bool = False
    acted: bool = False  # an effective action of the subgoal's own kind
    scrolled: bool = False


def _can(e: Mapping, op: str) -> bool:
    return op in e.get("operations", ())


def _exact_link(elements: Sequence[Mapping], name: str) -> Mapping | None:
    want = normalize(name)
    return next((e for e in elements if _can(e, "CLICK") and e.get("role") == "link" and
                 any(want in name_forms(n) for n in [e.get("label", ""), *e.get("aliases", [])])), None)


def _search_field(elements: Sequence[Mapping]) -> Mapping | None:
    fields = [e for e in elements if _can(e, "TYPE_TEXT")]
    # Only fields that say they search: typing a name into an arbitrary field (a newsletter box) does harm.
    return (next((e for e in fields if e.get("role") == "searchbox"), None)
            or next((e for e in fields if "search" in normalize(e.get("label", "")).split()), None))


def _search_opener(elements: Sequence[Mapping]) -> Mapping | None:
    """A button or link that says 'search' (opens a collapsed search box, or submits a typed one)."""
    return next((e for e in elements if _can(e, "CLICK") and not _can(e, "TYPE_TEXT")
                 and e.get("role") in {"button", "link"} and "search" in normalize(e.get("label", "")).split()), None)


def matching_option(elements: Sequence[Mapping], value: str) -> Mapping | None:
    words = normalize(value).split()
    return next((e for e in elements if _can(e, "CLICK") and e.get("role") in OPTION_ROLES and words
                 and all(w in normalize(e.get("label", "")).split() for w in words)), None)


def _find(sub: Subgoal, elements: Sequence[Mapping], progress: Progress, template: str | None) -> Step:
    name = sub.target
    link = _exact_link(elements, name)
    if link is not None:
        return Step("CLICK", name, "", instruction("LINK", name), index=link["index"])
    if template and not progress.searched and not progress.typed:
        return Step("GOTO", render(template, name), "", instruction("SEARCH", value=name), templates=(template,),
                    purpose="search")
    if not progress.typed and not progress.searched:
        field = _search_field(elements)
        if field is not None:
            return Step("TYPE_TEXT", field["label"], name, instruction("SEARCH", value=name), index=field["index"],
                        purpose="search")
        opener = _search_opener(elements)
        if opener is not None and not progress.opened_search:
            return Step("CLICK", opener["label"], "", instruction("OPEN_SEARCH"), index=opener["index"],
                        purpose="open_search")
    elif progress.typed and not progress.submitted:
        button = _search_opener(elements) if progress.misses else None
        if button is not None:  # Enter did nothing: press the search button instead
            return Step("CLICK", button["label"], "", instruction("CLICK", button["label"]), index=button["index"],
                        purpose="submit")
        return Step("SUBMIT", "", "", instruction("SUBMIT"), purpose="search")
    return Step("CLICK", name, "", instruction("RESULT", name), purpose="result")


def _jump(sub: Subgoal, page: Mapping, elements: Sequence[Mapping], progress: Progress) -> Step:
    here = urldefrag(page.get("url", ""))[0]
    link = next((e for e in elements if _can(e, "CLICK") and e.get("href") and
                 urldefrag(e["href"])[0] == here and fragment_names(e["href"], sub.target)), None)
    if link is not None:
        return Step("CLICK", link.get("label", sub.target), "", instruction("JUMP", sub.target), index=link["index"])
    want = normalize(sub.target)
    heading = next((h for h in page.get("headings", []) if h.get("id") and normalize(h.get("text", "")) == want), None)
    if heading is not None and not progress.misses:
        # No visible contents link (e.g. a collapsed table of contents): go to the observed heading's own anchor.
        return Step("FRAGMENT", heading["id"], "", instruction("JUMP", sub.target))
    return Step("SCROLL_TO_TEXT", sub.target, "", instruction("SCROLL", sub.target))


def _fill(sub: Subgoal, elements: Sequence[Mapping], progress: Progress) -> Step | None:
    if not progress.typed:
        return Step("TYPE_TEXT", sub.target, sub.value, instruction("FILL", sub.target, sub.value))
    option = None if progress.picked else matching_option(elements, sub.value)
    if option is None:
        return None
    return Step("CLICK", option["label"], "", instruction("OPTION", value=sub.value), index=option["index"],
                purpose="pick")


def _select(sub: Subgoal, elements: Sequence[Mapping], progress: Progress) -> Step | None:
    option = None if progress.picked else matching_option(elements, sub.value)
    if option is not None:
        return Step("CLICK", option["label"], "", instruction("OPTION", value=sub.value), index=option["index"],
                    purpose="pick")
    selects = [e for e in elements if _can(e, "SELECT")]
    native, _ = resolve(sub.target, selects)
    native = native or (selects[0] if len(selects) == 1 else None)
    if native is not None:
        return Step("SELECT", sub.target, sub.value, instruction("SELECT", sub.target, sub.value),
                    index=native["index"])
    if progress.opened:
        return None
    return Step("CLICK", sub.target, "", instruction("CLICK", sub.target), purpose="open")


def next_step(subgoal: Subgoal, page: Mapping, elements: Sequence[Mapping], progress: Progress,
              template: str | None = None) -> Step | None:
    kind, target = subgoal.kind, subgoal.target
    if kind == "FIND":
        return _find(subgoal, elements, progress, template)
    if kind == "JUMP":
        return _jump(subgoal, page, elements, progress)
    if kind == "FILL":
        return _fill(subgoal, elements, progress)
    if kind == "SELECT":
        return _select(subgoal, elements, progress)
    if kind == "SCROLL":
        return Step("SCROLL_TO_TEXT", target, "", instruction("SCROLL", target))
    if kind == "OPEN":
        return Step("CLICK", target, "", instruction("OPEN", target), ordinal=subgoal.ordinal)
    if kind == "SUBMIT" and not target:
        return Step("SUBMIT", "", "", instruction("SUBMIT"))
    if kind in ("CLICK", "SUBMIT"):
        return Step("CLICK", target, "", instruction("CLICK", target), ordinal=subgoal.ordinal)
    return Step("DO", target, "", instruction("DO", target))
