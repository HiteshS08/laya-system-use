"""Ordinal targets ("the comments of the second story"): the Nth member of a group of repeated elements."""

import re
from collections.abc import Callable, Mapping, Sequence

from .resolver import normalize

_NUMBER = re.compile(r"\d+")
# Label and description words that name the same repeated link: Hacker News labels an uncommented story's
# comments link "discuss", and people call the comments page "the discussion" or "the thread".
_ALIASES = {"discuss": "comment", "discussion": "comment", "thread": "comment"}
_STEM = 4
GroupKey = tuple[str, str, str]


def _singular(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _word(word: str) -> str:
    return _ALIASES.get(word, _singular(word))


def shape(label: str) -> str:
    """'48 comments', '1 comment' and 'discuss' share the shape '# comment'."""
    if normalize(label) == "discuss":
        return "# comment"
    return " ".join(_word(w) for w in _NUMBER.sub("#", normalize(label)).split())


def _key(e: Mapping) -> GroupKey:
    return e.get("role", ""), e.get("landmark", ""), shape(e.get("label", ""))


def _groups(elements: Sequence[Mapping]) -> dict[GroupKey, list[Mapping]]:
    groups: dict[GroupKey, list[Mapping]] = {}
    for e in elements:
        if "CLICK" in e.get("operations", ()):
            groups.setdefault(_key(e), []).append(e)
    return groups


def _largest(groups: Mapping[GroupKey, list[Mapping]], n: int, fits: Callable[[set[str]], bool]) -> Mapping | None:
    eligible = [members for (_, _, s), members in groups.items() if len(members) >= n and fits(set(s.split()))]
    # dicts keep insertion order, so on equal size the group that starts first on the page wins.
    return max(eligible, key=len)[n - 1] if eligible else None


def _resembles(wanted: set[str], words: set[str]) -> bool:
    stems = {w[:_STEM] for w in wanted if len(w) >= _STEM}
    return any(w[:_STEM] in stems for w in words if len(w) >= _STEM)


def ordinal_pick(elements: Sequence[Mapping], description: str, n: int,
                 anchor: Callable[[], Mapping | None] | None = None) -> Mapping | None:
    """The Nth member of: the largest repeated group whose shape shares a word with the description; else the
    group of the anchor element (the resolver's or actor's pick for the description); else the largest repeated
    group whose shape resembles the description."""
    wanted = {_word(w) for w in normalize(description).split()}
    groups = _groups(elements)
    picked = _largest(groups, n, lambda words: bool(wanted & words))
    if picked is not None:
        return picked
    element = anchor() if anchor else None
    members = groups.get(_key(element), []) if element is not None else []
    if len(members) >= n:
        return members[n - 1]
    return _largest(groups, n, lambda words: _resembles(wanted, words))
