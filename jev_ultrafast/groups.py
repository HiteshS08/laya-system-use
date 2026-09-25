"""Ordinal targets ("the comments of the second story"): the Nth member of a group of repeated elements."""

import re
from collections.abc import Mapping, Sequence

from .resolver import normalize

_NUMBER = re.compile(r"\d+")


def _singular(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def shape(label: str) -> str:
    """'48 comments' and '1 comment' share the shape '# comment'."""
    return " ".join(_singular(w) for w in _NUMBER.sub("#", normalize(label)).split())


def ordinal_pick(elements: Sequence[Mapping], description: str, n: int) -> Mapping | None:
    wanted = {_singular(w) for w in normalize(description).split()}
    groups: dict[tuple[str, str, str], list[Mapping]] = {}
    for e in elements:
        if "CLICK" in e.get("operations", ()):
            key = (e.get("role", ""), e.get("landmark", ""), shape(e.get("label", "")))
            groups.setdefault(key, []).append(e)
    eligible = [members for (_, _, s), members in groups.items() if len(members) >= n and wanted & set(s.split())]
    if not eligible:
        return None
    # dicts keep insertion order, so on equal size the group that starts first on the page wins.
    return max(eligible, key=len)[n - 1]
