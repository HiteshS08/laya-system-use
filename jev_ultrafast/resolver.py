"""Deterministic element lookup: when the planner names exactly one element, no model call is needed."""

import re
import unicodedata
from collections.abc import Mapping, Sequence

_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(_PUNCT.sub(" ", plain.casefold()).split())


def _names(element: Mapping) -> list[str]:
    return [normalize(n) for n in [element.get("label", ""), *element.get("aliases", [])]]


def resolve(target_text: str, elements: Sequence[Mapping]) -> tuple[Mapping | None, str | None]:
    want = normalize(target_text)
    if not want:
        return None, None
    exact = [e for e in elements if want in _names(e)]
    if len(exact) == 1:
        return exact[0], "resolver"
    if exact:
        return None, None  # the same label twice: only the actor can tell them apart
    words = set(want.split())
    fuzzy = [e for e in elements if any(words <= set(name.split()) for name in _names(e))]
    if len(fuzzy) == 1:
        return fuzzy[0], "resolver_fuzzy"
    return None, None
