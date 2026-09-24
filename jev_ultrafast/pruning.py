"""Option-set hygiene before any model sees the page: drop known detours, merge links to the same place."""

import re
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

_DETOUR_QUERY = re.compile(r"(?:^|&)action=(?:edit|history|raw|info)(?:&|$)")
_DETOUR_PATH = re.compile(r"/edit(?:/|$)")
# A goal that talks about editing, history or source really wants those pages.
_DETOUR_GOAL = re.compile(r"\b(?:edit\w*|history|revisions?|source)\b", re.IGNORECASE)


def is_detour(href: str, goal: str) -> bool:
    if not href or _DETOUR_GOAL.search(goal):
        return False
    parts = urlsplit(href)
    return bool(_DETOUR_QUERY.search(parts.query) or _DETOUR_PATH.search(parts.path))


def _merge_key(href: str) -> str | None:
    """Only real destinations merge; '#' placeholders and javascript: links are distinct controls."""
    if not href.startswith(("http://", "https://")) or href.endswith("#"):
        return None
    return href


def prune_actions(actions: Sequence[Mapping], goal: str) -> list[dict]:
    kept: list[dict] = []
    first_by_href: dict[str, dict] = {}
    for action in actions:
        href = action.get("href") or ""
        is_link = action.get("kind") == "click" and action.get("role") == "link"
        if action.get("kind") == "click" and is_detour(href, goal):
            continue
        key = _merge_key(href) if is_link else None
        if key is None:
            kept.append(dict(action))
            continue
        first = first_by_href.get(key)
        if first is None:
            first = {**action, "aliases": list(action.get("aliases", []))}
            first_by_href[key] = first
            kept.append(first)
        elif action.get("label") and action["label"] != first["label"] and action["label"] not in first["aliases"]:
            first["aliases"] = [*first["aliases"], action["label"]]
    return kept
