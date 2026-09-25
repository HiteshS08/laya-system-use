"""What the agent already tried on each page, so failed or repeated actions are not offered again."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urldefrag

from .resolver import normalize

REPEAT_LIMIT = 2
# Roles whose CLICK is typically a focus click (e.g. opening a search box) rather than a real navigation:
# it does not change the page by itself, so a lack of change there is not a failure.
EDITABLE_ROLES = frozenset({"textbox", "searchbox", "combobox", "spinbutton"})


@dataclass(frozen=True)
class Attempt:
    url: str
    operation: str
    label: str
    display: str
    value: str
    outcome: str  # url_changed | page_changed | no_change
    role: str = ""
    tool_ok: bool | None = None


def _outcome(entry: Mapping) -> str:
    before = entry.get("url_before")
    if before is not None and entry.get("url") != before:
        return "url_changed"
    return "page_changed" if entry.get("page_changed") else "no_change"


def _attempt(entry: Mapping) -> Attempt:
    url = urldefrag(entry.get("url_before") or entry.get("url") or "")[0]
    operation = entry.get("operation") or str(entry.get("kind", "")).upper()
    label = str(entry.get("action") or "")
    return Attempt(url, operation, normalize(label), label, entry.get("text") or "", _outcome(entry),
                   str(entry.get("role") or ""), entry.get("tool_ok"))


def _is_failure(attempt: Attempt) -> bool:
    # A tool's success is reported directly; trust it over the page-change heuristic below.
    if attempt.tool_ok is not None:
        return attempt.tool_ok is False
    # Typing rarely changes the page by itself, and a focus click on an editable field (opening a search
    # box) doesn't either; both effects are inside the field, not on the page.
    return (attempt.outcome == "no_change" and attempt.operation != "TYPE_TEXT"
            and not (attempt.operation == "CLICK" and attempt.role in EDITABLE_ROLES))


class StepMemory:
    def __init__(self) -> None:
        self._attempts: tuple[Attempt, ...] = ()

    @property
    def attempts(self) -> tuple[Attempt, ...]:
        return self._attempts

    def sync(self, history: Sequence[Mapping]) -> None:
        new = tuple(_attempt(h) for h in history[len(self._attempts):])
        self._attempts = (*self._attempts, *new)

    def _on(self, url: str) -> list[Attempt]:
        page = urldefrag(url)[0]
        return [a for a in self._attempts if a.url == page]

    def _reasons(self, url: str) -> dict[tuple[str, str], tuple[str, str]]:
        here = self._on(url)
        counts = Counter((a.operation, a.label) for a in here)
        reasons: dict[tuple[str, str], tuple[str, str]] = {}
        for a in here:
            key = (a.operation, a.label)
            if _is_failure(a):
                reasons[key] = (a.display, "no effect")
            elif counts[key] >= REPEAT_LIMIT and key not in reasons:
                reasons[key] = (a.display, "repeated")
        return reasons

    def excluded(self, url: str) -> frozenset[tuple[str, str]]:
        return frozenset(self._reasons(url))

    def failed(self, url: str) -> list[str]:
        return [f"{op} {display} ({why})" for (op, _), (display, why) in self._reasons(url).items()]

    def last_failed(self) -> bool:
        return bool(self._attempts) and _is_failure(self._attempts[-1])

    def streak_without_url_change(self) -> int:
        streak = 0
        for a in reversed(self._attempts):
            if a.outcome == "url_changed":
                break
            streak += 1
        return streak
