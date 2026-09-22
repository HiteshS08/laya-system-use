"""Lexical ranking that keeps the K elements most related to the goal. Identical at training and serving time."""

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

from .candidates import Candidate

DEFAULT_K = 20
_WORD = re.compile(r"[a-z0-9]{2,}")
# Operation names appear in history strings; without this every "click" label would score on them.
_QUERY_STOP = frozenset({"click", "type", "text", "select", "the", "and", "for", "with", "from", "that", "this"})
# Additive priors that decide which unmatched elements fill the shortlist (measured on Mind2Web, see
# training/recall_study.py): fields are far likelier targets than links. Weights are small next to a rare-word match.
_ROLE_PRIOR = {
    "textbox": 1.0, "searchbox": 1.0, "combobox": 1.0,
    "button": 0.5, "option": 0.5, "tab": 0.5, "checkbox": 0.5, "spinbutton": 0.5, "switch": 0.5,
}
ROLE_WEIGHT = 0.4
POSITION_WEIGHT = 0.3  # earlier on the page is likelier; candidates arrive in page order


def _tokens(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return _WORD.findall(folded)


def _words(text: str) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _tokens(text))


def _ranked_indices(goal: str, history: Sequence[str], candidates: Sequence[Candidate]) -> list[int]:
    query = _words(" ".join([goal, *history])) - _QUERY_STOP
    texts = [f"{c.label} {c.value}" for c in candidates]
    words = [_words(t) for t in texts]
    doc_freq = Counter(w for ws in words for w in ws)
    lengths = [len(_tokens(t)) for t in texts]
    avg_length = (sum(lengths) / len(lengths)) if lengths else 1.0
    last = max(len(candidates) - 1, 1)
    # Rarer words weigh more; a match in a short label beats the same match buried in a long one.
    scores = [
        sum(1.0 / doc_freq[w] for w in ws & query) * avg_length / max(n, 1)
        + ROLE_WEIGHT * _ROLE_PRIOR.get(c.role, 0.0)
        + POSITION_WEIGHT * (1.0 - i / last)
        for i, (ws, n, c) in enumerate(zip(words, lengths, candidates, strict=True))
    ]
    return sorted(range(len(candidates)), key=lambda i: (-scores[i], i))


def rank_candidates(goal: str, history: Sequence[str], candidates: Sequence[Candidate]) -> list[Candidate]:
    return [candidates[i] for i in _ranked_indices(goal, history, candidates)]


def shortlist(
    goal: str, history: Sequence[str], candidates: Sequence[Candidate], k: int = DEFAULT_K
) -> tuple[Candidate, ...]:
    if len(candidates) <= k:
        return tuple(candidates)
    keep = sorted(_ranked_indices(goal, history, candidates)[:k])
    return tuple(candidates[i] for i in keep)
