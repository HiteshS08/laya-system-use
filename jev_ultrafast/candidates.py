"""Neutral element record shared by training-data generation and live serving."""

from dataclasses import dataclass

OPERATIONS = ("CLICK", "TYPE_TEXT", "SELECT")


@dataclass(frozen=True)
class Candidate:
    id: str
    label: str
    role: str
    value: str = ""
    ops: frozenset[str] = frozenset({"CLICK"})
    context: str = ""
