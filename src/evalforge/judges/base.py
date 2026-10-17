"""The judge contract every evaluator implements."""

from __future__ import annotations

from typing import Protocol

from evalforge.models import Case, Verdict


class Judge(Protocol):
    """Scores one case. Implementations must be deterministic for a given input."""

    name: str

    def judge(self, case: Case) -> Verdict: ...


def require_reference(case: Case, judge_name: str) -> str:
    """Return the case reference or raise, so missing labels never score silently."""
    if case.reference is None:
        raise ValueError(f"{judge_name} needs a reference answer (case {case.id!r})")
    return case.reference
