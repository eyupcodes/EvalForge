"""Reference-based judges that need no model calls."""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field

from evalforge.judges.base import require_reference
from evalforge.models import Case, Verdict

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and collapse whitespace."""
    return " ".join(text.lower().translate(_PUNCT).split())


@dataclass
class ExactMatchJudge:
    """Passes when output equals the reference after optional normalization."""

    normalized: bool = True
    name: str = "exact_match"

    def judge(self, case: Case) -> Verdict:
        reference = require_reference(case, self.name)
        output = case.output
        if self.normalized:
            output, reference = normalize(output), normalize(reference)
        hit = output == reference
        return Verdict(
            case_id=case.id,
            judge=self.name,
            score=1.0 if hit else 0.0,
            passed=hit,
            reason="output matches reference" if hit else "output differs",
        )


@dataclass
class ContainsAllJudge:
    """Scores the share of required phrases present in the output."""

    phrases: list[str]
    threshold: float = 1.0
    name: str = "contains_all"

    def judge(self, case: Case) -> Verdict:
        if not self.phrases:
            raise ValueError("ContainsAllJudge needs at least one phrase")
        output = normalize(case.output)
        missing = [p for p in self.phrases if normalize(p) not in output]
        score = 1 - len(missing) / len(self.phrases)
        return Verdict(
            case_id=case.id,
            judge=self.name,
            score=score,
            passed=score >= self.threshold,
            reason=f"missing: {', '.join(missing)}" if missing else "all present",
        )


@dataclass
class RegexJudge:
    """Passes when the output matches a regular expression."""

    pattern: str
    flags: int = re.IGNORECASE
    name: str = "regex"
    _compiled: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._compiled = re.compile(self.pattern, self.flags)

    def judge(self, case: Case) -> Verdict:
        hit = self._compiled.search(case.output) is not None
        return Verdict(
            case_id=case.id,
            judge=self.name,
            score=1.0 if hit else 0.0,
            passed=hit,
            reason=f"pattern {'found' if hit else 'not found'}: {self.pattern}",
        )


@dataclass
class TokenF1Judge:
    """SQuAD-style token F1 between output and reference."""

    threshold: float = 0.5
    name: str = "token_f1"

    def judge(self, case: Case) -> Verdict:
        reference = require_reference(case, self.name)
        predicted = normalize(case.output).split()
        gold = normalize(reference).split()
        common = sum((Counter(predicted) & Counter(gold)).values())
        if common == 0:
            f1 = 0.0
        else:
            precision = common / len(predicted)
            recall = common / len(gold)
            f1 = 2 * precision * recall / (precision + recall)
        return Verdict(
            case_id=case.id,
            judge=self.name,
            score=round(f1, 4),
            passed=f1 >= self.threshold,
            reason=f"token F1 {f1:.2f} (threshold {self.threshold})",
        )
