"""Measure how well judge verdicts agree with human pass/fail labels."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import BaseModel

from evalforge.models import Verdict


class ConfusionMatrix(BaseModel):
    """Judge decisions versus human labels, with "pass" as the positive class."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


class AgreementReport(BaseModel):
    """Agreement statistics between a judge and human labels."""

    n: int
    accuracy: float
    kappa: float | None
    confusion: ConfusionMatrix


def agreement(
    verdicts: Iterable[Verdict], human_labels: Mapping[str, bool]
) -> AgreementReport:
    """Compare ``verdicts`` with ``human_labels`` (case id -> human pass/fail).

    ``kappa`` is Cohen's kappa. It is ``None`` when it is undefined, which happens
    when judge and human both use a single class for every case (chance agreement
    is then 1), so a degenerate dataset is never reported as perfect agreement.

    Raises ``ValueError`` if there are no verdicts, a case id appears twice, or the
    verdict ids and label ids do not match exactly.
    """
    by_id: dict[str, bool] = {}
    for verdict in verdicts:
        if verdict.case_id in by_id:
            raise ValueError(f"duplicate verdict for case {verdict.case_id!r}")
        by_id[verdict.case_id] = verdict.passed
    if not by_id:
        raise ValueError("no verdicts to compare")

    missing = sorted(set(by_id) - set(human_labels))
    unused = sorted(set(human_labels) - set(by_id))
    if missing or unused:
        raise ValueError(
            f"verdicts and human labels differ: no label for {missing}, "
            f"no verdict for {unused}"
        )

    matrix = ConfusionMatrix()
    for case_id, predicted in by_id.items():
        actual = human_labels[case_id]
        if predicted and actual:
            matrix.tp += 1
        elif predicted:
            matrix.fp += 1
        elif actual:
            matrix.fn += 1
        else:
            matrix.tn += 1

    n = matrix.total
    observed = (matrix.tp + matrix.tn) / n
    judge_pass = (matrix.tp + matrix.fp) / n
    human_pass = (matrix.tp + matrix.fn) / n
    expected = judge_pass * human_pass + (1 - judge_pass) * (1 - human_pass)
    kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    return AgreementReport(n=n, accuracy=observed, kappa=kappa, confusion=matrix)
