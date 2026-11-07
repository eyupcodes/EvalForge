"""Measure how well judge verdicts agree with human pass/fail labels."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping

from pydantic import BaseModel

from evalforge.models import Verdict

CI_LEVEL = 0.95


class ConfusionMatrix(BaseModel):
    """Judge decisions versus human labels, with "pass" as the positive class."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


class Interval(BaseModel):
    """A two-sided percentile interval; a bound is ``None`` when undefined."""

    low: float | None
    high: float | None


class ConfidenceIntervals(BaseModel):
    """Bootstrap intervals for the headline agreement statistics."""

    level: float
    resamples: int
    seed: int
    accuracy: Interval
    kappa: Interval


class AgreementReport(BaseModel):
    """Agreement statistics between a judge and human labels."""

    n: int
    accuracy: float
    kappa: float | None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    confusion: ConfusionMatrix
    ci: ConfidenceIntervals | None = None


def agreement(
    verdicts: Iterable[Verdict],
    human_labels: Mapping[str, bool],
    *,
    bootstrap: int = 0,
    seed: int = 0,
) -> AgreementReport:
    """Compare ``verdicts`` with ``human_labels`` (case id -> human pass/fail).

    ``kappa`` is Cohen's kappa. It is ``None`` when it is undefined, which happens
    when judge and human both use a single class for every case (chance agreement
    is then 1), so a degenerate dataset is never reported as perfect agreement.

    ``precision`` (share of judge passes that humans also passed), ``recall`` (share
    of human passes the judge found) and ``f1`` treat "pass" as the positive class.
    Each is ``None`` when its denominator is zero.

    With ``bootstrap=N`` (N > 0) the cases are resampled with replacement N times
    and ``ci`` holds 95% percentile intervals for accuracy and kappa. The resampling
    is seeded, so the same inputs and ``seed`` give the same intervals. Resamples
    where kappa is undefined are left out of the kappa interval, which is ``None`` on
    both sides if every resample was degenerate.

    Raises ``ValueError`` if there are no verdicts, a case id appears twice, or the
    verdict ids and label ids do not match exactly, or ``bootstrap`` is negative.
    """
    if bootstrap < 0:
        raise ValueError("bootstrap must not be negative")
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

    pairs = [(predicted, human_labels[case_id]) for case_id, predicted in by_id.items()]
    matrix = _confusion(pairs)
    n = matrix.total
    observed, kappa = _accuracy_kappa(matrix)
    precision = _ratio(matrix.tp, matrix.tp + matrix.fp)
    recall = _ratio(matrix.tp, matrix.tp + matrix.fn)
    f1 = _ratio(2 * matrix.tp, 2 * matrix.tp + matrix.fp + matrix.fn)
    return AgreementReport(
        n=n,
        accuracy=observed,
        kappa=kappa,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion=matrix,
        ci=_bootstrap(pairs, bootstrap, seed) if bootstrap else None,
    )


def _confusion(pairs: list[tuple[bool, bool]]) -> ConfusionMatrix:
    matrix = ConfusionMatrix()
    for predicted, actual in pairs:
        if predicted and actual:
            matrix.tp += 1
        elif predicted:
            matrix.fp += 1
        elif actual:
            matrix.fn += 1
        else:
            matrix.tn += 1
    return matrix


def _accuracy_kappa(matrix: ConfusionMatrix) -> tuple[float, float | None]:
    n = matrix.total
    observed = (matrix.tp + matrix.tn) / n
    judge_pass = (matrix.tp + matrix.fp) / n
    human_pass = (matrix.tp + matrix.fn) / n
    expected = judge_pass * human_pass + (1 - judge_pass) * (1 - human_pass)
    kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    return observed, kappa


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile ``q`` in [0, 1] of a sorted, non-empty list."""
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _interval(values: list[float]) -> Interval:
    if not values:
        return Interval(low=None, high=None)
    ordered = sorted(values)
    return Interval(
        low=_percentile(ordered, (1 - CI_LEVEL) / 2),
        high=_percentile(ordered, 1 - (1 - CI_LEVEL) / 2),
    )


def _bootstrap(
    pairs: list[tuple[bool, bool]], resamples: int, seed: int
) -> ConfidenceIntervals:
    rng = random.Random(seed)
    accuracies: list[float] = []
    kappas: list[float] = []
    for _ in range(resamples):
        sample = rng.choices(pairs, k=len(pairs))
        accuracy, kappa = _accuracy_kappa(_confusion(sample))
        accuracies.append(accuracy)
        if kappa is not None:
            kappas.append(kappa)
    return ConfidenceIntervals(
        level=CI_LEVEL,
        resamples=resamples,
        seed=seed,
        accuracy=_interval(accuracies),
        kappa=_interval(kappas),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator
