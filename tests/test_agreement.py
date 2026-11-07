import pytest

from evalforge.agreement import agreement
from evalforge.models import Verdict


def _verdict(case_id: str, passed: bool) -> Verdict:
    return Verdict(
        case_id=case_id, judge="j", score=1.0 if passed else 0.0, passed=passed
    )


def test_perfect_agreement() -> None:
    verdicts = [_verdict("a", True), _verdict("b", False)]
    report = agreement(verdicts, {"a": True, "b": False})

    assert report.n == 2
    assert report.accuracy == 1.0
    assert report.kappa == 1.0
    assert (report.confusion.tp, report.confusion.tn) == (1, 1)


def test_known_kappa_value() -> None:
    # 10 cases: tp=4, fp=1, fn=2, tn=3 -> po=0.7, pe=0.5*0.6+0.5*0.4=0.5, k=0.4
    pairs = [(True, True)] * 4 + [(True, False)] + [(False, True)] * 2
    pairs += [(False, False)] * 3
    verdicts = [_verdict(str(i), judge) for i, (judge, _) in enumerate(pairs)]
    labels = {str(i): human for i, (_, human) in enumerate(pairs)}

    report = agreement(verdicts, labels)

    assert report.accuracy == pytest.approx(0.7)
    assert report.kappa == pytest.approx(0.4)
    assert report.confusion.model_dump() == {"tp": 4, "fp": 1, "fn": 2, "tn": 3}


def test_kappa_is_none_when_chance_agreement_is_total() -> None:
    verdicts = [_verdict("a", True), _verdict("b", True)]
    report = agreement(verdicts, {"a": True, "b": True})

    assert report.accuracy == 1.0
    assert report.kappa is None


def test_systematic_disagreement_gives_negative_kappa() -> None:
    verdicts = [_verdict("a", True), _verdict("b", False)]
    report = agreement(verdicts, {"a": False, "b": True})

    assert report.accuracy == 0.0
    assert report.kappa == -1.0


def test_mismatched_ids_raise() -> None:
    with pytest.raises(ValueError, match="differ"):
        agreement([_verdict("a", True)], {"b": True})


def test_duplicate_verdicts_raise() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        agreement([_verdict("a", True), _verdict("a", False)], {"a": True})


def test_empty_verdicts_raise() -> None:
    with pytest.raises(ValueError, match="no verdicts"):
        agreement([], {})


def test_precision_recall_f1_known_values() -> None:
    # tp=4, fp=1, fn=2, tn=3 -> precision 4/5, recall 4/6, f1 8/11
    pairs = [(True, True)] * 4 + [(True, False)] + [(False, True)] * 2
    pairs += [(False, False)] * 3
    verdicts = [_verdict(str(i), judge) for i, (judge, _) in enumerate(pairs)]
    labels = {str(i): human for i, (_, human) in enumerate(pairs)}

    report = agreement(verdicts, labels)

    assert report.precision == pytest.approx(0.8)
    assert report.recall == pytest.approx(4 / 6)
    assert report.f1 == pytest.approx(8 / 11)


def test_precision_is_none_when_judge_never_passes() -> None:
    report = agreement(
        [_verdict("a", False), _verdict("b", False)], {"a": True, "b": False}
    )

    assert report.precision is None
    assert report.recall == 0.0
    assert report.f1 == 0.0


def test_recall_is_none_when_humans_never_pass() -> None:
    report = agreement(
        [_verdict("a", True), _verdict("b", False)], {"a": False, "b": False}
    )

    assert report.recall is None
    assert report.precision == 0.0
    assert report.f1 == 0.0


def test_all_metrics_none_without_any_pass() -> None:
    report = agreement([_verdict("a", False)], {"a": False})

    assert (report.precision, report.recall, report.f1) == (None, None, None)


def _mixed_verdicts():
    pairs = [(True, True)] * 12 + [(True, False)] * 3 + [(False, True)] * 4
    pairs += [(False, False)] * 11
    verdicts = [_verdict(f"c{i}", p) for i, (p, _) in enumerate(pairs)]
    labels = {f"c{i}": h for i, (_, h) in enumerate(pairs)}
    return verdicts, labels


def test_no_intervals_by_default() -> None:
    verdicts, labels = _mixed_verdicts()

    assert agreement(verdicts, labels).ci is None


def test_bootstrap_intervals_bracket_the_estimate_and_are_seeded() -> None:
    verdicts, labels = _mixed_verdicts()

    report = agreement(verdicts, labels, bootstrap=500, seed=3)
    again = agreement(verdicts, labels, bootstrap=500, seed=3)
    other = agreement(verdicts, labels, bootstrap=500, seed=4)

    ci = report.ci
    assert ci is not None
    assert (ci.level, ci.resamples, ci.seed) == (0.95, 500, 3)
    assert ci.accuracy.low < report.accuracy < ci.accuracy.high
    assert ci.kappa.low < report.kappa < ci.kappa.high
    assert ci.accuracy.low >= 0.0 and ci.accuracy.high <= 1.0
    assert again.ci == ci
    assert other.ci != ci


def test_bootstrap_with_a_single_case_has_degenerate_kappa() -> None:
    report = agreement([_verdict("a", True)], {"a": True}, bootstrap=20)

    assert report.ci is not None
    assert (report.ci.accuracy.low, report.ci.accuracy.high) == (1.0, 1.0)
    assert (report.ci.kappa.low, report.ci.kappa.high) == (None, None)


def test_bootstrap_ignores_undefined_kappa_resamples() -> None:
    # Two cases: resamples that draw the same case twice have undefined kappa.
    verdicts = [_verdict("a", True), _verdict("b", False)]

    report = agreement(verdicts, {"a": True, "b": False}, bootstrap=200, seed=1)

    assert report.ci is not None
    assert report.ci.kappa.low == 1.0 and report.ci.kappa.high == 1.0


def test_negative_bootstrap_is_rejected() -> None:
    with pytest.raises(ValueError, match="bootstrap"):
        agreement([_verdict("a", True)], {"a": True}, bootstrap=-1)
