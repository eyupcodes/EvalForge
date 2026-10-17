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
