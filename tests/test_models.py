import pytest
from pydantic import ValidationError

from evalforge import Case, Verdict


def test_case_defaults_to_empty_context() -> None:
    case = Case(id="c1", input="2+2?", output="4")

    assert case.reference is None
    assert case.context == []


def test_verdict_rejects_score_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        Verdict(case_id="c1", judge="exact", score=1.5, passed=True)
