import pytest

from evalforge import Case
from evalforge.judges import (
    ContainsAllJudge,
    ExactMatchJudge,
    RegexJudge,
    TokenF1Judge,
)
from evalforge.judges.deterministic import normalize


def make_case(output: str, reference: str | None = "Paris") -> Case:
    return Case(id="c1", input="Capital of France?", output=output, reference=reference)


def test_normalize_strips_case_punctuation_and_spacing() -> None:
    assert normalize("  Hello,   WORLD! ") == "hello world"


def test_exact_match_ignores_case_and_punctuation_by_default() -> None:
    verdict = ExactMatchJudge().judge(make_case("paris."))

    assert verdict.passed
    assert verdict.score == 1.0


def test_exact_match_strict_mode_is_case_sensitive() -> None:
    verdict = ExactMatchJudge(normalized=False).judge(make_case("paris"))

    assert not verdict.passed
    assert verdict.reason == "output differs"


def test_reference_judges_refuse_cases_without_reference() -> None:
    case = make_case("Paris", reference=None)

    with pytest.raises(ValueError, match="needs a reference"):
        ExactMatchJudge().judge(case)
    with pytest.raises(ValueError, match="needs a reference"):
        TokenF1Judge().judge(case)


def test_contains_all_scores_partial_coverage() -> None:
    judge = ContainsAllJudge(phrases=["Paris", "Seine"])

    verdict = judge.judge(make_case("Paris is the capital."))

    assert verdict.score == 0.5
    assert not verdict.passed
    assert "Seine" in verdict.reason


def test_contains_all_threshold_allows_partial_pass() -> None:
    judge = ContainsAllJudge(phrases=["Paris", "Seine"], threshold=0.5)

    assert judge.judge(make_case("Paris")).passed


def test_contains_all_requires_phrases() -> None:
    with pytest.raises(ValueError, match="at least one phrase"):
        ContainsAllJudge(phrases=[]).judge(make_case("Paris"))


def test_regex_judge_matches_case_insensitively() -> None:
    judge = RegexJudge(pattern=r"\bparis\b")

    assert judge.judge(make_case("It is PARIS.")).passed
    assert not judge.judge(make_case("It is Lyon.")).passed


def test_token_f1_gives_partial_credit() -> None:
    case = Case(
        id="c2",
        input="Who wrote it?",
        output="Victor Hugo wrote it",
        reference="Victor Hugo",
    )

    verdict = TokenF1Judge().judge(case)

    assert verdict.score == pytest.approx(0.6667, abs=1e-4)
    assert verdict.passed


def test_token_f1_is_zero_without_overlap() -> None:
    verdict = TokenF1Judge().judge(make_case("Berlin"))

    assert verdict.score == 0.0
    assert not verdict.passed


@pytest.mark.parametrize(
    ("output", "reference"),
    [
        ("Paris Paris", "Paris"),
        ("Paris", "Paris Paris"),
        ("Paris Paris Lyon", "Paris Lyon Lyon"),
    ],
)
def test_token_f1_counts_repeated_tokens(output: str, reference: str) -> None:
    verdict = TokenF1Judge().judge(make_case(output, reference))

    assert verdict.score == pytest.approx(0.6667, abs=1e-4)
    assert verdict.passed


@pytest.mark.parametrize(
    ("output", "reference"),
    [("", "Paris"), ("Paris", ""), ("", ""), (" !? ", " ... ")],
)
def test_token_f1_is_zero_for_empty_normalized_answers(
    output: str, reference: str
) -> None:
    verdict = TokenF1Judge().judge(make_case(output, reference))

    assert verdict.score == 0.0
    assert not verdict.passed


@pytest.mark.parametrize(("threshold", "passed"), [(2 / 3, True), (0.66668, False)])
def test_token_f1_compares_threshold_before_rounding(
    threshold: float, passed: bool
) -> None:
    verdict = TokenF1Judge(threshold=threshold).judge(make_case("Paris Paris"))

    assert verdict.score == 0.6667
    assert verdict.passed is passed
