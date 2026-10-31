"""Command line interface: run judges over JSONL datasets and measure agreement."""

from __future__ import annotations

import re
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import httpx
import typer

from evalforge.agreement import AgreementReport, agreement
from evalforge.io import (
    JsonlError,
    read_cases,
    read_labels,
    read_verdicts,
    write_verdicts,
)
from evalforge.judges import (
    ContainsAllJudge,
    ExactMatchJudge,
    Judge,
    RegexJudge,
    TokenF1Judge,
)
from evalforge.judges.llm import JudgeClient, LLMJudge, OpenAICompatibleClient

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"

app = typer.Typer(
    help="Run EvalForge judges over JSONL datasets.",
    no_args_is_help=True,
    add_completion=False,
)


class JudgeName(StrEnum):
    exact_match = "exact_match"
    contains_all = "contains_all"
    regex = "regex"
    token_f1 = "token_f1"
    llm = "llm"


def fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def build_llm_client(model: str, base_url: str, api_key_env: str) -> JudgeClient:
    return OpenAICompatibleClient(
        model=model, base_url=base_url, api_key_env=api_key_env
    )


def build_judge(
    name: JudgeName,
    threshold: float | None,
    phrases: list[str],
    pattern: str | None,
    rubric: str | None = None,
    model: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    api_key_env: str = DEFAULT_API_KEY_ENV,
) -> Judge:
    if name is JudgeName.exact_match:
        return ExactMatchJudge()
    if name is JudgeName.llm:
        if not rubric:
            fail("--judge llm needs --rubric")
        if not model:
            fail("--judge llm needs --model")
        client = build_llm_client(model, base_url, api_key_env)
        return LLMJudge(client, rubric, 0.5 if threshold is None else threshold)
    if name is JudgeName.token_f1:
        return TokenF1Judge() if threshold is None else TokenF1Judge(threshold)
    if name is JudgeName.contains_all:
        if not phrases:
            fail("--judge contains_all needs at least one --phrase")
        return ContainsAllJudge(phrases, 1.0 if threshold is None else threshold)
    if pattern is None:
        fail("--judge regex needs --pattern")
    try:
        return RegexJudge(pattern)
    except re.error as exc:
        fail(f"invalid --pattern: {exc}")


@app.command()
def run(
    cases: Annotated[Path, typer.Argument(help="JSONL file of cases.")],
    judge: Annotated[JudgeName, typer.Option(help="Judge to run.")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write verdicts here.")
    ] = None,
    threshold: Annotated[
        float | None,
        typer.Option(
            min=0.0, max=1.0, help="Pass threshold (token_f1, contains_all, llm)."
        ),
    ] = None,
    phrase: Annotated[
        list[str] | None, typer.Option(help="Required phrase (contains_all).")
    ] = None,
    pattern: Annotated[str | None, typer.Option(help="Regex (regex judge).")] = None,
    rubric: Annotated[
        str | None, typer.Option(help="Grading rubric (llm judge).")
    ] = None,
    model: Annotated[str | None, typer.Option(help="Judge model (llm judge).")] = None,
    base_url: Annotated[
        str, typer.Option(help="OpenAI-compatible endpoint (llm judge).")
    ] = DEFAULT_BASE_URL,
    api_key_env: Annotated[
        str, typer.Option(help="Environment variable holding the API key (llm judge).")
    ] = DEFAULT_API_KEY_ENV,
    min_pass_rate: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Exit 1 if the share of passed cases is below this value.",
        ),
    ] = None,
) -> None:
    """Judge every case and write one verdict per line (stdout by default)."""
    selected = build_judge(
        judge, threshold, phrase or [], pattern, rubric, model, base_url, api_key_env
    )
    try:
        loaded = read_cases(cases)
        verdicts = [selected.judge(case) for case in loaded]
        if output is None:
            for verdict in verdicts:
                sys.stdout.write(verdict.model_dump_json() + "\n")
        else:
            write_verdicts(verdicts, output)
    except (JsonlError, ValueError, OSError, RuntimeError, httpx.HTTPError) as exc:
        fail(str(exc))
    passed = sum(v.passed for v in verdicts)
    typer.echo(f"{passed}/{len(verdicts)} passed ({selected.name})", err=True)
    if min_pass_rate is not None and passed / len(verdicts) < min_pass_rate:
        fail(
            f"pass rate {passed / len(verdicts):.3f} is below "
            f"--min-pass-rate {min_pass_rate}"
        )


@app.command("agreement")
def agreement_command(
    verdicts: Annotated[Path, typer.Argument(help="Verdict JSONL from `run`.")],
    labels: Annotated[Path, typer.Argument(help="JSONL of case_id/human_pass.")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the report as one JSON object.")
    ] = False,
    min_kappa: Annotated[
        float | None,
        typer.Option(
            min=-1.0,
            max=1.0,
            help="Exit 1 if kappa is below this value or undefined.",
        ),
    ] = None,
) -> None:
    """Compare judge verdicts with human pass/fail labels."""
    try:
        report = agreement(read_verdicts(verdicts), read_labels(labels))
    except (JsonlError, ValueError) as exc:
        fail(str(exc))
    if as_json:
        typer.echo(report.model_dump_json())
    else:
        print_report(report)
    if min_kappa is not None:
        if report.kappa is None:
            fail(f"kappa is undefined, cannot demonstrate --min-kappa {min_kappa}")
        if report.kappa < min_kappa:
            fail(f"kappa {report.kappa:.3f} is below --min-kappa {min_kappa}")


def print_report(report: AgreementReport) -> None:

    def show(value: float | None) -> str:
        return "undefined" if value is None else f"{value:.3f}"

    matrix = report.confusion
    typer.echo(f"cases:    {report.n}")
    typer.echo(f"accuracy: {report.accuracy:.3f}")
    typer.echo(f"kappa:    {show(report.kappa)}")
    typer.echo(f"precision: {show(report.precision)}")
    typer.echo(f"recall:    {show(report.recall)}")
    typer.echo(f"f1:        {show(report.f1)}")
    typer.echo(f"tp={matrix.tp} fp={matrix.fp} fn={matrix.fn} tn={matrix.tn}")
