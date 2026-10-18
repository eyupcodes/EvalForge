"""Command line interface: run judges over JSONL datasets and measure agreement."""

from __future__ import annotations

import re
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

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


def fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


def build_judge(
    name: JudgeName, threshold: float | None, phrases: list[str], pattern: str | None
) -> Judge:
    if name is JudgeName.exact_match:
        return ExactMatchJudge()
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
        typer.Option(min=0.0, max=1.0, help="Pass threshold (token_f1, contains_all)."),
    ] = None,
    phrase: Annotated[
        list[str] | None, typer.Option(help="Required phrase (contains_all).")
    ] = None,
    pattern: Annotated[str | None, typer.Option(help="Regex (regex judge).")] = None,
) -> None:
    """Judge every case and write one verdict per line (stdout by default)."""
    selected = build_judge(judge, threshold, phrase or [], pattern)
    try:
        loaded = read_cases(cases)
        verdicts = [selected.judge(case) for case in loaded]
        if output is None:
            for verdict in verdicts:
                sys.stdout.write(verdict.model_dump_json() + "\n")
        else:
            write_verdicts(verdicts, output)
    except (JsonlError, ValueError, OSError) as exc:
        fail(str(exc))
    passed = sum(v.passed for v in verdicts)
    typer.echo(f"{passed}/{len(verdicts)} passed ({selected.name})", err=True)


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
    kappa = "undefined" if report.kappa is None else f"{report.kappa:.3f}"
    matrix = report.confusion
    typer.echo(f"cases:    {report.n}")
    typer.echo(f"accuracy: {report.accuracy:.3f}")
    typer.echo(f"kappa:    {kappa}")
    typer.echo(f"tp={matrix.tp} fp={matrix.fp} fn={matrix.fn} tn={matrix.tn}")
