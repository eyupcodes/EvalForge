"""Read and write the JSONL files used by the command line interface."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from evalforge.models import Case, Verdict

_T = TypeVar("_T", bound=BaseModel)


class JsonlError(ValueError):
    """A JSONL file could not be read; the message names the file and line."""


def _rows(path: Path) -> Iterator[tuple[int, object]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise JsonlError(f"cannot read {path}: {exc.strerror or exc}") from exc
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield number, json.loads(line)
        except json.JSONDecodeError as exc:
            raise JsonlError(f"{path}:{number}: invalid JSON ({exc.msg})") from exc


def _read_models(path: Path, model: type[_T]) -> list[_T]:
    items: list[_T] = []
    for number, row in _rows(path):
        try:
            items.append(model.model_validate(row))
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(part) for part in first["loc"]) or "row"
            raise JsonlError(
                f"{path}:{number}: invalid {model.__name__} ({where}: {first['msg']})"
            ) from exc
    return items


def read_cases(path: Path) -> list[Case]:
    """Load cases, rejecting empty files and duplicate case ids."""
    cases = _read_models(path, Case)
    if not cases:
        raise JsonlError(f"{path}: no cases found")
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise JsonlError(f"{path}: duplicate case id {case.id!r}")
        seen.add(case.id)
    return cases


def read_verdicts(path: Path) -> list[Verdict]:
    """Load verdicts written by ``evalforge run``."""
    return _read_models(path, Verdict)


def read_labels(path: Path) -> dict[str, bool]:
    """Load human labels from rows like ``{"case_id": "q1", "human_pass": true}``."""
    labels: dict[str, bool] = {}
    for number, row in _rows(path):
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("case_id"), str)
            or not isinstance(row.get("human_pass"), bool)
        ):
            raise JsonlError(
                f'{path}:{number}: expected {{"case_id": str, "human_pass": bool}}'
            )
        if row["case_id"] in labels:
            raise JsonlError(f"{path}: duplicate label for case {row['case_id']!r}")
        labels[row["case_id"]] = row["human_pass"]
    return labels


def write_verdicts(verdicts: Iterable[Verdict], path: Path) -> None:
    """Write verdicts as one JSON object per line."""
    with path.open("w", encoding="utf-8") as handle:
        for verdict in verdicts:
            handle.write(verdict.model_dump_json() + "\n")
