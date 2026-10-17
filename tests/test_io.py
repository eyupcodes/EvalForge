import json

import pytest

from evalforge.io import (
    JsonlError,
    read_cases,
    read_labels,
    read_verdicts,
    write_verdicts,
)
from evalforge.models import Verdict


def write(path, *lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_read_cases_skips_blank_lines(tmp_path):
    path = write(
        tmp_path / "c.jsonl",
        '{"id": "a", "input": "q", "output": "x"}',
        "",
        '{"id": "b", "input": "q", "output": "y", "reference": "y"}',
    )
    cases = read_cases(path)
    assert [c.id for c in cases] == ["a", "b"]
    assert cases[1].reference == "y"


def test_read_cases_reports_invalid_json_with_line(tmp_path):
    path = write(tmp_path / "c.jsonl", '{"id": "a", "input": "q", "output": "x"}', "{")
    with pytest.raises(JsonlError, match=r"c\.jsonl:2: invalid JSON"):
        read_cases(path)


def test_read_cases_reports_validation_error_with_field(tmp_path):
    path = write(tmp_path / "c.jsonl", '{"id": "a", "input": "q"}')
    with pytest.raises(JsonlError, match=r"c\.jsonl:1: invalid Case \(output:"):
        read_cases(path)


def test_read_cases_rejects_duplicate_ids(tmp_path):
    row = '{"id": "a", "input": "q", "output": "x"}'
    with pytest.raises(JsonlError, match="duplicate case id 'a'"):
        read_cases(write(tmp_path / "c.jsonl", row, row))


def test_read_cases_rejects_empty_file(tmp_path):
    with pytest.raises(JsonlError, match="no cases found"):
        read_cases(write(tmp_path / "c.jsonl", ""))


def test_read_cases_missing_file(tmp_path):
    with pytest.raises(JsonlError, match="cannot read"):
        read_cases(tmp_path / "nope.jsonl")


def test_verdicts_round_trip(tmp_path):
    verdicts = [
        Verdict(case_id="a", judge="regex", score=1.0, passed=True, reason="ok"),
        Verdict(case_id="b", judge="regex", score=0.0, passed=False),
    ]
    path = tmp_path / "v.jsonl"
    write_verdicts(verdicts, path)
    assert read_verdicts(path) == verdicts


def test_read_labels(tmp_path):
    path = write(
        tmp_path / "l.jsonl",
        '{"case_id": "a", "human_pass": true}',
        '{"case_id": "b", "human_pass": false}',
    )
    assert read_labels(path) == {"a": True, "b": False}


@pytest.mark.parametrize(
    "row",
    [
        '{"case_id": "a"}',
        '{"case_id": "a", "human_pass": "yes"}',
        '{"case_id": 1, "human_pass": true}',
        "[1, 2]",
    ],
)
def test_read_labels_rejects_malformed_rows(tmp_path, row):
    with pytest.raises(JsonlError, match=r"l\.jsonl:1: expected"):
        read_labels(write(tmp_path / "l.jsonl", row))


def test_read_labels_rejects_duplicates(tmp_path):
    row = json.dumps({"case_id": "a", "human_pass": True})
    with pytest.raises(JsonlError, match="duplicate label for case 'a'"):
        read_labels(write(tmp_path / "l.jsonl", row, row))
