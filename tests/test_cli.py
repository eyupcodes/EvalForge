import json

import pytest
from typer.testing import CliRunner

from evalforge.cli import app

runner = CliRunner()

CASES = [
    {"id": "q1", "input": "?", "output": "Victor Hugo", "reference": "Victor Hugo"},
    {"id": "q2", "input": "?", "output": "Balzac", "reference": "Victor Hugo"},
]


@pytest.fixture
def cases_file(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in CASES) + "\n", encoding="utf-8")
    return path


def test_run_writes_verdicts_to_file(cases_file, tmp_path):
    out = tmp_path / "verdicts.jsonl"
    result = runner.invoke(
        app, ["run", str(cases_file), "--judge", "exact_match", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["case_id"], r["passed"]) for r in rows] == [("q1", True), ("q2", False)]
    assert "1/2 passed (exact_match)" in result.output


def test_run_prints_jsonl_to_stdout(cases_file):
    result = runner.invoke(app, ["run", str(cases_file), "--judge", "token_f1"])
    assert result.exit_code == 0, result.output
    first = json.loads(result.stdout.splitlines()[0])
    assert first["judge"] == "token_f1"
    assert first["score"] == 1.0


def test_run_threshold_changes_outcome(cases_file):
    result = runner.invoke(
        app, ["run", str(cases_file), "--judge", "token_f1", "--threshold", "0"]
    )
    assert result.exit_code == 0
    assert "2/2 passed" in result.output


def test_run_contains_all_with_phrases(cases_file):
    result = runner.invoke(
        app,
        ["run", str(cases_file), "--judge", "contains_all", "--phrase", "hugo"],
    )
    assert result.exit_code == 0
    assert "1/2 passed (contains_all)" in result.output


def test_run_regex_with_pattern(cases_file):
    result = runner.invoke(
        app, ["run", str(cases_file), "--judge", "regex", "--pattern", "^balzac$"]
    )
    assert result.exit_code == 0
    assert "1/2 passed (regex)" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--judge", "contains_all"], "needs at least one --phrase"),
        (["--judge", "regex"], "needs --pattern"),
        (["--judge", "regex", "--pattern", "("], "invalid --pattern"),
    ],
)
def test_run_rejects_bad_judge_options(cases_file, args, message):
    result = runner.invoke(app, ["run", str(cases_file), *args])
    assert result.exit_code == 1
    assert message in result.output
    assert "Traceback" not in result.output


def test_run_reports_missing_reference_without_traceback(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text('{"id": "a", "input": "q", "output": "x"}\n')
    result = runner.invoke(app, ["run", str(path), "--judge", "exact_match"])
    assert result.exit_code == 1
    assert "needs a reference answer" in result.output


def test_run_reports_bad_input_file(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text("not json\n")
    result = runner.invoke(app, ["run", str(path), "--judge", "exact_match"])
    assert result.exit_code == 1
    assert "c.jsonl:1: invalid JSON" in result.output


def test_run_reports_unwritable_output(cases_file, tmp_path):
    result = runner.invoke(
        app,
        ["run", str(cases_file), "--judge", "exact_match", "-o", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert result.output.startswith("error:")


def write_labels(tmp_path, labels):
    path = tmp_path / "labels.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"case_id": k, "human_pass": v}) for k, v in labels.items()
        )
        + "\n"
    )
    return path


def make_verdicts(cases_file, tmp_path):
    out = tmp_path / "verdicts.jsonl"
    runner.invoke(
        app, ["run", str(cases_file), "--judge", "exact_match", "-o", str(out)]
    )
    return out


def test_agreement_prints_report(cases_file, tmp_path):
    verdicts = make_verdicts(cases_file, tmp_path)
    labels = write_labels(tmp_path, {"q1": True, "q2": False})
    result = runner.invoke(app, ["agreement", str(verdicts), str(labels)])
    assert result.exit_code == 0, result.output
    assert "accuracy: 1.000" in result.output
    assert "kappa:    1.000" in result.output
    assert "tp=1 fp=0 fn=0 tn=1" in result.output


def test_agreement_reports_undefined_kappa(cases_file, tmp_path):
    verdicts = tmp_path / "v.jsonl"
    verdicts.write_text('{"case_id": "a", "judge": "j", "score": 1, "passed": true}\n')
    labels = write_labels(tmp_path, {"a": True})
    result = runner.invoke(app, ["agreement", str(verdicts), str(labels)])
    assert result.exit_code == 0
    assert "kappa:    undefined" in result.output


def test_agreement_reports_mismatched_ids(cases_file, tmp_path):
    verdicts = make_verdicts(cases_file, tmp_path)
    labels = write_labels(tmp_path, {"q1": True})
    result = runner.invoke(app, ["agreement", str(verdicts), str(labels)])
    assert result.exit_code == 1
    assert "differ" in result.output


def test_agreement_reports_missing_file(tmp_path):
    result = runner.invoke(
        app, ["agreement", str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")]
    )
    assert result.exit_code == 1
    assert "cannot read" in result.output


def test_agreement_json_output(cases_file, tmp_path):
    verdicts = make_verdicts(cases_file, tmp_path)
    labels = write_labels(tmp_path, {"q1": True, "q2": False})
    result = runner.invoke(app, ["agreement", str(verdicts), str(labels), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["n"] == 2
    assert report["accuracy"] == 1.0
    assert report["kappa"] == 1.0
    assert report["confusion"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 1}


def test_agreement_json_reports_null_kappa(tmp_path):
    verdicts = tmp_path / "v.jsonl"
    verdicts.write_text('{"case_id": "a", "judge": "j", "score": 1, "passed": true}\n')
    labels = write_labels(tmp_path, {"a": True})
    result = runner.invoke(app, ["agreement", str(verdicts), str(labels), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["kappa"] is None
