import json

import httpx
import pytest
from typer.testing import CliRunner

from evalforge.cli import app
from evalforge.judges.llm import OpenAICompatibleClient

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


def test_min_pass_rate_passes_when_met_and_still_writes_verdicts(cases_file, tmp_path):
    out = tmp_path / "verdicts.jsonl"
    result = runner.invoke(
        app,
        ["run", str(cases_file), "--judge", "exact_match", "-o", str(out)]
        + ["--min-pass-rate", "0.5"],
    )
    assert result.exit_code == 0, result.output
    assert len(out.read_text().splitlines()) == 2


def test_min_pass_rate_fails_when_below_threshold(cases_file, tmp_path):
    out = tmp_path / "verdicts.jsonl"
    result = runner.invoke(
        app,
        ["run", str(cases_file), "--judge", "exact_match", "-o", str(out)]
        + ["--min-pass-rate", "0.9"],
    )
    assert result.exit_code == 1
    assert "1/2 passed" in result.output
    assert "pass rate 0.500 is below --min-pass-rate 0.9" in result.output
    assert len(out.read_text().splitlines()) == 2


def test_min_pass_rate_rejects_out_of_range_value(cases_file):
    result = runner.invoke(
        app, ["run", str(cases_file), "--judge", "exact_match", "--min-pass-rate", "2"]
    )
    assert result.exit_code == 2


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


def test_run_and_agreement_preserve_unicode_line_separators(tmp_path):
    case_id = f"q{chr(0x2028)}1"
    answer = f"first{chr(0x2029)}second"
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {"id": case_id, "input": "?", "output": answer, "reference": answer},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    verdicts = tmp_path / "verdicts.jsonl"
    run = runner.invoke(
        app, ["run", str(cases), "--judge", "exact_match", "-o", str(verdicts)]
    )
    assert run.exit_code == 0, run.output
    assert json.loads(verdicts.read_text(encoding="utf-8"))["case_id"] == case_id

    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"case_id": case_id, "human_pass": True}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = runner.invoke(app, ["agreement", str(verdicts), str(labels), "--json"])
    assert report.exit_code == 0, report.output
    assert json.loads(report.stdout)["accuracy"] == 1.0


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
    assert "precision: 1.000" in result.output
    assert "recall:    1.000" in result.output
    assert "f1:        1.000" in result.output
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


def test_min_kappa_passes_when_met(cases_file, tmp_path):
    verdicts = make_verdicts(cases_file, tmp_path)
    labels = write_labels(tmp_path, {"q1": True, "q2": False})
    result = runner.invoke(
        app, ["agreement", str(verdicts), str(labels), "--min-kappa", "0.6"]
    )
    assert result.exit_code == 0, result.output
    assert "kappa:    1.000" in result.output


def test_min_kappa_fails_when_below_threshold(cases_file, tmp_path):
    verdicts = make_verdicts(cases_file, tmp_path)
    labels = write_labels(tmp_path, {"q1": False, "q2": True})
    result = runner.invoke(
        app, ["agreement", str(verdicts), str(labels), "--min-kappa", "0.6"]
    )
    assert result.exit_code == 1
    assert "kappa:    -1.000" in result.output
    assert "below --min-kappa 0.6" in result.output


def test_min_kappa_fails_when_undefined(tmp_path):
    verdicts = tmp_path / "v.jsonl"
    verdicts.write_text('{"case_id": "a", "judge": "j", "score": 1, "passed": true}\n')
    labels = write_labels(tmp_path, {"a": True})
    result = runner.invoke(
        app, ["agreement", str(verdicts), str(labels), "--json", "--min-kappa", "0.1"]
    )
    assert result.exit_code == 1
    assert "undefined" in result.output


class FakeJudgeClient:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)

    def complete(self, system: str, user: str) -> str:
        return self.replies.pop(0)


@pytest.fixture
def fake_llm(monkeypatch):
    def install(*replies: str) -> list[tuple[str, str, str]]:
        built: list[tuple[str, str, str]] = []

        def factory(model: str, base_url: str, api_key_env: str):
            built.append((model, base_url, api_key_env))
            return FakeJudgeClient(*replies)

        monkeypatch.setattr("evalforge.cli.build_llm_client", factory)
        return built

    return install


def test_run_llm_judge_writes_verdicts_agreement_can_read(
    cases_file, tmp_path, fake_llm
):
    built = fake_llm('{"score": 1, "reason": "ok"}', '{"score": 0.1, "reason": "no"}')
    out = tmp_path / "verdicts.jsonl"
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        '{"case_id": "q1", "human_pass": true}\n'
        '{"case_id": "q2", "human_pass": false}\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            *["run", str(cases_file), "--judge", "llm", "-o", str(out)],
            *["--rubric", "Is it right?", "--model", "judge-1"],
            *["--base-url", "http://localhost:8000/v1", "--api-key-env", "MY_KEY"],
        ],
    )
    assert result.exit_code == 0, result.output
    assert built == [("judge-1", "http://localhost:8000/v1", "MY_KEY")]
    assert "1/2 passed (llm_judge)" in result.output

    report = runner.invoke(app, ["agreement", str(out), str(labels), "--json"])
    assert report.exit_code == 0, report.output
    assert json.loads(report.stdout)["accuracy"] == 1.0


def test_run_llm_judge_threshold(cases_file, fake_llm):
    fake_llm('{"score": 0.6}', '{"score": 0.6}')
    result = runner.invoke(
        app,
        [
            *["run", str(cases_file), "--judge", "llm", "--rubric", "r"],
            *["--model", "m", "--threshold", "0.7"],
        ],
    )
    assert result.exit_code == 0, result.output
    assert "0/2 passed" in result.output


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--model", "m"], "--rubric"),
        (["--rubric", "r"], "--model"),
    ],
)
def test_run_llm_judge_requires_rubric_and_model(cases_file, extra, message):
    result = runner.invoke(app, ["run", str(cases_file), "--judge", "llm", *extra])
    assert result.exit_code == 1
    assert message in result.output


def test_run_llm_judge_reports_unparseable_reply(cases_file, fake_llm):
    fake_llm("not json", "still not json")
    result = runner.invoke(
        app,
        ["run", str(cases_file), "--judge", "llm", "--rubric", "r", "--model", "m"],
    )
    assert result.exit_code == 1
    assert "error:" in result.output
    assert "q1" in result.output


def test_run_llm_judge_reports_missing_api_key(cases_file, monkeypatch):
    monkeypatch.delenv("EVALFORGE_TEST_KEY", raising=False)
    result = runner.invoke(
        app,
        [
            *["run", str(cases_file), "--judge", "llm", "--rubric", "r"],
            *["--model", "m", "--api-key-env", "EVALFORGE_TEST_KEY"],
        ],
    )
    assert result.exit_code == 1
    assert "EVALFORGE_TEST_KEY" in result.output
    assert "Traceback" not in result.output


def test_run_llm_judge_reports_http_failure(cases_file, monkeypatch):
    class Failing:
        def complete(self, system: str, user: str) -> str:
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("evalforge.cli.build_llm_client", lambda *args: Failing())
    result = runner.invoke(
        app,
        ["run", str(cases_file), "--judge", "llm", "--rubric", "r", "--model", "m"],
    )
    assert result.exit_code == 1
    assert "connection refused" in result.output


def test_run_llm_judge_reports_malformed_provider_response(cases_file, monkeypatch):
    def factory(model: str, base_url: str, api_key_env: str):
        return OpenAICompatibleClient(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
        )

    monkeypatch.setenv("EVALFORGE_TEST_KEY", "secret")
    monkeypatch.setattr("evalforge.cli.build_llm_client", factory)
    result = runner.invoke(
        app,
        [
            *["run", str(cases_file), "--judge", "llm", "--rubric", "r"],
            *["--model", "m", "--api-key-env", "EVALFORGE_TEST_KEY"],
        ],
    )
    assert result.exit_code == 1
    assert "unexpected response" in result.output
    assert "Traceback" not in result.output


def test_agreement_json_includes_precision_recall_f1(cases_file, tmp_path):
    verdicts = make_verdicts(cases_file, tmp_path)
    labels = write_labels(tmp_path, {"q1": True, "q2": False})
    result = runner.invoke(app, ["agreement", str(verdicts), str(labels), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert (data["precision"], data["recall"], data["f1"]) == (1.0, 1.0, 1.0)
