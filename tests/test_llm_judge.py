import json

import httpx
import pytest

from evalforge import Case
from evalforge.judges.llm import (
    JudgeOutputError,
    LLMJudge,
    OpenAICompatibleClient,
    build_prompt,
    parse_verdict,
)

CASE = Case(
    id="c1",
    input="Capital of France?",
    output="Paris",
    reference="Paris",
    context=["France is in Europe."],
)


class ScriptedClient:
    """Returns canned replies in order and records the prompts it saw."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        return self.replies.pop(0)


def test_build_prompt_includes_every_available_field() -> None:
    prompt = build_prompt(CASE, "Is it correct?")

    for expected in ["RUBRIC", "QUESTION", "CONTEXT", "REFERENCE ANSWER", "RESPONSE"]:
        assert expected in prompt


def test_build_prompt_omits_missing_reference_and_context() -> None:
    prompt = build_prompt(Case(id="c", input="q", output="a"), "r")

    assert "REFERENCE" not in prompt
    assert "CONTEXT" not in prompt


def test_parse_verdict_accepts_fenced_json() -> None:
    raw = 'Sure.\n```json\n{"score": 0.8, "reason": "close"}\n```'

    assert parse_verdict(raw) == (0.8, "close")


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[1, 2]",
        '{"score": "high"}',
        '{"score": true}',
        '{"score": 1.4}',
    ],
)
def test_parse_verdict_rejects_malformed_replies(raw: str) -> None:
    with pytest.raises(JudgeOutputError):
        parse_verdict(raw)


def test_llm_judge_returns_verdict_from_model_reply() -> None:
    client = ScriptedClient('{"score": 0.9, "reason": "correct"}')

    verdict = LLMJudge(client=client, rubric="Is it correct?").judge(CASE)

    assert verdict.passed
    assert verdict.score == 0.9
    assert verdict.reason == "correct"
    assert "Paris" in client.prompts[0][1]


def test_llm_judge_retries_once_after_malformed_reply() -> None:
    client = ScriptedClient("oops", '{"score": 0.2, "reason": "wrong"}')

    verdict = LLMJudge(client=client, rubric="r").judge(CASE)

    assert not verdict.passed
    assert len(client.prompts) == 2


def test_llm_judge_raises_after_exhausting_attempts() -> None:
    client = ScriptedClient("oops", "still oops")

    with pytest.raises(JudgeOutputError, match="after 2 attempts"):
        LLMJudge(client=client, rubric="r").judge(CASE)


def test_openai_client_posts_chat_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        reply = {"choices": [{"message": {"content": '{"score": 1}'}}]}
        return httpx.Response(200, json=reply)

    monkeypatch.setenv("TEST_KEY", "secret")
    client = OpenAICompatibleClient(
        model="m",
        base_url="https://llm.example/v1/",
        api_key_env="TEST_KEY",
        transport=httpx.MockTransport(handler),
    )

    assert client.complete("sys", "usr") == '{"score": 1}'
    assert seen["url"] == "https://llm.example/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["temperature"] == 0


def test_openai_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_KEY", raising=False)
    client = OpenAICompatibleClient(model="m", api_key_env="MISSING_KEY")

    with pytest.raises(RuntimeError, match="MISSING_KEY"):
        client.complete("s", "u")
