"""LLM-as-judge with a strict JSON verdict contract."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import httpx

from evalforge.models import Case, Verdict

SYSTEM_PROMPT = (
    "You are a strict evaluator. Score the RESPONSE against the RUBRIC. "
    "Reply with only a JSON object: "
    '{"score": <number between 0 and 1>, "reason": "<one sentence>"}.'
)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

_FENCED = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class JudgeOutputError(ValueError):
    """The judge model did not return a valid verdict."""


class JudgeClient(Protocol):
    """Anything that turns a system and user prompt into a completion."""

    def complete(self, system: str, user: str) -> str: ...


@dataclass
class OpenAICompatibleClient:
    """Minimal client for any OpenAI-compatible /chat/completions endpoint."""

    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout: float = 60.0
    transport: httpx.BaseTransport | None = None
    max_retries: int = 3
    backoff: float = 1.0
    sleep: Callable[[float], None] = time.sleep

    def complete(self, system: str, user: str) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"set {self.api_key_env} to call {self.base_url}")
        with httpx.Client(timeout=self.timeout, transport=self.transport) as http:
            for attempt in range(self.max_retries + 1):
                try:
                    response = http.post(
                        f"{self.base_url.rstrip('/')}/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": self.model,
                            "temperature": 0,
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                        },
                    )
                except httpx.TransportError:
                    if attempt == self.max_retries:
                        raise
                    self.sleep(self._delay(attempt, None))
                    continue
                if (
                    response.status_code in RETRY_STATUSES
                    and attempt < self.max_retries
                ):
                    self.sleep(self._delay(attempt, response))
                    continue
                response.raise_for_status()
                return self._content(response)
        raise AssertionError("unreachable")  # pragma: no cover

    def _delay(self, attempt: int, response: httpx.Response | None) -> float:
        """Seconds to wait before retrying: ``Retry-After`` if numeric, else backoff."""
        if response is not None:
            try:
                return max(0.0, float(response.headers["Retry-After"]))
            except (KeyError, ValueError):
                pass
        return self.backoff * 2**attempt

    def _content(self, response: httpx.Response) -> str:
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, LookupError, TypeError) as exc:
            raise RuntimeError(
                f"unexpected response from {self.base_url}: no completion in reply"
            ) from exc
        if not isinstance(content, str):
            raise RuntimeError(
                f"unexpected response from {self.base_url}: empty completion"
            )
        return content


def build_prompt(case: Case, rubric: str) -> str:
    """Render the user prompt the judge model sees."""
    parts = [f"RUBRIC:\n{rubric}", f"QUESTION:\n{case.input}"]
    if case.context:
        parts.append("CONTEXT:\n" + "\n---\n".join(case.context))
    if case.reference is not None:
        parts.append(f"REFERENCE ANSWER:\n{case.reference}")
    parts.append(f"RESPONSE:\n{case.output}")
    return "\n\n".join(parts)


def parse_verdict(raw: str) -> tuple[float, str]:
    """Extract (score, reason) from a model reply or raise JudgeOutputError."""
    fenced = _FENCED.search(raw)
    text = fenced.group(1) if fenced else raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeOutputError(f"judge reply is not JSON: {raw[:120]!r}") from exc
    if not isinstance(data, dict):
        raise JudgeOutputError("judge reply must be a JSON object")
    score = data.get("score")
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise JudgeOutputError(f"score must be a number, got {score!r}")
    if not 0 <= score <= 1:
        raise JudgeOutputError(f"score must be within [0, 1], got {score}")
    return float(score), str(data.get("reason", ""))


@dataclass
class LLMJudge:
    """Scores a case with a judge model; malformed replies are retried, then raised."""

    client: JudgeClient
    rubric: str
    threshold: float = 0.5
    max_attempts: int = 2
    name: str = "llm_judge"

    def judge(self, case: Case) -> Verdict:
        prompt = build_prompt(case, self.rubric)
        last_error: JudgeOutputError | None = None
        for _ in range(self.max_attempts):
            try:
                score, reason = parse_verdict(
                    self.client.complete(SYSTEM_PROMPT, prompt)
                )
            except JudgeOutputError as exc:
                last_error = exc
                continue
            return Verdict(
                case_id=case.id,
                judge=self.name,
                score=score,
                passed=score >= self.threshold,
                reason=reason,
            )
        raise JudgeOutputError(
            f"{self.name} failed on case {case.id!r} after {self.max_attempts} "
            f"attempts: {last_error}"
        )
