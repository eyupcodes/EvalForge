# EvalForge

Evaluate LLM outputs with **deterministic checks** and **LLM-as-judge** scoring — and
measure how much you can trust the judge itself.

Most evaluation setups stop at "the judge gave it 8/10". EvalForge treats the judge as
another system under test: every judge produces structured verdicts, and those verdicts
can be compared against human labels before you rely on them.

## Status

Early development. Planned for the first milestone:

- Deterministic judges: exact match, contains-all, regex, token F1
- LLM-as-judge with a strict JSON verdict contract and any OpenAI-compatible endpoint
- Judge reliability: agreement with human labels (Cohen's kappa, accuracy)
- `evalforge` CLI over JSONL datasets

## Deterministic judges

Reference-based judges run without model calls and return structured verdicts:

```python
from evalforge import Case
from evalforge.judges import ContainsAllJudge, TokenF1Judge

case = Case(
    id="q1",
    input="Who wrote Les Misérables?",
    output="It was written by Victor Hugo.",
    reference="Victor Hugo",
)

TokenF1Judge(threshold=0.5).judge(case)
# Verdict(case_id='q1', judge='token_f1', score=0.5, passed=True, ...)

ContainsAllJudge(phrases=["Victor Hugo"]).judge(case).passed  # True
```

| Judge | Passes when | Needs reference |
|---|---|---|
| `ExactMatchJudge` | output equals reference (normalized by default) | yes |
| `ContainsAllJudge` | share of required phrases ≥ threshold | no |
| `RegexJudge` | pattern found in output | no |
| `TokenF1Judge` | SQuAD-style token F1 ≥ threshold | yes |

Judges that need a reference raise instead of scoring a case without one, so a missing
label can never turn into a silent zero.

## LLM-as-judge

`LLMJudge` sends a rubric, the question, optional context and reference, and the
response to any OpenAI-compatible endpoint, and accepts only a strict JSON verdict:

```python
from evalforge.judges.llm import LLMJudge, OpenAICompatibleClient

client = OpenAICompatibleClient(model="gpt-4o-mini")  # reads OPENAI_API_KEY
judge = LLMJudge(client=client, rubric="Is the answer factually correct?")
verdict = judge.judge(case)
```

A reply that is not a JSON object with a numeric `score` in `[0, 1]` is retried
(`max_attempts`, default 2) and then raises `JudgeOutputError`. Unparseable judge
output is never counted as a low score. Any object with a
`complete(system, user) -> str` method can replace the HTTP client, which keeps tests
and local models simple.

## Development

Requires Python 3.11+.

```bash
python -m venv .venv
.venv/Scripts/activate   # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
ruff format --check . && ruff check . && pytest --cov
```

## License

MIT
