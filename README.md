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
