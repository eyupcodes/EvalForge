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
