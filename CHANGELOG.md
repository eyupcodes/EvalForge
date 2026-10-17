# Changelog

## Unreleased

- Project skeleton: `Case` and `Verdict` models, CI, packaging.
- Deterministic judges: exact match, contains-all, regex, token F1 (#1).
- LLM-as-judge with strict JSON verdicts and OpenAI-compatible client (#2).
- Judge reliability: accuracy, Cohen's kappa and confusion matrix against human labels (#7).
- `evalforge run` and `evalforge agreement` CLI commands over JSONL datasets (#9).
