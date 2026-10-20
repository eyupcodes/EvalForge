# Changelog

## Unreleased

- Project skeleton: `Case` and `Verdict` models, CI, packaging.
- Deterministic judges: exact match, contains-all, regex, token F1 (#1).
- LLM-as-judge with strict JSON verdicts and OpenAI-compatible client (#2).
- Judge reliability: accuracy, Cohen's kappa and confusion matrix against human labels (#7).
- `evalforge run` and `evalforge agreement` CLI commands over JSONL datasets (#9).
- `evalforge agreement --json` and `--min-kappa` to script and gate on judge reliability (#11).
- `evalforge run --judge llm` with `--rubric`, `--model`, `--base-url` and `--api-key-env` runs the LLM judge from the command line (#13).
- `OpenAICompatibleClient` raises a clear error on non-JSON bodies, error objects, empty `choices` and null content, so `evalforge run --judge llm` exits 1 with a message instead of a traceback (#15).
