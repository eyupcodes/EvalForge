# Changelog

## [Unreleased]

- `evalforge run` refuses an `--output` that is the cases file (also through a symlink or a differently spelled path) instead of overwriting the dataset with verdicts (#66).
- The error for verdicts and human labels that cover different cases reports how many ids lack a label or a verdict and lists at most 5 of each, instead of printing every id (#64).
- CI also runs on Python 3.13 (classifier added) and fails when branch coverage drops below 95% (the gate was 85%) (#62).
- `evalforge run --judge regex --case-sensitive` matches the pattern case-sensitively instead of always ignoring case; the option is rejected for other judges (#60).
- `ContainsAllJudge` (and `evalforge run --judge contains_all`) rejects a required phrase that is blank or punctuation-only instead of counting it as present in every output (#58).
- `evalforge run --max-attempts N` sets the LLM judge's retry budget for malformed replies (default 2, only valid with `--judge llm`), and `LLMJudge` rejects `max_attempts < 1` instead of failing with a misleading "after 0 attempts" error (#56).
- The LLM judge client joins `message.content` returned as typed parts (`[{"type": "text", "text": ...}]`) instead of rejecting it as an empty completion, and a reply with an `error` and no completion reports the server's message (#54).
- Dependabot proposes monthly, grouped updates for GitHub Actions and pip dependencies (#51).
- `mypy` runs in strict mode; the LLM judge's retry delay is typed as a float instead of leaking `Any` (#49).
- `evalforge run` writes the verdicts judged so far before exiting 1 when a later case fails, and the error says how many cases were judged, so a failure at case 400 of 500 no longer discards the other 399 verdicts (#47).
- Deterministic judges apply NFKC normalization and case folding before comparing, so composed vs decomposed accents, `ß`/`SS`, ligatures and full-width digits no longer score as different text (#45).
- `evalforge run` checks `--output` before judging, so an unwritable path no longer discards the verdicts (and the API spend) of a finished `--judge llm` run (#43).
- `evalforge run` rejects judge options that the selected judge does not use (`--threshold`, `--phrase`, `--pattern`, `--rubric`, `--model`) instead of silently ignoring them (#39).
- `evalforge agreement --bootstrap N [--seed S]` and `agreement(..., bootstrap=N)` add seeded 95% bootstrap confidence intervals for accuracy and kappa, so a kappa from a small label set is shown with its uncertainty (#37).
- The LLM judge client honours `Retry-After` given as an HTTP date, not only as seconds (#33).
- The LLM judge accepts a verdict JSON object embedded in surrounding prose ("Here is my verdict: {...}"), so a chatty reply no longer aborts a run; replies without a valid score are still rejected (#31).
- Agreement reports include precision, recall and F1 for the "pass" class (`None` when undefined), in the Python API, the `evalforge agreement` text output and `--json` (#29).
- The LLM judge client caps `Retry-After` (and backoff) at `max_retry_delay` (60 s) and ignores non-finite values instead of sleeping indefinitely; a `null` judge reason is stored as an empty string rather than `'None'` (#27).
- Deterministic judges strip Unicode punctuation (curly quotes, ellipses, Unicode dashes) as well as ASCII punctuation, so `“Paris”…` matches `Paris` (#23).
- JSONL files with a UTF-8 BOM now load, and files that are not valid UTF-8 fail with an error naming the file instead of a raw codec message (#25).
- Protect token F1 scoring with regression tests for repeated tokens, empty normalized answers, and pass thresholds before score rounding.

- Preserve Unicode line separators inside JSONL strings so valid datasets and verdicts can be read back without spurious JSON errors.

- Project skeleton: `Case` and `Verdict` models, CI, packaging.
- Deterministic judges: exact match, contains-all, regex, token F1 (#1).
- LLM-as-judge with strict JSON verdicts and OpenAI-compatible client (#2).
- Judge reliability: accuracy, Cohen's kappa and confusion matrix against human labels (#7).
- `evalforge run` and `evalforge agreement` CLI commands over JSONL datasets (#9).
- `evalforge agreement --json` and `--min-kappa` to script and gate on judge reliability (#11).
- `evalforge run --judge llm` with `--rubric`, `--model`, `--base-url` and `--api-key-env` runs the LLM judge from the command line (#13).
- `OpenAICompatibleClient` raises a clear error on non-JSON bodies, error objects, empty `choices` and null content, so `evalforge run --judge llm` exits 1 with a message instead of a traceback (#15).
- `OpenAICompatibleClient` retries 429/5xx responses and transport errors with exponential backoff (`max_retries`, `backoff`, numeric `Retry-After` honoured), so a transient provider failure no longer aborts a whole `run` (#17).
- `evalforge run --min-pass-rate` exits 1 when the share of passed cases is below the threshold, after the verdicts are written, so a judge can gate CI (#19).
