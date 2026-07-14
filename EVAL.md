# Benchmark evaluation protocol

The benchmark pipeline deliberately separates GPU generation from local scoring:

```text
run_eval.py -> eval/outputs/*/*/predictions.jsonl
score_eval.py -> eval/scores/{scores,summary}.csv
visual_eval.py -> eval/figures/*.png
```

## Scoring protocols

| Benchmark | Primary metric | Protocol |
|---|---|---|
| HumanEval | pass@1 | Execute the official `test` and `check(entry_point)` for one saved completion per task. |
| MBPP Sanitized | pass@1 | Execute the sanitized `test_imports` and `test_list`; challenge tests are not mixed into the public score. |
| GSM8K | exact match | Accept only the numeric payload on the last `Final answer:` line, then remove `$` and thousands separators. Intermediate reasoning numbers are ignored. |
| MATH-500 | exact match | Math-Verify with LaTeX-only gold extraction, boxed-LaTeX priority for predictions, expression fallback, strict comparison, and six-digit float rounding. |

Every detailed score records `metric`, `protocol`, `scorer_version`,
`extracted_answer`, and `gold_answer`. A scoring dependency failure or an
unparseable MATH-500 gold stops the run; it never silently falls back to a
different metric.

The prompts in this project are controlled zero-shot chat prompts. Therefore,
the resulting scores are internally comparable between self-conditioning
settings, but should be labelled **custom zero-shot protocol** rather than an
exact reproduction of leaderboards that use different few-shot prompts or
generation settings.

## Commands

Generate outputs on the GPU machine:

```bash
python run_eval.py
```

Score saved outputs locally. Generated benchmark code is untrusted, so use an
isolated machine or sandbox before enabling code execution:

```bash
python score_eval.py --allow-code-execution
```

Existing MBPP output files created before `test_imports` was persisted are
hydrated from the official dataset during scoring; no GPU regeneration is
required.

Create figures from the strict summary:

```bash
python visual_eval.py
```

For a publication-quality result, run the complete test splits. Results from
`run_eval.py --limit 50` must be reported as a 50-sample pilot.
