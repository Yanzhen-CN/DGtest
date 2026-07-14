from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from eval_benchmarks import build_code_program, load_benchmark


@dataclass(frozen=True)
class ScoreOutcome:
    passed: bool
    detail: str
    protocol: str
    metric: str
    extracted_answer: str = ""
    gold_answer: str = ""
    scorer_version: str = ""


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


GSM8K_FINAL_LINE = re.compile(r"(?im)^\s*Final answer:\s*([^\r\n]+?)\s*$")
GSM8K_NUMBER = re.compile(r"^-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")


def normalize_gsm8k_number(value: str) -> str:
    value = value.strip().strip("*` ").replace("$", "")
    if value.endswith("."):
        value = value[:-1].rstrip()
    if not GSM8K_NUMBER.fullmatch(value):
        return ""
    return value.replace(",", "")


def score_gsm8k(row: dict[str, Any]) -> ScoreOutcome:
    prediction = str(row.get("generated_text", ""))
    gold = str(row.get("gold_answer", ""))
    matches = GSM8K_FINAL_LINE.findall(prediction)
    extracted = normalize_gsm8k_number(matches[-1]) if matches else ""
    normalized_gold = normalize_gsm8k_number(gold)
    if not normalized_gold:
        raise RuntimeError(f"invalid GSM8K gold for {row.get('sample_id')}: {gold!r}")

    passed = bool(extracted) and extracted == normalized_gold
    detail = "passed" if passed else (
        "missing_or_invalid_final_answer" if not extracted else "wrong_answer"
    )
    return ScoreOutcome(
        passed=passed,
        detail=detail,
        protocol="gsm8k_zero_shot_final_exact_v1",
        metric="exact_match",
        extracted_answer=extracted,
        gold_answer=normalized_gold,
        scorer_version="builtin-v1",
    )


def score_math500(row: dict[str, Any], precision: int) -> ScoreOutcome:
    prediction = str(row.get("generated_text", ""))
    gold = str(row.get("gold_answer", ""))
    try:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    except ImportError as exc:
        raise SystemExit(
            "[ERROR] MATH-500 requires Math-Verify; run: "
            "pip install 'math-verify[antlr4_13_2]'"
        ) from exc

    # Match the public Math-Verify extraction protocol explicitly. Dataset
    # golds are clean LaTeX. Predictions prefer boxed LaTeX and then allow the
    # same LaTeX/plain-expression fallbacks used by the public evaluator.
    gold_config = (LatexExtractionConfig(),)
    prediction_config = (
        LatexExtractionConfig(boxed_match_priority=0),
        ExprExtractionConfig(),
    )
    parse_kwargs = {"parsing_timeout": None} if os.name == "nt" else {}
    verify_kwargs: dict[str, Any] = {"float_rounding": precision, "strict": True}
    if os.name == "nt":
        # Math-Verify's timeout helper can fail under Microsoft Store Python
        # with WinError 6. Scoring remains sequential; only its helper timeout
        # is disabled on Windows.
        verify_kwargs["timeout_seconds"] = None

    try:
        parsed_gold = parse(
            f"${gold}$", extraction_config=gold_config, **parse_kwargs
        )
        parsed_prediction = parse(
            prediction, extraction_config=prediction_config, **parse_kwargs
        )
        if not parsed_gold:
            raise RuntimeError(
                f"Math-Verify could not parse gold for {row.get('sample_id')}: {gold!r}"
            )
        passed = bool(parsed_prediction) and bool(
            verify(parsed_gold, parsed_prediction, **verify_kwargs)
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Math-Verify failed for {row.get('sample_id')}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        extracted = " | ".join(str(value) for value in parsed_prediction)
    except Exception:
        extracted = "<unprintable>"
    return ScoreOutcome(
        passed=passed,
        detail="passed" if passed else (
            "wrong_answer" if parsed_prediction else "answer_extraction_failed"
        ),
        protocol="math500_math_verify_v1",
        metric="exact_match",
        extracted_answer=extracted,
        gold_answer=gold,
        scorer_version=importlib.metadata.version("math-verify"),
    )


def score_code(row: dict[str, Any], timeout: float) -> ScoreOutcome:
    benchmark = str(row["benchmark"])
    program = build_code_program(benchmark, row)
    protocol = f"{benchmark}_official_tests_chat_v1"
    version = f"python-{sys.version_info.major}.{sys.version_info.minor}"

    with tempfile.TemporaryDirectory(prefix="dg-eval-") as temp_dir:
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", program],
                cwd=temp_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ScoreOutcome(
                False, "timeout", protocol, "pass@1", scorer_version=version
            )

    if result.returncode == 0:
        passed, detail = True, "passed"
    else:
        lines = (result.stderr or result.stdout).strip().splitlines()
        passed = False
        detail = lines[-1][:300] if lines else f"exit_{result.returncode}"
    return ScoreOutcome(
        passed, detail, protocol, "pass@1", scorer_version=version
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score saved DiffusionGemma benchmark outputs")
    parser.add_argument("--eval-config", default="eval_config.yaml")
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="execute untrusted generated Python for HumanEval/MBPP scoring",
    )
    args = parser.parse_args()
    cfg = load_yaml(args.eval_config)
    output_root = Path(cfg.get("output_root", "eval/outputs"))
    scores_root = Path(cfg.get("scores_root", "eval/scores"))
    scoring_cfg = cfg.get("scoring", {})
    timeout = float(scoring_cfg.get("code_timeout_seconds", 10))
    math_precision = int(scoring_cfg.get("math_precision", 6))
    if bool(cfg.get("disable_xet_for_benchmarks", True)):
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    detail_rows: list[dict[str, Any]] = []
    mbpp_metadata: dict[str, dict[str, Any]] | None = None
    for path in sorted(output_root.glob("*/*/predictions.jsonl")):
        for row in read_jsonl(path):
            benchmark = str(row["benchmark"])
            if benchmark == "mbpp" and "test_imports" not in row:
                # Outputs produced before test_imports was persisted remain
                # scoreable without another GPU run: hydrate only the missing
                # official execution metadata from the small public dataset.
                if mbpp_metadata is None:
                    mbpp_metadata = {
                        str(item["sample_id"]): item for item in load_benchmark("mbpp")
                    }
                official = mbpp_metadata.get(str(row["sample_id"]))
                if official is None:
                    raise RuntimeError(f"unknown MBPP sample_id: {row['sample_id']}")
                row = {**official, **row}
            if benchmark in {"humaneval", "mbpp"}:
                if args.allow_code_execution:
                    outcome = score_code(row, timeout)
                    scored = True
                else:
                    outcome = ScoreOutcome(
                        False,
                        "code_execution_disabled",
                        f"{benchmark}_official_tests_chat_v1",
                        "pass@1",
                    )
                    scored = False
            elif benchmark == "gsm8k":
                outcome = score_gsm8k(row)
                scored = True
            elif benchmark == "math500":
                outcome = score_math500(row, math_precision)
                scored = True
            else:
                raise RuntimeError(f"unsupported benchmark in predictions: {benchmark}")

            detail_rows.append({
                "benchmark": benchmark,
                "experiment": row["experiment"],
                "sample_id": row["sample_id"],
                "scored": int(scored),
                "passed": int(outcome.passed) if scored else "",
                "score": float(outcome.passed) if scored else "",
                "metric": outcome.metric,
                "protocol": outcome.protocol,
                "scorer_version": outcome.scorer_version,
                "extracted_answer": outcome.extracted_answer,
                "gold_answer": outcome.gold_answer,
                "detail": outcome.detail,
                "latency_sec": row.get("latency_sec", ""),
            })

    if not detail_rows:
        raise SystemExit(f"[ERROR] no predictions found under {output_root}")
    fields = [
        "benchmark", "experiment", "sample_id", "scored", "passed", "score",
        "metric", "protocol", "scorer_version", "extracted_answer", "gold_answer",
        "detail", "latency_sec",
    ]
    write_csv(scores_root / "scores.csv", detail_rows, fields)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        if row["scored"]:
            grouped[(str(row["experiment"]), str(row["benchmark"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (experiment, benchmark), rows in sorted(grouped.items()):
        protocols = {str(row["protocol"]) for row in rows}
        metrics = {str(row["metric"]) for row in rows}
        versions = {str(row["scorer_version"]) for row in rows}
        if len(protocols) != 1 or len(metrics) != 1 or len(versions) != 1:
            raise RuntimeError(f"mixed scoring protocols in {experiment}/{benchmark}")
        summary.append({
            "experiment": experiment,
            "benchmark": benchmark,
            "metric": next(iter(metrics)),
            "protocol": next(iter(protocols)),
            "scorer_version": next(iter(versions)),
            "samples": len(rows),
            "passed": sum(int(row["passed"]) for row in rows),
            "accuracy": sum(float(row["score"]) for row in rows) / len(rows),
            "mean_latency_sec": sum(float(row["latency_sec"]) for row in rows) / len(rows),
        })
    write_csv(
        scores_root / "summary.csv",
        summary,
        [
            "experiment", "benchmark", "metric", "protocol", "scorer_version",
            "samples", "passed", "accuracy", "mean_latency_sec",
        ],
    )
    print(f"[DONE] detailed scores: {scores_root / 'scores.csv'}")
    print(f"[DONE] summary: {scores_root / 'summary.csv'}")
    if not args.allow_code_execution:
        print("[NOTE] code benchmarks were not scored; rerun with --allow-code-execution in a sandbox")


if __name__ == "__main__":
    main()
