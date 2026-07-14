from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from eval_benchmarks import build_code_program, extract_numeric_answer


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_number(value: str) -> str:
    value = value.strip().replace(",", "").replace("$", "")
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:.12g}"
    except ValueError:
        return value.lower().replace(" ", "")


def score_math(row: dict[str, Any]) -> tuple[bool, str]:
    prediction = str(row.get("generated_text", ""))
    gold = str(row.get("gold_answer", ""))
    try:
        from math_verify import parse, verify

        parsed_gold = parse(gold)
        parsed_prediction = parse(prediction)
        return bool(verify(parsed_gold, parsed_prediction)), "math_verify"
    except Exception:
        predicted = extract_numeric_answer(prediction)
        return normalize_number(predicted) == normalize_number(gold), "numeric_fallback"


def score_code(row: dict[str, Any], timeout: float) -> tuple[bool, str]:
    program = build_code_program(str(row["benchmark"]), row)
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
            return False, "timeout"
    if result.returncode == 0:
        return True, "passed"
    detail = (result.stderr or result.stdout).strip().splitlines()
    return False, (detail[-1][:300] if detail else f"exit_{result.returncode}")


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
    timeout = float(cfg.get("scoring", {}).get("code_timeout_seconds", 10))

    detail_rows: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("*/*/predictions.jsonl")):
        for row in read_jsonl(path):
            benchmark = str(row["benchmark"])
            if benchmark in {"humaneval", "mbpp"}:
                if not args.allow_code_execution:
                    passed, detail, scored = False, "code_execution_disabled", False
                else:
                    passed, detail = score_code(row, timeout)
                    scored = True
            else:
                passed, detail = score_math(row)
                scored = True
            detail_rows.append({
                "benchmark": benchmark,
                "experiment": row["experiment"],
                "sample_id": row["sample_id"],
                "scored": int(scored),
                "passed": int(passed) if scored else "",
                "score": float(passed) if scored else "",
                "detail": detail,
                "latency_sec": row.get("latency_sec", ""),
            })

    if not detail_rows:
        raise SystemExit(f"[ERROR] no predictions found under {output_root}")
    fields = ["benchmark", "experiment", "sample_id", "scored", "passed", "score", "detail", "latency_sec"]
    write_csv(scores_root / "scores.csv", detail_rows, fields)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        if row["scored"]:
            grouped[(str(row["experiment"]), str(row["benchmark"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (experiment, benchmark), rows in sorted(grouped.items()):
        summary.append({
            "experiment": experiment,
            "benchmark": benchmark,
            "samples": len(rows),
            "passed": sum(int(row["passed"]) for row in rows),
            "accuracy": sum(float(row["score"]) for row in rows) / len(rows),
            "mean_latency_sec": sum(float(row["latency_sec"]) for row in rows) / len(rows),
        })
    write_csv(
        scores_root / "summary.csv",
        summary,
        ["experiment", "benchmark", "samples", "passed", "accuracy", "mean_latency_sec"],
    )
    print(f"[DONE] detailed scores: {scores_root / 'scores.csv'}")
    print(f"[DONE] summary: {scores_root / 'summary.csv'}")
    if not args.allow_code_execution:
        print("[NOTE] code benchmarks were not scored; rerun with --allow-code-execution in a sandbox")


if __name__ == "__main__":
    main()
