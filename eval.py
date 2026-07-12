#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit("[ERROR] missing dependency: pyyaml") from exc


def fail(msg: str) -> None:
    raise SystemExit(f"[ERROR] {msg}")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        fail(f"bad config: {config_path}")
    return cfg


def load_prompts(path: Path) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sample_id = str(obj.get("id", "")).strip()
            if not sample_id:
                raise ValueError(f"missing id at line {line_no}")
            if sample_id in items:
                raise ValueError(f"duplicate id {sample_id!r} at line {line_no}")
            items[sample_id] = obj
    return items


def experiment_dirs(output_root: Path) -> list[Path]:
    if not output_root.exists():
        fail(f"output root not found: {output_root}")
    return sorted(
        p for p in output_root.iterdir()
        if p.is_dir() and (p / "params.csv").exists()
    )


def extract_python(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.I | re.S)
    return max(blocks, key=len).strip() if blocks else text.strip()


def run_tests(code: str, tests: list[str], timeout: float) -> tuple[bool, int, int, str]:
    passed = 0
    errors: list[str] = []
    for idx, test in enumerate(tests):
        program = code + "\n\n" + test + "\nprint('__PASS__')\n"
        with tempfile.TemporaryDirectory(prefix="dgtest_code_") as td:
            script = Path(td) / "candidate.py"
            script.write_text(program, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, str(script)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                errors.append(f"test{idx + 1}: timeout>{timeout}s")
                continue

            if proc.returncode == 0 and "__PASS__" in proc.stdout:
                passed += 1
            else:
                message = (proc.stderr or proc.stdout).strip().replace("\n", " ")
                errors.append(f"test{idx + 1}: {message[-300:]}")

    return passed == len(tests), passed, len(tests), " | ".join(errors)


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def content_words(text: str) -> list[str]:
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was",
        "were", "be", "been", "being", "that", "this", "it", "for", "on",
        "with", "as", "by", "from", "at", "which", "can", "may", "then",
        "than", "not", "into", "after", "before", "if", "while", "only",
        "more", "each", "every", "another", "same",
    }
    return [w for w in words(text) if len(w) > 3 and w not in stop]


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = set(content_words(a)), set(content_words(b))
    return len(sa & sb) / max(1, len(sa | sb))


def copied_ngram_ratio(source: str, output: str, n: int = 8) -> float:
    sw, ow = words(source), words(output)
    source_ngrams = {tuple(sw[i:i + n]) for i in range(max(0, len(sw) - n + 1))}
    output_ngrams = [tuple(ow[i:i + n]) for i in range(max(0, len(ow) - n + 1))]
    if not output_ngrams:
        return 0.0
    return sum(gram in source_ngrams for gram in output_ngrams) / len(output_ngrams)


def score_rewrite(text: str, meta: dict[str, Any]) -> dict[str, Any]:
    spec = meta.get("score", {})
    ps = paragraphs(text)
    target_n = int(spec.get("paragraphs", 3))
    lo = int(spec.get("min_words_per_paragraph", 170))
    hi = int(spec.get("max_words_per_paragraph", 220))
    counts = [len(words(p)) for p in ps]
    paragraph_ok = int(len(ps) == target_n)
    length_ok_count = sum(lo <= count <= hi for count in counts)

    terms = [str(x).lower() for x in spec.get("required_terms", [])]
    lower = text.lower()
    term_hits = sum(term in lower for term in terms)
    bullets = int(bool(re.search(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text)))

    source = str(spec.get("source_text", ""))
    similarity = jaccard_similarity(source, text)
    copied = copied_ngram_ratio(source, text, 8)

    score = 0.0
    score += 2.0 * paragraph_ok
    score += 3.0 * (length_ok_count / max(1, target_n))
    score += 3.0 * (term_hits / max(1, len(terms)))
    score += 1.0 if not bullets else 0.0
    score += min(1.0, similarity / 0.45)

    if copied > 0.35:
        score -= min(2.0, (copied - 0.35) * 4.0)

    score = max(0.0, min(10.0, score))
    return {
        "score": round(score, 3),
        "passed": int(score >= 8.0 and paragraph_ok and length_ok_count == target_n),
        "paragraph_count": len(ps),
        "paragraph_words": json.dumps(counts),
        "required_term_hits": f"{term_hits}/{len(terms)}",
        "semantic_jaccard": round(similarity, 4),
        "copied_8gram_ratio": round(copied, 4),
        "exact_hits": "",
        "section_hits": "",
        "tests_passed": "",
        "error": "",
    }


def normalize_math(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("\\frac", "frac")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = normalized.replace("−", "-")
    return normalized


def fraction_present(normalized: str, value: str) -> bool:
    num, den = value.split("/")
    candidates = {value, f"frac{num}{den}", f"({num})/({den})"}
    return any(candidate.replace(" ", "") in normalized for candidate in candidates)


def score_math(text: str, meta: dict[str, Any]) -> dict[str, Any]:
    spec = meta.get("score", {})
    exact = spec.get("exact", {})
    normalized = normalize_math(text)
    exact_hits = {key: fraction_present(normalized, str(value)) for key, value in exact.items()}

    required = [str(x).lower() for x in spec.get("required_sections", [])]
    section_hits = {
        "lagrangian": ("lagrangian" in text.lower() or "l=" in normalized),
        "stationarity": (
            "stationarity" in text.lower()
            or "∂" in text
            or "partial" in text.lower()
        ),
        "constraint": ("2x-y+3z=7" in normalized),
        "4x4": (
            "4x4" in normalized
            or "4×4" in text
            or "\\begin{bmatrix}" in text
            or "\\begin{pmatrix}" in text
        ),
        "hessian": (
            "hessian" in text.lower()
            or "positive definite" in text.lower()
            or "reduced hessian" in text.lower()
        ),
    }

    exact_count = sum(exact_hits.values())
    section_count = sum(section_hits.get(key, False) for key in required)
    score = (
        7.0 * exact_count / max(1, len(exact))
        + 3.0 * section_count / max(1, len(required))
    )
    passed = int(
        exact_count == len(exact)
        and section_count >= max(4, len(required) - 1)
    )

    return {
        "score": round(score, 3),
        "passed": passed,
        "paragraph_count": "",
        "paragraph_words": "",
        "required_term_hits": "",
        "semantic_jaccard": "",
        "copied_8gram_ratio": "",
        "exact_hits": json.dumps(exact_hits, ensure_ascii=False),
        "section_hits": json.dumps(section_hits, ensure_ascii=False),
        "tests_passed": "",
        "error": "",
    }


def score_code(text: str, meta: dict[str, Any], timeout: float) -> dict[str, Any]:
    tests = [str(x) for x in meta.get("tests", [])]
    code = extract_python(text)
    ok, passed_count, total_count, error = run_tests(code, tests, timeout)
    score = 10.0 * passed_count / max(1, total_count)

    return {
        "score": round(score, 3),
        "passed": int(ok),
        "paragraph_count": "",
        "paragraph_words": "",
        "required_term_hits": "",
        "semantic_jaccard": "",
        "copied_8gram_ratio": "",
        "exact_hits": "",
        "section_hits": "",
        "tests_passed": f"{passed_count}/{total_count}",
        "error": error,
    }


def score_one(text: str, meta: dict[str, Any], timeout: float) -> dict[str, Any]:
    sample_type = str(meta.get("type", "")).lower()
    if sample_type == "rewrite":
        return score_rewrite(text, meta)
    if sample_type == "math":
        return score_math(text, meta)
    if sample_type == "code":
        return score_code(text, meta, timeout)
    return {
        "score": 0.0,
        "passed": 0,
        "error": f"unsupported type: {sample_type}",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_summaries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_alpha: dict[str, list[dict[str, Any]]] = {}
    by_alpha_type: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for row in rows:
        experiment = str(row["experiment"])
        sample_type = str(row["type"])
        by_alpha.setdefault(experiment, []).append(row)
        by_alpha_type.setdefault((experiment, sample_type), []).append(row)

    summary_by_alpha: list[dict[str, Any]] = []
    for experiment, group in sorted(by_alpha.items()):
        scores = [float(row.get("score", 0.0)) for row in group]
        passed = [int(row.get("passed", 0)) for row in group]
        summary_by_alpha.append({
            "experiment": experiment,
            "mean_score": round(mean(scores), 4),
            "pass_rate": round(mean([float(x) for x in passed]), 4),
            "passed": sum(passed),
            "total": len(group),
        })

    summary_by_type: list[dict[str, Any]] = []
    for (experiment, sample_type), group in sorted(by_alpha_type.items()):
        scores = [float(row.get("score", 0.0)) for row in group]
        passed = [int(row.get("passed", 0)) for row in group]
        summary_by_type.append({
            "experiment": experiment,
            "type": sample_type,
            "mean_score": round(mean(scores), 4),
            "pass_rate": round(mean([float(x) for x in passed]), 4),
            "passed": sum(passed),
            "total": len(group),
        })

    summary_by_sample = [
        {
            "experiment": row["experiment"],
            "sample_id": row["sample_id"],
            "type": row["type"],
            "score": row.get("score", 0.0),
            "passed": row.get("passed", 0),
        }
        for row in rows
    ]
    return summary_by_alpha, summary_by_type, summary_by_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Score typed DGtest prompts")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}

    prompts_path = Path(args.prompts or paths.get("sample_file", "samples/prompts.jsonl"))
    output_root = Path(args.output_root or paths.get("output_root", "outputs"))
    eval_dir = Path(args.eval_dir or paths.get("eval_dir", "eval"))
    scores_path = Path(args.out) if args.out else eval_dir / "scores.csv"

    prompts = load_prompts(prompts_path)
    rows: list[dict[str, Any]] = []

    for exp_dir in experiment_dirs(output_root):
        for sample_id, meta in prompts.items():
            sample_type = str(meta.get("type", "")).lower()
            if sample_type not in {"rewrite", "math", "code"}:
                continue

            output_file = exp_dir / f"{sample_id}_final_output.txt"
            base = {
                "experiment": exp_dir.name,
                "sample_id": sample_id,
                "type": sample_type,
            }

            if not output_file.exists():
                rows.append({
                    **base,
                    "score": 0.0,
                    "passed": 0,
                    "error": "missing output file",
                })
                continue

            text = output_file.read_text(encoding="utf-8", errors="replace")
            rows.append({**base, **score_one(text, meta, args.timeout)})

    write_csv(scores_path, rows)

    summary_by_alpha, summary_by_type, summary_by_sample = build_summaries(rows)
    write_csv(eval_dir / "summary_by_alpha.csv", summary_by_alpha)
    write_csv(eval_dir / "summary_by_type.csv", summary_by_type)
    write_csv(eval_dir / "summary_by_sample.csv", summary_by_sample)

    for row in summary_by_alpha:
        print(
            f"{row['experiment']}: mean_score={float(row['mean_score']):.3f}, "
            f"passed={row['passed']}/{row['total']}"
        )

    print(f"[DONE] scores: {scores_path}")
    print(f"[DONE] summaries: {eval_dir}")


if __name__ == "__main__":
    main()
