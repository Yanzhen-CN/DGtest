from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def alpha_dir_name(old_name: str, params_rows: list[dict[str, str]]) -> str:
    """Convert legacy names such as none / alpha_1 / alpha_0p5 to current names."""
    for row in params_rows:
        raw = str(row.get("self_conditioning_alpha", "")).strip()
        mode = str(row.get("mode", "")).strip().lower()

        if mode == "none" or raw == "":
            return "alphanone"

        try:
            value = float(raw)
            text = f"{value:g}".replace(".", "p").replace("-", "minus_")
            return f"alpha{text}"
        except ValueError:
            pass

    normalized = old_name.lower().strip()
    if normalized in {"none", "alphanone", "alpha_none"}:
        return "alphanone"

    normalized = normalized.removeprefix("alpha_").removeprefix("alpha")
    normalized = normalized.replace(".", "p")
    return f"alpha{normalized}"


def normalize_experiment_name(row: dict[str, Any], new_name: str) -> dict[str, Any]:
    result = dict(row)
    if "experiment_name" in result:
        result["experiment_name"] = new_name
    return result


def find_final_output(exp_dir: Path, sample_id: str) -> Path | None:
    candidates = [
        exp_dir / f"{sample_id}_final_output.txt",
        exp_dir / f"{sample_id}.txt",
        exp_dir / sample_id / "final_output.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def migrate(
    source_root: Path,
    destination_root: Path,
    canvas_length: int,
    steps: int,
    overwrite: bool,
) -> None:
    if not source_root.is_dir():
        raise SystemExit(f"[ERROR] source directory not found: {source_root}")

    budget_root = destination_root / f"len{canvas_length}" / f"step{steps}"
    if budget_root.exists() and overwrite:
        shutil.rmtree(budget_root)
    budget_root.mkdir(parents=True, exist_ok=True)

    legacy_experiment_dirs = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and (path / "params.csv").exists() and (path / "trace.csv").exists()
    )
    if not legacy_experiment_dirs:
        raise SystemExit(
            f"[ERROR] no legacy experiment directories with params.csv and trace.csv under {source_root}"
        )

    migrated_results: list[dict[str, Any]] = []
    source_results = source_root / "results.jsonl"
    if source_results.exists():
        with source_results.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    migrated_results.append(json.loads(line))

    total_samples = 0

    for exp_dir in legacy_experiment_dirs:
        params_path = exp_dir / "params.csv"
        trace_path = exp_dir / "trace.csv"

        params_rows = read_csv(params_path)
        trace_rows = read_csv(trace_path)

        if not params_rows:
            print(f"[SKIP] empty params.csv: {params_path}")
            continue

        new_experiment = alpha_dir_name(exp_dir.name, params_rows)
        sample_ids = sorted(
            {
                str(row.get("sample_id", "")).strip()
                for row in params_rows + trace_rows
                if str(row.get("sample_id", "")).strip()
            }
        )

        for sample_id in sample_ids:
            leaf = budget_root / new_experiment / sample_id
            if leaf.exists() and not overwrite:
                raise SystemExit(
                    f"[ERROR] destination already exists: {leaf}\n"
                    "Use --overwrite to replace the migrated step directory."
                )
            leaf.mkdir(parents=True, exist_ok=True)

            sample_params = [
                normalize_experiment_name(row, new_experiment)
                for row in params_rows
                if str(row.get("sample_id", "")).strip() == sample_id
            ]
            sample_trace = [
                normalize_experiment_name(row, new_experiment)
                for row in trace_rows
                if str(row.get("sample_id", "")).strip() == sample_id
            ]

            if sample_params:
                write_csv(leaf / "params.csv", sample_params, list(sample_params[0].keys()))
            else:
                print(f"[WARN] no params rows for {exp_dir.name}/{sample_id}")

            if sample_trace:
                write_csv(leaf / "trace.csv", sample_trace, list(sample_trace[0].keys()))
            else:
                print(f"[WARN] no trace rows for {exp_dir.name}/{sample_id}")

            final_source = find_final_output(exp_dir, sample_id)
            if final_source is not None:
                shutil.copy2(final_source, leaf / "final_output.txt")
            elif sample_params:
                generated = sample_params[0].get("generated_text", "")
                (leaf / "final_output.txt").write_text(generated, encoding="utf-8")
                print(f"[WARN] rebuilt final_output.txt from params.csv: {sample_id}")
            else:
                print(f"[WARN] final output missing: {exp_dir.name}/{sample_id}")

            print(
                f"[OK] {exp_dir.name}/{sample_id} -> "
                f"{leaf}  params={len(sample_params)} trace={len(sample_trace)}"
            )
            total_samples += 1

        # clean_trace.csv is deliberately not migrated: the current format no longer uses it.
        if (exp_dir / "clean_trace.csv").exists():
            print(f"[SKIP] obsolete clean trace: {exp_dir / 'clean_trace.csv'}")

    if migrated_results:
        experiment_map: dict[str, str] = {}
        for exp_dir in legacy_experiment_dirs:
            rows = read_csv(exp_dir / "params.csv")
            experiment_map[exp_dir.name] = alpha_dir_name(exp_dir.name, rows)

        result_path = budget_root / "results.jsonl"
        with result_path.open("w", encoding="utf-8") as file:
            for row in migrated_results:
                old_name = str(row.get("experiment_name", ""))
                if old_name in experiment_map:
                    row["experiment_name"] = experiment_map[old_name]
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[OK] aggregate results -> {result_path}")

    print(f"[DONE] migrated sample/experiment leaves: {total_samples}")
    print(f"[DONE] destination: {budget_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate legacy outputs_256 layout into the current "
            "outputs/len256/step48/<alpha>/<sample>/ layout, "
            "splitting shared params.csv and trace.csv by sample_id."
        )
    )
    parser.add_argument("--src", default="outputs_256", help="legacy output root")
    parser.add_argument("--dst", default="outputs", help="new output root")
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    migrate(
        source_root=Path(args.src),
        destination_root=Path(args.dst),
        canvas_length=args.length,
        steps=args.steps,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
