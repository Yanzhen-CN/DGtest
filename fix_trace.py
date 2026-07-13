#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return default


def parse_list(value: Any) -> list[Any]:
    text = str(value or "").strip()
    if not text:
        return []
    obj = json.loads(text)
    if not isinstance(obj, list):
        raise ValueError("expected JSON list")
    return obj


def trace_key(row: dict[str, str]) -> tuple[str, int]:
    return str(row.get("sample_id", "")), parse_int(row.get("trace_index"), 0)


def normalize_canvas_indices(rows: list[dict[str, str]]) -> None:
    previous_sample: str | None = None
    previous_step: int | None = None
    canvas_index = 0

    for row in sorted(rows, key=trace_key):
        sample_id = str(row.get("sample_id", ""))
        step = parse_int(row.get("generation_step"), 0)

        if sample_id != previous_sample:
            previous_sample = sample_id
            previous_step = None
            canvas_index = 0
        elif previous_step is not None and step > previous_step:
            canvas_index += 1

        row["canvas_index"] = str(canvas_index)
        previous_step = step


def repair_rows(
    rows: list[dict[str, str]],
    original_fields: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    normalize_canvas_indices(rows)

    fields = list(original_fields)
    required_new = [
        "changed_count",
        "changed_positions",
        "update_ranks",
        "output_changed_count",
        "output_changed_positions",
    ]
    for field in required_new:
        if field not in fields:
            # Place event fields near accepted positions when possible.
            fields.append(field)

    acceptance_counts: dict[tuple[str, int, int], int] = {}

    for row in sorted(rows, key=trace_key):
        sample_id = str(row.get("sample_id", ""))
        canvas_index = parse_int(row.get("canvas_index"), 0)

        accepted_positions = [
            int(x) for x in parse_list(row.get("accepted_positions"))
        ]

        update_ranks: list[int] = []
        for position in accepted_positions:
            key = (sample_id, canvas_index, position)
            rank = acceptance_counts.get(key, 0) + 1
            acceptance_counts[key] = rank
            update_ranks.append(rank)

        # Project semantics: accepted events only.
        row["accepted_count"] = str(len(accepted_positions))
        row["changed_count"] = str(len(accepted_positions))
        row["changed_positions"] = json.dumps(
            accepted_positions, ensure_ascii=False
        )
        row["update_ranks"] = json.dumps(update_ranks, ensure_ascii=False)

        # Diagnostic raw canvas differences. Never used as commit/revision.
        input_ids = [int(x) for x in parse_list(row.get("input_canvas_token_ids"))]
        output_ids = [int(x) for x in parse_list(row.get("output_canvas_token_ids"))]
        output_changed = [
            index
            for index in range(min(len(input_ids), len(output_ids)))
            if input_ids[index] != output_ids[index]
        ]
        row["output_changed_count"] = str(len(output_changed))
        row["output_changed_positions"] = json.dumps(
            output_changed, ensure_ascii=False
        )

    return rows, fields


def process_trace(path: Path, remove_clean: bool) -> None:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    required = {
        "sample_id",
        "trace_index",
        "generation_step",
        "accepted_positions",
        "input_canvas_token_ids",
        "output_canvas_token_ids",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(missing)}")

    repaired, out_fields = repair_rows(rows, fields)

    backup = path.with_suffix(path.suffix + ".before_accept_fix")
    if not backup.exists():
        shutil.copy2(path, backup)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(repaired)

    clean_path = path.parent / "clean_trace.csv"
    if remove_clean and clean_path.exists():
        clean_backup = clean_path.with_suffix(
            clean_path.suffix + ".removed_backup"
        )
        if not clean_backup.exists():
            shutil.copy2(clean_path, clean_backup)
        clean_path.unlink()

    repeated = sum(
        1
        for row in repaired
        for rank in parse_list(row.get("update_ranks"))
        if int(rank) > 1
    )
    print(
        f"[OK] {path}: rows={len(repaired)}, "
        f"accepted/revision events={sum(parse_int(r.get('accepted_count')) for r in repaired)}, "
        f"repeat events={repeated}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Repair existing DG trace.csv files so commit/revision means "
            "accepted_positions only; re-noise remains diagnostic only."
        )
    )
    parser.add_argument("--root", default="outputs")
    parser.add_argument(
        "--remove-clean",
        action="store_true",
        help="remove stale clean_trace.csv after writing a backup",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"[ERROR] root not found: {root}")

    trace_paths = sorted(root.rglob("trace.csv"))
    if not trace_paths and (root / "trace.csv").exists():
        trace_paths = [root / "trace.csv"]
    if not trace_paths:
        raise SystemExit(f"[ERROR] no trace.csv found under: {root}")

    for path in trace_paths:
        process_trace(path, remove_clean=args.remove_clean)

    print(
        "[DONE] accepted_positions = commit/revision; "
        "output_changed_positions = accepted + re-noise diagnostic"
    )


if __name__ == "__main__":
    main()
