#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"[ERROR] {label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch DGtest run.py for dual true/clean canvas traces")
    ap.add_argument("--run", default="run.py")
    args = ap.parse_args()

    path = Path(args.run).resolve()
    if not path.exists():
        raise SystemExit(f"[ERROR] missing: {path}")

    text = path.read_text(encoding="utf-8")
    marker = "DGTEST_DUAL_CANVAS_TRACE_V3"
    if marker in text:
        print(f"[OK] already patched: {path}")
        return

    backup = path.with_suffix(path.suffix + ".before_dual_trace")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[BACKUP] {backup}")

    text = replace_once(
        text,
        'PATCH_MARKER = "DGTEST_SELFCOND_ALPHA_SWEEP_PATCH"',
        f'PATCH_MARKER = "{marker}"',
        "PATCH_MARKER",
    )

    old_payload = '''                        "position_entropy": token_entropy.detach().cpu().tolist(),
                        "argmax_token_ids": new_argmax_canvas.detach().cpu().tolist(),
                        "accepted_canvas_token_ids": accepted_canvas.detach().cpu().tolist(),
'''
    new_payload = '''                        "position_entropy": token_entropy.detach().cpu().tolist(),
                        "input_canvas_token_ids": current_canvas.detach().cpu().tolist(),
                        "sampled_canvas_token_ids": denoiser_canvas.detach().cpu().tolist(),
                        "accepted_canvas_token_ids": accepted_canvas.detach().cpu().tolist(),
                        "output_canvas_token_ids": new_current_canvas.detach().cpu().tolist(),
                        "argmax_token_ids": new_argmax_canvas.detach().cpu().tolist(),
'''
    text = replace_once(text, old_payload, new_payload, "sampler trace payload")

    old_extract = '''        positions = (item.get("accepted_positions") or [[]])[0]
        accepted_canvas = (item.get("accepted_canvas_token_ids") or [[]])[0]
        argmax_canvas = (item.get("argmax_token_ids") or [[]])[0]
        entropies = (item.get("position_entropy") or [[]])[0]
'''
    new_extract = '''        positions = (item.get("accepted_positions") or [[]])[0]
        input_canvas = (item.get("input_canvas_token_ids") or [[]])[0]
        sampled_canvas = (item.get("sampled_canvas_token_ids") or [[]])[0]
        accepted_canvas = (item.get("accepted_canvas_token_ids") or [[]])[0]
        output_canvas = (item.get("output_canvas_token_ids") or [[]])[0]
        argmax_canvas = (item.get("argmax_token_ids") or [[]])[0]
        entropies = (item.get("position_entropy") or [[]])[0]
'''
    text = replace_once(text, old_extract, new_extract, "trace extraction")

    old_changed = '''        changed_entropies = [
            entropies[p] if 0 <= p < len(entropies) else None for p in positions
        ]
        rows.append({
'''
    new_changed = '''        changed_entropies = [
            entropies[p] if 0 <= p < len(entropies) else None for p in positions
        ]

        def decode_tokens(token_ids):
            return [decode_piece(processor, int(token_id)) for token_id in token_ids]

        input_tokens = decode_tokens(input_canvas)
        sampled_tokens = decode_tokens(sampled_canvas)
        accepted_canvas_tokens = decode_tokens(accepted_canvas)
        output_tokens = decode_tokens(output_canvas)
        argmax_tokens = decode_tokens(argmax_canvas)

        rows.append({
'''
    text = replace_once(text, old_changed, new_changed, "decode full canvases")

    old_row = '''            "accepted_tokens": json.dumps(changed_tokens, ensure_ascii=False),
            "accepted_canvas_token_ids": json.dumps(accepted_canvas, ensure_ascii=False),
            "argmax_token_ids": json.dumps(argmax_canvas, ensure_ascii=False),
            "position_entropy": json.dumps(entropies, ensure_ascii=False),
'''
    new_row = '''            "accepted_tokens": json.dumps(changed_tokens, ensure_ascii=False),
            "input_canvas_token_ids": json.dumps(input_canvas, ensure_ascii=False),
            "input_canvas_tokens": json.dumps(input_tokens, ensure_ascii=False),
            "sampled_canvas_token_ids": json.dumps(sampled_canvas, ensure_ascii=False),
            "sampled_canvas_tokens": json.dumps(sampled_tokens, ensure_ascii=False),
            "accepted_canvas_token_ids": json.dumps(accepted_canvas, ensure_ascii=False),
            "accepted_canvas_tokens": json.dumps(accepted_canvas_tokens, ensure_ascii=False),
            "output_canvas_token_ids": json.dumps(output_canvas, ensure_ascii=False),
            "output_canvas_tokens": json.dumps(output_tokens, ensure_ascii=False),
            "argmax_token_ids": json.dumps(argmax_canvas, ensure_ascii=False),
            "argmax_tokens": json.dumps(argmax_tokens, ensure_ascii=False),
            "position_entropy": json.dumps(entropies, ensure_ascii=False),
'''
    text = replace_once(text, old_row, new_row, "trace row fields")

    anchor = '''    return rows


def clear_experiment_outputs'''
    clean_fn = '''    return rows


def clean_trace_csv_rows(
    true_rows: list[dict[str, Any]],
    sample_id: str,
    experiment_name: str,
    alpha: float | None,
) -> list[dict[str, Any]]:
    if not true_rows:
        return []

    def parse_list(value: Any) -> list[Any]:
        if value is None or str(value).strip() == "":
            return []
        obj = json.loads(str(value))
        return list(obj) if isinstance(obj, list) else []

    first_ids = parse_list(true_rows[0].get("input_canvas_token_ids"))
    canvas_len = len(first_ids) or 256
    rows: list[dict[str, Any]] = [{
        "sample_id": sample_id,
        "experiment_name": experiment_name,
        "self_conditioning_alpha": alpha,
        "generation_step": -1,
        "phase": "initial_all_mask",
        "accepted_count": 0,
        "accepted_positions": "[]",
        "newly_committed_positions": "[]",
        "unaccepted_positions": "[]",
        "clean_canvas_tokens": json.dumps(["□"] * canvas_len, ensure_ascii=False),
        "clean_canvas_text": "□" * canvas_len,
        "mean_entropy": None,
    }]

    previous: set[int] = set()
    for true_row in true_rows:
        positions = [int(x) for x in parse_list(true_row.get("accepted_positions"))]
        accepted_tokens = [str(x) for x in parse_list(true_row.get("accepted_tokens"))]
        visible = ["□"] * canvas_len
        for pos, token in zip(positions, accepted_tokens):
            if 0 <= pos < canvas_len:
                visible[pos] = token

        current = {p for p in positions if 0 <= p < canvas_len}
        rows.append({
            "sample_id": sample_id,
            "experiment_name": experiment_name,
            "self_conditioning_alpha": alpha,
            "generation_step": true_row.get("generation_step"),
            "phase": "accepted_state",
            "accepted_count": true_row.get("accepted_count"),
            "accepted_positions": json.dumps(sorted(current), ensure_ascii=False),
            "newly_committed_positions": json.dumps(sorted(current - previous), ensure_ascii=False),
            "unaccepted_positions": json.dumps(sorted(previous - current), ensure_ascii=False),
            "clean_canvas_tokens": json.dumps(visible, ensure_ascii=False),
            "clean_canvas_text": "".join(visible),
            "mean_entropy": true_row.get("mean_entropy"),
        })
        previous = current

    return rows


def clear_experiment_outputs'''
    text = replace_once(text, anchor, clean_fn, "insert clean trace builder")

    text = text.replace(
        'for filename in ("params.csv", "trace.csv"):',
        'for filename in ("params.csv", "trace.csv", "clean_trace.csv"):',
        1,
    )

    old_fields = '''        "accepted_tokens",
        "accepted_canvas_token_ids",
        "argmax_token_ids",
        "position_entropy",
'''
    new_fields = '''        "accepted_tokens",
        "input_canvas_token_ids",
        "input_canvas_tokens",
        "sampled_canvas_token_ids",
        "sampled_canvas_tokens",
        "accepted_canvas_token_ids",
        "accepted_canvas_tokens",
        "output_canvas_token_ids",
        "output_canvas_tokens",
        "argmax_token_ids",
        "argmax_tokens",
        "position_entropy",
'''
    text = replace_once(text, old_fields, new_fields, "trace field list")

    dict_anchor = '''    per_experiment_trace: dict[str, list[dict[str, Any]]] = {
        exp["name"]: [] for exp in experiments
    }
'''
    dict_new = dict_anchor + '''    per_experiment_clean_trace: dict[str, list[dict[str, Any]]] = {
        exp["name"]: [] for exp in experiments
    }
'''
    text = replace_once(text, dict_anchor, dict_new, "clean trace accumulator")

    call_anchor = '''                per_experiment_trace[exp["name"]].extend(
                    trace_csv_rows(
                        model._dg_trace,
                        processor=processor,
                        sample_id=sample["id"],
                        experiment_name=exp["name"],
                        alpha=alpha,
                        latency_sec=latency,
                        generated_text=generated_text,
                    )
                )
'''
    call_new = '''                true_rows = trace_csv_rows(
                    model._dg_trace,
                    processor=processor,
                    sample_id=sample["id"],
                    experiment_name=exp["name"],
                    alpha=alpha,
                    latency_sec=latency,
                    generated_text=generated_text,
                )
                per_experiment_trace[exp["name"]].extend(true_rows)
                per_experiment_clean_trace[exp["name"]].extend(
                    clean_trace_csv_rows(
                        true_rows,
                        sample_id=sample["id"],
                        experiment_name=exp["name"],
                        alpha=alpha,
                    )
                )
'''
    text = replace_once(text, call_anchor, call_new, "collect dual traces")

    write_anchor = '''        write_csv(
            exp_dir / "trace.csv",
            trace_fields,
            per_experiment_trace[exp["name"]],
        )
'''
    write_new = '''        write_csv(
            exp_dir / "trace.csv",
            trace_fields,
            per_experiment_trace[exp["name"]],
        )
        clean_trace_fields = [
            "sample_id",
            "experiment_name",
            "self_conditioning_alpha",
            "generation_step",
            "phase",
            "accepted_count",
            "accepted_positions",
            "newly_committed_positions",
            "unaccepted_positions",
            "clean_canvas_tokens",
            "clean_canvas_text",
            "mean_entropy",
        ]
        write_csv(
            exp_dir / "clean_trace.csv",
            clean_trace_fields,
            per_experiment_clean_trace[exp["name"]],
        )
'''
    text = replace_once(text, write_anchor, write_new, "write dual traces")

    text = text.replace(
        'print(f"[DONE] per-alpha CSVs: {output_root}/alpha_*/params.csv and trace.csv")',
        'print(f"[DONE] per-alpha CSVs: {output_root}/alpha_*/params.csv, trace.csv and clean_trace.csv")',
        1,
    )

    path.write_text(text, encoding="utf-8")
    print(f"[OK] patched: {path}")
    print("python run.py --config config.yaml --restore-patch")
    print("python run.py --config config.yaml --patch-only")


if __name__ == "__main__":
    main()
