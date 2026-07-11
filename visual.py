from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
from transformers import AutoProcessor

try:
    import yaml
except ImportError as exc:
    raise SystemExit("[ERROR] missing dependency: pyyaml. Install with: pip install pyyaml") from exc


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first_scalar(v):
    if isinstance(v, list):
        return v[0] if v else None
    return v


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def exp_sort_key(row: dict[str, Any]):
    exp_id = row.get("experiment_id")
    try:
        return int(exp_id)
    except Exception:
        return str(exp_id)


def task_check(sample_id: str, text: str):
    text = text or ""
    if sample_id in {"math_17x19"}:
        return "323" in text
    if sample_id == "math_answer_first":
        return "8" in text
    if sample_id == "length_5_words":
        cleaned = text.replace("<eos>", " ").replace("<pad>", " ")
        return len([w for w in cleaned.strip().split() if w]) == 5
    if sample_id == "json_person":
        return "{" in text and "}" in text and '"name"' in text and '"age"' in text and '"city"' in text
    return None


def summarize_one(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("trace") or []
    accepted = [first_scalar(t.get("accepted_count")) for t in trace]
    entropy = [first_scalar(t.get("mean_entropy")) for t in trace]
    active = [x for x in accepted if x is not None and x > 0]
    generated = row.get("generated_text") or ""
    return {
        "sample_id": row.get("sample_id"),
        "experiment_id": row.get("experiment_id"),
        "experiment_name": row.get("experiment_name"),
        "alpha": row.get("alpha"),
        "seed": row.get("seed"),
        "latency_sec": row.get("latency_sec"),
        "tokens_per_forward": first_scalar(row.get("tokens_per_forward")),
        "num_steps": len(trace),
        "active_accept_steps": len(active),
        "mean_accept_active": safe_mean(active),
        "max_accept": max(active) if active else None,
        "first_entropy": entropy[0] if entropy else None,
        "last_entropy": entropy[-1] if entropy else None,
        "entropy_auc": sum(x for x in entropy if x is not None),
        "task_check": task_check(str(row.get("sample_id")), generated),
        "trace_error": row.get("trace_error"),
        "generated_text": generated.replace("\n", "\\n")[:500],
    }


def write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    summary = [summarize_one(r) for r in rows]
    path = out_dir / "summary.csv"
    fields = list(summary[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    print(f"[OK] summary: {path}")


def plot_metric(rows: list[dict[str, Any]], sample_id: str, metric: str, ylabel: str, out: Path) -> None:
    subset = sorted([r for r in rows if r["sample_id"] == sample_id], key=exp_sort_key)

    plt.figure(figsize=(9, 5))
    for r in subset:
        xs, ys = [], []
        for t in r.get("trace") or []:
            x = t.get("cur_step")
            y = first_scalar(t.get(metric))
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        plt.plot(xs, ys, marker="o", linewidth=1, markersize=3, label=f"{r['experiment_id']}:{r['experiment_name']}")

    plt.xlabel("Denoising step")
    plt.ylabel(ylabel)
    plt.title(f"{sample_id}: {ylabel}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[OK] figure: {out}")


def decode_piece(processor, token_id: int) -> str:
    try:
        return processor.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return str(token_id)


def chain_rows(processor, trace: list[dict[str, Any]], max_tokens: int, stride: int) -> list[dict[str, Any]]:
    canvas = None
    rows = []

    for idx, step in enumerate(trace):
        if idx % stride != 0 and idx != len(trace) - 1:
            continue

        accepted_positions = (step.get("accepted_positions") or [[]])[0]
        accepted_canvas = (step.get("accepted_canvas_token_ids") or [[]])[0]
        argmax_canvas = (step.get("argmax_token_ids") or [[]])[0]

        if canvas is None:
            length = len(accepted_canvas) if accepted_canvas else len(argmax_canvas)
            canvas = [None] * length

        for pos in accepted_positions:
            if 0 <= pos < len(canvas) and pos < len(accepted_canvas):
                canvas[pos] = accepted_canvas[pos]

        upto = min(max_tokens, len(canvas))
        accepted_text = "".join("□" if canvas[i] is None else decode_piece(processor, canvas[i]) for i in range(upto))
        draft_text = "".join(decode_piece(processor, argmax_canvas[i]) for i in range(min(upto, len(argmax_canvas))))

        rows.append({
            "step": step.get("cur_step"),
            "accepted_count": first_scalar(step.get("accepted_count")),
            "mean_entropy": first_scalar(step.get("mean_entropy")),
            "accepted_so_far": accepted_text,
            "argmax_draft": draft_text,
        })

    return rows


def write_chain_html(rows: list[dict[str, Any]], processor, out: Path, max_tokens: int, stride: int) -> None:
    parts = [
        "<html><head><meta charset='utf-8'><style>",
        """
body { font-family: Arial, sans-serif; margin: 24px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }
td, th { border: 1px solid #ddd; padding: 6px; vertical-align: top; }
th { background: #f5f5f5; }
pre { white-space: pre-wrap; word-break: break-word; margin: 0; font-family: Consolas, monospace; }
.small { color: #555; font-size: 12px; }
""",
        "</style></head><body>",
        "<h1>DiffusionGemma self-conditioning output chain</h1>",
        "<p><b>accepted_so_far</b> is cumulative accepted tokens. □ means not accepted yet. <b>argmax_draft</b> is the model's current top-token draft.</p>",
    ]

    for sample_id in sorted({r["sample_id"] for r in rows}):
        subset = sorted([r for r in rows if r["sample_id"] == sample_id], key=exp_sort_key)
        parts.append(f"<h2>{html.escape(str(sample_id))}</h2>")
        if subset:
            parts.append(f"<p><b>Prompt:</b> {html.escape(subset[0].get('prompt',''))}</p>")

        for r in subset:
            parts.append(f"<h3>{html.escape(str(r['experiment_id']))}: {html.escape(str(r['experiment_name']))}</h3>")
            parts.append(
                f"<p class='small'>latency={r.get('latency_sec'):.3f}s | "
                f"tokens_per_forward={html.escape(str(r.get('tokens_per_forward')))} | "
                f"trace_error={html.escape(str(r.get('trace_error')))}</p>"
            )
            parts.append("<table><tr><th>step</th><th>accepted_count</th><th>mean_entropy</th><th>accepted_so_far</th><th>argmax_draft</th></tr>")
            for cr in chain_rows(processor, r.get("trace") or [], max_tokens=max_tokens, stride=stride):
                parts.append(
                    "<tr>"
                    f"<td>{html.escape(str(cr['step']))}</td>"
                    f"<td>{html.escape(str(cr['accepted_count']))}</td>"
                    f"<td>{html.escape(str(cr['mean_entropy']))}</td>"
                    f"<td><pre>{html.escape(cr['accepted_so_far'])}</pre></td>"
                    f"<td><pre>{html.escape(cr['argmax_draft'])}</pre></td>"
                    "</tr>"
                )
            parts.append("</table>")
            parts.append("<p><b>Final generated text:</b></p>")
            parts.append(f"<pre>{html.escape(r.get('generated_text') or '')}</pre>")

    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"[OK] chain: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DiffusionGemma self-conditioning outputs")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["visual_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(cfg["paths"]["output_file"]))
    if not rows:
        raise SystemExit("[ERROR] no rows found")

    write_summary(rows, out_dir)

    for sample_id in sorted({r["sample_id"] for r in rows}):
        plot_metric(rows, sample_id, "accepted_count", "Accepted token count", out_dir / f"{sample_id}_accepted_count.png")
        plot_metric(rows, sample_id, "mean_entropy", "Mean entropy", out_dir / f"{sample_id}_mean_entropy.png")

    processor = AutoProcessor.from_pretrained(
        cfg["model_id"],
        local_files_only=bool(cfg["generation"].get("local_files_only", False)),
    )
    write_chain_html(
        rows,
        processor,
        out_dir / "chain.html",
        max_tokens=int(cfg["visual"]["max_chain_tokens"]),
        stride=int(cfg["visual"]["step_stride"]),
    )
    print(f"[DONE] visual outputs written to {out_dir}")


if __name__ == "__main__":
    main()
