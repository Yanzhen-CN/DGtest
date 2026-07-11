from __future__ import annotations

import argparse
import ast
import csv
import html
import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt

try:
    import yaml
except ImportError as exc:
    raise SystemExit("[ERROR] missing dependency: pyyaml") from exc


def fail(msg: str) -> None:
    raise SystemExit(f"[ERROR] {msg}")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        fail(f"bad config: {path}")
    return cfg


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
    except Exception:
        obj = ast.literal_eval(text)
    return list(obj) if isinstance(obj, (list, tuple)) else []


def to_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def to_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except Exception:
        return None


def safe_mean(values: list[float | None]) -> float | None:
    xs = [x for x in values if x is not None]
    return mean(xs) if xs else None


def experiment_dirs(output_root: Path) -> list[Path]:
    dirs = []
    for path in output_root.iterdir() if output_root.exists() else []:
        if path.is_dir() and (path / "params.csv").exists() and (path / "trace.csv").exists():
            dirs.append(path)
    return sorted(dirs, key=lambda p: alpha_sort_key(p.name))


def alpha_sort_key(name: str) -> float:
    text = name.removeprefix("alpha_").replace("p", ".")
    try:
        return float(text)
    except Exception:
        return 9999.0


def load_outputs(output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    params: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for exp_dir in experiment_dirs(output_root):
        for row in read_csv(exp_dir / "params.csv"):
            row["_experiment_dir"] = exp_dir.name
            params.append(row)
        for row in read_csv(exp_dir / "trace.csv"):
            row["_experiment_dir"] = exp_dir.name
            traces.append(row)
    if not params:
        fail(f"no experiment CSVs found under: {output_root}")
    return params, traces


def summarize_params(params: list[dict[str, Any]], traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in traces:
        key = (str(row.get("sample_id")), str(row.get("experiment_name")))
        by_key.setdefault(key, []).append(row)

    summary = []
    for row in params:
        key = (str(row.get("sample_id")), str(row.get("experiment_name")))
        trace_rows = by_key.get(key, [])
        accepted = [to_float(r.get("accepted_count")) for r in trace_rows]
        entropy = [to_float(r.get("mean_entropy")) for r in trace_rows]
        active = [x for x in accepted if x is not None and x > 0]
        summary.append({
            "sample_id": row.get("sample_id"),
            "experiment_name": row.get("experiment_name"),
            "self_conditioning_alpha": row.get("self_conditioning_alpha"),
            "mode": row.get("mode"),
            "seed": row.get("seed"),
            "latency_sec": row.get("latency_sec"),
            "num_steps": len(trace_rows),
            "active_accept_steps": len(active),
            "mean_accept_active": safe_mean(active),
            "max_accept": max(active) if active else None,
            "first_entropy": entropy[0] if entropy else None,
            "last_entropy": entropy[-1] if entropy else None,
            "entropy_auc": sum(x for x in entropy if x is not None),
            "trace_error": row.get("trace_error"),
            "generated_text": (row.get("generated_text") or "").replace("\n", "\\n")[:500],
        })
    return summary


def write_summary(params: list[dict[str, Any]], traces: list[dict[str, Any]], out_dir: Path) -> None:
    rows = summarize_params(params, traces)
    path = out_dir / "summary.csv"
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] summary: {path}")


def plot_metric(traces: list[dict[str, Any]], sample_id: str, metric: str, ylabel: str, out: Path) -> None:
    subset = [r for r in traces if r.get("sample_id") == sample_id]
    names = sorted({str(r.get("experiment_name")) for r in subset}, key=alpha_sort_key)

    plt.figure(figsize=(9, 5))
    for name in names:
        rows = [r for r in subset if r.get("experiment_name") == name]
        rows.sort(key=lambda r: to_int(r.get("generation_step")) or 0)
        xs = [to_int(r.get("generation_step")) for r in rows]
        ys = [to_float(r.get(metric)) for r in rows]
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if pairs:
            plt.plot([x for x, _ in pairs], [y for _, y in pairs], marker="o", linewidth=1, markersize=3, label=name)

    plt.xlabel("Denoising step")
    plt.ylabel(ylabel)
    plt.title(f"{sample_id}: {ylabel}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"[OK] figure: {out}")


def chain_rows(trace_rows: list[dict[str, Any]], max_tokens: int, stride: int) -> list[dict[str, Any]]:
    rows = sorted(trace_rows, key=lambda r: to_int(r.get("generation_step")) or 0)
    canvas: list[str | None] | None = None
    out = []

    for idx, row in enumerate(rows):
        if idx % stride != 0 and idx != len(rows) - 1:
            continue

        positions = [int(x) for x in parse_json_list(row.get("accepted_positions"))]
        tokens = [str(x) for x in parse_json_list(row.get("accepted_tokens"))]

        if canvas is None:
            length = max(positions) + 1 if positions else max_tokens
            canvas = [None] * max(length, max_tokens)

        for pos, token in zip(positions, tokens):
            if pos >= len(canvas):
                canvas.extend([None] * (pos + 1 - len(canvas)))
            canvas[pos] = token

        upto = min(max_tokens, len(canvas))
        accepted_so_far = "".join("□" if canvas[i] is None else canvas[i] for i in range(upto))
        out.append({
            "step": row.get("generation_step"),
            "accepted_count": row.get("accepted_count"),
            "mean_entropy": row.get("mean_entropy"),
            "accepted_so_far": accepted_so_far,
        })

    return out


def write_chain_html(params: list[dict[str, Any]], traces: list[dict[str, Any]], out: Path, max_tokens: int, stride: int) -> None:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in traces:
        key = (str(row.get("sample_id")), str(row.get("experiment_name")))
        by_key.setdefault(key, []).append(row)

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
        "<h1>DiffusionGemma self-conditioning alpha sweep</h1>",
        "<p><b>accepted_so_far</b> is cumulative accepted text. □ means not accepted yet.</p>",
    ]

    for sample_id in sorted({str(r.get("sample_id")) for r in params}):
        sample_params = [r for r in params if str(r.get("sample_id")) == sample_id]
        sample_params.sort(key=lambda r: alpha_sort_key(str(r.get("experiment_name"))))
        parts.append(f"<h2>{html.escape(sample_id)}</h2>")
        if sample_params:
            parts.append(f"<p><b>Prompt:</b> {html.escape(sample_params[0].get('prompt') or '')}</p>")

        for row in sample_params:
            name = str(row.get("experiment_name"))
            key = (sample_id, name)
            parts.append(f"<h3>{html.escape(name)}</h3>")
            parts.append(
                f"<p class='small'>alpha={html.escape(str(row.get('self_conditioning_alpha')))} | "
                f"mode={html.escape(str(row.get('mode')))} | "
                f"latency={html.escape(str(row.get('latency_sec')))} | "
                f"trace_error={html.escape(str(row.get('trace_error')))}</p>"
            )
            parts.append("<table><tr><th>step</th><th>accepted_count</th><th>mean_entropy</th><th>accepted_so_far</th></tr>")
            for chain in chain_rows(by_key.get(key, []), max_tokens=max_tokens, stride=stride):
                parts.append(
                    "<tr>"
                    f"<td>{html.escape(str(chain['step']))}</td>"
                    f"<td>{html.escape(str(chain['accepted_count']))}</td>"
                    f"<td>{html.escape(str(chain['mean_entropy']))}</td>"
                    f"<td><pre>{html.escape(chain['accepted_so_far'])}</pre></td>"
                    "</tr>"
                )
            parts.append("</table>")
            parts.append("<p><b>Final generated text:</b></p>")
            parts.append(f"<pre>{html.escape(row.get('generated_text') or '')}</pre>")

    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"[OK] chain: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DiffusionGemma self-conditioning alpha sweep")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg["paths"]["output_root"])
    out_dir = Path(cfg["paths"].get("visual_dir", "visual"))
    out_dir.mkdir(parents=True, exist_ok=True)

    params, traces = load_outputs(output_root)
    write_summary(params, traces, out_dir)

    for sample_id in sorted({str(r.get("sample_id")) for r in params}):
        plot_metric(traces, sample_id, "accepted_count", "Accepted token count", out_dir / f"{sample_id}_accepted_count.png")
        plot_metric(traces, sample_id, "mean_entropy", "Mean entropy", out_dir / f"{sample_id}_mean_entropy.png")

    write_chain_html(
        params,
        traces,
        out_dir / "chain.html",
        max_tokens=int(cfg["visual"].get("max_chain_tokens", 80)),
        stride=max(1, int(cfg["visual"].get("step_stride", 1))),
    )
    print(f"[DONE] visual outputs written to {out_dir}")


if __name__ == "__main__":
    main()
