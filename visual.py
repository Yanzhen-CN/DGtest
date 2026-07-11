from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

try:
    import yaml
except ImportError as exc:
    raise SystemExit("[ERROR] missing dependency: pyyaml") from exc

MASK_CHAR = "□"
BG = (248, 248, 248)
PANEL_BG = (255, 255, 255)
BORDER = (190, 190, 190)
TEXT = (30, 30, 30)
MUTED = (100, 100, 100)
GREEN = (28, 150, 70)


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
        try:
            obj = ast.literal_eval(text)
        except Exception:
            return []
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


def safe_name(value: Any) -> str:
    text = str(value if value is not None else "none").replace(".", "p")
    text = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    return text.strip("_") or "none"


def alpha_sort_key(name: str) -> float:
    if name == "none":
        return 9999.0
    text = name.removeprefix("alpha_").replace("p", ".")
    try:
        return float(text)
    except Exception:
        return 9999.0


def experiment_dirs(output_root: Path) -> list[Path]:
    dirs = []
    for path in output_root.iterdir() if output_root.exists() else []:
        if path.is_dir() and (path / "params.csv").exists() and (path / "trace.csv").exists():
            dirs.append(path)
    return sorted(dirs, key=lambda p: alpha_sort_key(p.name))


def load_outputs(output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    params: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    clean_traces: list[dict[str, Any]] = []
    for exp_dir in experiment_dirs(output_root):
        for row in read_csv(exp_dir / "params.csv"):
            row["_experiment_dir"] = exp_dir.name
            params.append(row)
        for row in read_csv(exp_dir / "trace.csv"):
            row["_experiment_dir"] = exp_dir.name
            traces.append(row)
        clean_path = exp_dir / "clean_trace.csv"
        if clean_path.exists():
            for row in read_csv(clean_path):
                row["_experiment_dir"] = exp_dir.name
                clean_traces.append(row)
    if not params:
        fail(f"no experiment CSVs found under: {output_root}")
    return params, traces, clean_traces


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
    if not rows:
        return
    path = out_dir / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] summary: {path}")


def plot_metric(traces: list[dict[str, Any]], sample_id: str, metric: str, ylabel: str, out: Path) -> None:
    subset = [r for r in traces if str(r.get("sample_id")) == sample_id]
    names = sorted({str(r.get("experiment_name")) for r in subset}, key=alpha_sort_key)
    plt.figure(figsize=(9, 5))
    for name in names:
        rows = [r for r in subset if str(r.get("experiment_name")) == name]
        rows.sort(key=lambda r: to_int(r.get("generation_step")) or 0)
        pairs = []
        for row in rows:
            x = to_int(row.get("generation_step"))
            y = to_float(row.get(metric))
            if x is not None and y is not None:
                pairs.append((x, y))
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


def write_chart_outputs(params: list[dict[str, Any]], traces: list[dict[str, Any]], out_dir: Path) -> None:
    chart_dir = out_dir / "chart"
    chart_dir.mkdir(parents=True, exist_ok=True)
    write_summary(params, traces, chart_dir)
    for sample_id in sorted({str(r.get("sample_id")) for r in params}):
        plot_metric(traces, sample_id, "accepted_count", "Accepted token count", chart_dir / f"{safe_name(sample_id)}_accepted_count.png")
        plot_metric(traces, sample_id, "mean_entropy", "Mean entropy", chart_dir / f"{safe_name(sample_id)}_mean_entropy.png")


def load_font(size: int, mono: bool = False) -> ImageFont.ImageFont:
    candidates = (["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf", "C:/Windows/Fonts/consola.ttf"] if mono else ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", "C:/Windows/Fonts/arial.ttf"])
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def group_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        sample = str(row.get("sample_id"))
        exp = str(row.get("experiment_name"))
        out.setdefault(sample, {}).setdefault(exp, []).append(row)
    for sample in out:
        for exp in out[sample]:
            out[sample][exp].sort(key=lambda r: to_int(r.get("generation_step")) or -1)
    return out


def build_true_frames(rows: list[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    frames: list[dict[str, Any]] = []
    first_input = [str(x) for x in parse_json_list(rows[0].get("input_canvas_tokens"))]
    if first_input:
        frames.append({"step": -1, "accepted_count": 0, "mean_entropy": None, "tokens": first_input[:max_tokens], "changed": list(range(min(len(first_input), max_tokens)))})
    prev_ids = [int(x) for x in parse_json_list(rows[0].get("input_canvas_token_ids"))]
    for row in rows:
        ids = [int(x) for x in parse_json_list(row.get("output_canvas_token_ids"))]
        toks = [str(x) for x in parse_json_list(row.get("output_canvas_tokens"))]
        changed = [i for i in range(min(max(len(prev_ids), len(ids)), max_tokens)) if (prev_ids[i] if i < len(prev_ids) else None) != (ids[i] if i < len(ids) else None)]
        frames.append({"step": to_int(row.get("generation_step")), "accepted_count": to_int(row.get("accepted_count")) or 0, "mean_entropy": to_float(row.get("mean_entropy")), "tokens": toks[:max_tokens], "changed": changed})
        prev_ids = ids
    return frames


def build_clean_frames(rows: list[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    frames = []
    for row in rows:
        tokens = [str(x) for x in parse_json_list(row.get("clean_canvas_tokens"))]
        changed = [int(x) for x in parse_json_list(row.get("newly_committed_positions"))]
        frames.append({"step": to_int(row.get("generation_step")), "accepted_count": to_int(row.get("accepted_count")) or 0, "mean_entropy": to_float(row.get("mean_entropy")), "tokens": tokens[:max_tokens], "changed": [x for x in changed if 0 <= x < max_tokens]})
    return frames


def frame_at(frames: list[dict[str, Any]], index: int, total: int) -> dict[str, Any]:
    if not frames:
        return {"step": None, "accepted_count": 0, "mean_entropy": None, "tokens": [], "changed": []}
    if total <= 1:
        return frames[-1]
    src = round(index * (len(frames) - 1) / (total - 1))
    return frames[max(0, min(len(frames) - 1, src))]


def draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, frame: dict[str, Any], max_tokens: int, clean_mode: bool) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, fill=PANEL_BG, outline=BORDER, width=2)
    title_font = load_font(22)
    meta_font = load_font(16)
    mono = load_font(18, mono=True)
    draw.text((x0 + 14, y0 + 10), title, fill=TEXT, font=title_font)
    entropy = frame.get("mean_entropy")
    meta = f"step={frame.get('step')}  accepted={frame.get('accepted_count')}  entropy={'' if entropy is None else f'{entropy:.4f}'}"
    draw.text((x0 + 14, y0 + 40), meta, fill=MUTED, font=meta_font)
    tokens = [str(x) for x in frame.get("tokens", [])]
    changed = set(frame.get("changed", []))
    cols = 72
    line_h = 24
    char_w = 11
    start_x = x0 + 14
    start_y = y0 + 68
    for idx, token in enumerate(tokens[:max_tokens]):
        row = idx // cols
        col = idx % cols
        if start_y + row * line_h > y1 - 28:
            break
        color = TEXT
        if clean_mode and token == MASK_CHAR:
            color = (180, 180, 180)
        if idx in changed:
            color = GREEN
        draw.text((start_x + col * char_w, start_y + row * line_h), token if token else " ", fill=color, font=mono)
    draw.text((x0 + 14, y1 - 24), "green = changed this frame", fill=GREEN, font=meta_font)


def render_compare_frame(sample_id: str, experiments: list[str], per_exp_frames: dict[str, list[dict[str, Any]]], frame_index: int, total_frames: int, max_tokens: int, clean_mode: bool) -> Image.Image:
    panel_h = 260
    width = 1280
    height = 90 + panel_h * len(experiments) + 30
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    title_font = load_font(30)
    sub_font = load_font(17)
    trace_name = "clean commit view" if clean_mode else "true sampler canvas"
    draw.text((30, 20), f"{sample_id} — {trace_name}", fill=TEXT, font=title_font)
    draw.text((30, 58), f"frame {frame_index + 1}/{total_frames}", fill=MUTED, font=sub_font)
    y = 88
    for exp in experiments:
        frame = frame_at(per_exp_frames.get(exp, []), frame_index, total_frames)
        draw_panel(draw, (24, y, width - 24, y + panel_h - 12), exp, frame, max_tokens, clean_mode)
        y += panel_h
    return img


def save_gif(images: list[Image.Image], gif_path: Path, final_path: Path, fps: float, final_hold_seconds: float) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(40, int(round(1000 / max(0.1, fps))))
    durations = [duration] * len(images)
    durations[-1] = max(duration, int(round(final_hold_seconds * 1000)))
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=False, disposal=2)
    images[-1].save(final_path)
    print(f"[OK] gif: {gif_path}")
    print(f"[OK] final: {final_path}")


def write_compare_gifs(params: list[dict[str, Any]], traces: list[dict[str, Any]], clean_traces: list[dict[str, Any]], out_dir: Path, max_tokens: int, stride: int, fps: float, final_hold_seconds: float) -> None:
    true_groups = group_rows(traces)
    clean_groups = group_rows(clean_traces)
    sample_ids = sorted({str(r.get("sample_id")) for r in params})
    for sample_id in sample_ids:
        exp_names = sorted({str(r.get("experiment_name")) for r in params if str(r.get("sample_id")) == sample_id}, key=alpha_sort_key)
        sample_dir = out_dir / "compare" / safe_name(sample_id)
        true_frames = {exp: build_true_frames(true_groups.get(sample_id, {}).get(exp, []), max_tokens) for exp in exp_names}
        if stride > 1:
            true_frames = {k: v[::stride] for k, v in true_frames.items()}
        true_count = max((len(v) for v in true_frames.values()), default=0)
        if true_count:
            imgs = [render_compare_frame(sample_id, exp_names, true_frames, i, true_count, max_tokens, False) for i in range(true_count)]
            save_gif(imgs, sample_dir / "true_trace_compare.gif", sample_dir / "true_trace_final.png", fps, final_hold_seconds)
        clean_frames = {exp: build_clean_frames(clean_groups.get(sample_id, {}).get(exp, []), max_tokens) for exp in exp_names}
        if stride > 1:
            clean_frames = {k: v[::stride] for k, v in clean_frames.items()}
        clean_count = max((len(v) for v in clean_frames.values()), default=0)
        if clean_count:
            imgs = [render_compare_frame(sample_id, exp_names, clean_frames, i, clean_count, max_tokens, True) for i in range(clean_count)]
            save_gif(imgs, sample_dir / "clean_trace_compare.gif", sample_dir / "clean_trace_final.png", fps, final_hold_seconds)


def resolve_mode(arg_mode: str | None, cfg: dict[str, Any]) -> str:
    mode = (arg_mode or cfg.get("visual", {}).get("mode", "all")).lower()
    aliases = {"charts": "chart", "animation": "dynamic"}
    mode = aliases.get(mode, mode)
    if mode not in {"chart", "dynamic", "all"}:
        fail("visual mode must be one of: chart, dynamic, all")
    return mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DiffusionGemma self-conditioning alpha sweep")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("-m", "--mode", choices=["chart", "dynamic", "all", "charts", "animation"], default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--final-hold-seconds", type=float, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_root = Path(cfg["paths"]["output_root"])
    out_dir = Path(cfg["paths"].get("visual_dir", "visual"))
    out_dir.mkdir(parents=True, exist_ok=True)
    params, traces, clean_traces = load_outputs(output_root)
    mode = resolve_mode(args.mode, cfg)
    visual_cfg = cfg.get("visual", {})
    max_tokens = int(visual_cfg.get("max_chain_tokens", 256))
    stride = max(1, int(visual_cfg.get("step_stride", 1)))
    fps = float(args.fps if args.fps is not None else visual_cfg.get("fps", 2))
    final_hold_seconds = float(args.final_hold_seconds if args.final_hold_seconds is not None else visual_cfg.get("final_hold_seconds", 3))
    if mode in {"chart", "all"}:
        write_chart_outputs(params, traces, out_dir)
    if mode in {"dynamic", "all"}:
        write_compare_gifs(params, traces, clean_traces, out_dir, max_tokens, stride, fps, final_hold_seconds)
    print(f"[DONE] visual outputs written to {out_dir}")


if __name__ == "__main__":
    main()
