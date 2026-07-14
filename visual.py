#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from PIL import Image, ImageDraw, ImageFont

try:
    import yaml
except ImportError as exc:
    raise SystemExit("[ERROR] missing dependency: pyyaml") from exc


MASK_CHAR = "□"

BG = (247, 247, 247)
PANEL_BG = (255, 255, 255)
BORDER = (185, 185, 185)
TEXT = (25, 25, 25)
MUTED = (95, 95, 95)
MASK_FILL = (238, 238, 238)
CELL_BORDER = (210, 210, 210)
STRIP_UNTOUCHED = (225, 225, 225)
STRIP_CURRENT_MARK = (220, 66, 47)
STRIP_BORDER = (120, 120, 120)

# Token-state colors in the GIF token grid.
NOISE_TEXT = (145, 110, 70)          # unaccepted / noisy text
FIRST_ACCEPT_FILL = (196, 239, 205)  # current frame: first acceptance
FIRST_ACCEPT_TEXT = (30, 120, 52)
FINAL_TEXT = (25, 25, 25)            # previously accepted once, now stable


def fail(message: str) -> None:
    raise SystemExit(f"[ERROR] {message}")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        fail(f"bad config: {path}")
    return cfg


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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
        result = float(value)
        return result if math.isfinite(result) else None
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
    valid = [x for x in values if x is not None]
    return mean(valid) if valid else None


def safe_name(value: Any) -> str:
    text = str(value if value is not None else "none").replace(".", "p")
    text = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    return text.strip("_") or "none"


def alpha_sort_key(name: str) -> tuple[int, float]:
    if name in {"none", "alphanone"}:
        return (1, 9999.0)
    text = name.removeprefix("alpha_").removeprefix("alpha").replace("p", ".")
    try:
        return (0, -float(text))
    except Exception:
        return (0, 9999.0)


def trace_index_key(row: dict[str, Any]) -> tuple[int, int]:
    """Order rows inside one canvas, keeping an optional `.initial` row first."""
    raw = str(row.get("trace_index", "")).strip()
    if raw.endswith(".initial"):
        return (to_int(raw[:-8]) or 0, -1)
    return (to_int(raw) or 0, 0)


def trace_order_key(row: dict[str, Any]) -> tuple[int, int, int]:
    """Chronological order for current multi-canvas traces.

    New traces already contain a reliable canvas_index.  Sort by it first;
    trace_index may restart at zero for every canvas.
    """
    canvas = to_int(row.get("canvas_index"))
    trace_index, initial_rank = trace_index_key(row)
    return (canvas if canvas is not None else 0, trace_index, initial_rank)


def load_outputs(output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    params: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    exp_dirs = [
        path.parent for path in output_root.rglob("params.csv")
        if (path.parent / "trace.csv").exists()
    ] if output_root.exists() else []

    for exp_dir in sorted(exp_dirs, key=lambda p: (alpha_sort_key(p.parent.name), p.name)):
        params.extend(read_csv(exp_dir / "params.csv"))
        traces.extend(read_csv(exp_dir / "trace.csv"))

    if not params:
        fail(f"no experiment outputs found under: {output_root}")
    return params, traces


def output_groups(output_root: Path) -> list[tuple[Path, Path]]:
    """Return experiment groups for a root, a len* directory, or a step* directory."""
    if not output_root.exists():
        fail(f"output root does not exist: {output_root}")

    # Config may point directly at outputs/len1024/step48.
    if output_root.name.startswith("step") and next(output_root.rglob("params.csv"), None):
        return [(output_root, Path(output_root.parent.name) / output_root.name)]

    # Or at outputs/len1024.
    if output_root.name.startswith("len"):
        groups = [
            (step_dir, Path(output_root.name) / step_dir.name)
            for step_dir in sorted(output_root.glob("step*"))
            if step_dir.is_dir() and next(step_dir.rglob("params.csv"), None)
        ]
        if groups:
            return groups

    # Or at the common outputs root.
    groups: list[tuple[Path, Path]] = []
    for length_dir in sorted(output_root.glob("len*")):
        if not length_dir.is_dir():
            continue
        for step_dir in sorted(length_dir.glob("step*")):
            if step_dir.is_dir() and next(step_dir.rglob("params.csv"), None):
                groups.append((step_dir, step_dir.relative_to(output_root)))

    if not groups:
        # Last-resort support for an arbitrary directory that directly contains runs.
        if next(output_root.rglob("params.csv"), None):
            return [(output_root, Path(output_root.name))]
        fail(f"no experiment groups found under: {output_root}")
    return groups



def group_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id"))
        experiment = str(row.get("experiment_name"))
        grouped.setdefault(sample_id, {}).setdefault(experiment, []).append(row)

    for sample_group in grouped.values():
        for experiment_rows in sample_group.values():
            experiment_rows.sort(key=trace_order_key)
    return grouped


def normalize_canvas_indices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows in true canvas order and attach `_canvas_index`.

    Current DG traces explicitly record canvas_index, so this value must be
    trusted.  The old visualizer ignored it and tried to infer canvases only
    from generation_step resets; that collapses/interleaves 4-canvas traces
    whenever trace_index restarts or the step sequence is not a clean reset.

    For legacy traces without canvas_index, retain the step-reset fallback.
    """
    copied = [dict(row) for row in rows]
    explicit = [to_int(row.get("canvas_index")) for row in copied]

    if any(value is not None for value in explicit):
        for row, value in zip(copied, explicit):
            row["_canvas_index"] = value if value is not None else 0
        copied.sort(
            key=lambda row: (
                to_int(row.get("_canvas_index")) or 0,
                *trace_index_key(row),
            )
        )
        return copied

    # Legacy fallback: preserve trace order, then detect a countdown reset.
    copied.sort(key=lambda row: trace_index_key(row))
    canvas_index = 0
    previous_step: int | None = None
    for row in copied:
        step = to_int(row.get("generation_step")) or 0
        if previous_step is not None and step > previous_step:
            canvas_index += 1
        row["_canvas_index"] = canvas_index
        previous_step = step
    return copied


def canvas_layout(rows: list[dict[str, Any]]) -> tuple[dict[int, int], dict[int, int], int]:
    rows = normalize_canvas_indices(rows)
    lengths: dict[int, int] = {}
    for row in rows:
        canvas = to_int(row.get("_canvas_index", row.get("canvas_index"))) or 0
        lengths[canvas] = max(
            lengths.get(canvas, 0),
            len(parse_json_list(row.get("input_canvas_token_ids"))),
            len(parse_json_list(row.get("output_canvas_token_ids"))),
        )

    offsets: dict[int, int] = {}
    total = 0
    for canvas in sorted(lengths):
        offsets[canvas] = total
        total += lengths[canvas]
    return lengths, offsets, total




def canvas_boundaries(rows: list[dict[str, Any]]) -> tuple[list[int], list[tuple[int, int, int]], int]:
    """Return global token boundaries and (canvas, start, end) ranges."""
    ordered = normalize_canvas_indices(rows)
    lengths, offsets, total_tokens = canvas_layout(ordered)
    ranges = [
        (canvas, offsets[canvas], offsets[canvas] + lengths[canvas])
        for canvas in sorted(lengths)
    ]
    boundaries = [start for _, start, _ in ranges if start > 0]
    return boundaries, ranges, total_tokens


def canvas_trace_ranges(rows: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    """Return chronological trace index ranges for each canvas."""
    ordered = normalize_canvas_indices(rows)
    if not ordered:
        return []

    result: list[tuple[int, int, int]] = []
    start = 0
    current = to_int(ordered[0].get("_canvas_index")) or 0
    for index, row in enumerate(ordered[1:], start=1):
        canvas = to_int(row.get("_canvas_index")) or 0
        if canvas != current:
            result.append((current, start, index - 1))
            current = canvas
            start = index
    result.append((current, start, len(ordered) - 1))
    return result


def draw_token_canvas_separators(
    ax: Any,
    ranges: list[tuple[int, int, int]],
    label_y: float,
) -> None:
    """Draw subtle canvas boundaries and labels on token-position plots."""
    for canvas, start, end in ranges:
        if start > 0:
            ax.axvline(start - 0.5, color="0.45", linestyle="--", linewidth=1.1, alpha=0.75)
        center = (start + end - 1) / 2
        ax.text(
            center, label_y, f"C{canvas}",
            ha="center", va="bottom", fontsize=9, color="0.35",
            clip_on=False,
        )


def draw_step_canvas_separators(
    axes: Any,
    ranges: list[tuple[int, int, int]],
) -> None:
    """Draw subtle canvas boundaries and labels on chronological speed plots."""
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        label_y = ymax - 0.03 * (ymax - ymin if ymax > ymin else 1.0)
        for canvas, start, end in ranges:
            if start > 0:
                ax.axvline(start - 0.5, color="0.45", linestyle="--", linewidth=1.1, alpha=0.75)
            center = (start + end) / 2
            ax.text(
                center, label_y, f"C{canvas}",
                ha="center", va="top", fontsize=9, color="0.35",
            )


def extract_update_events(rows: list[dict[str, Any]]) -> tuple[list[dict[str, int]], int]:
    ordered = normalize_canvas_indices(rows)
    lengths, offsets, total_tokens = canvas_layout(ordered)

    update_count: dict[tuple[int, int], int] = {}
    events: list[dict[str, int]] = []

    for chronological_step, row in enumerate(ordered):
        canvas = to_int(row.get("_canvas_index", row.get("canvas_index"))) or 0

        # Accepted events are the only commit/revision events.
        # Never infer them from input_canvas != output_canvas, because that
        # includes re-noised positions.
        changed = [int(x) for x in parse_json_list(row.get("accepted_positions"))]
        ranks = [int(x) for x in parse_json_list(row.get("update_ranks"))]

        if len(ranks) != len(changed):
            ranks = []
            for position in changed:
                key = (canvas, position)
                rank = update_count.get(key, 0) + 1
                update_count[key] = rank
                ranks.append(rank)
        else:
            for position, rank in zip(changed, ranks):
                update_count[(canvas, position)] = max(
                    update_count.get((canvas, position), 0),
                    rank,
                )

        for position, rank in zip(changed, ranks):
            events.append({
                "trace_step": chronological_step,
                "generation_step": to_int(row.get("generation_step")) or 0,
                "canvas_index": canvas,
                "local_position": position,
                "global_position": offsets.get(canvas, 0) + position,
                "update_rank": rank,
            })

    return events, total_tokens


def write_events_csv(path: Path, events: list[dict[str, int]]) -> None:
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trace_step",
        "generation_step",
        "canvas_index",
        "local_position",
        "global_position",
        "update_rank",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)


def sample_experiments(params: list[dict[str, Any]], sample_id: str) -> list[str]:
    return sorted(
        {
            str(row.get("experiment_name"))
            for row in params
            if str(row.get("sample_id")) == sample_id
        },
        key=alpha_sort_key,
    )


def plot_position_step_figures(
    params: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    grouped = group_rows(traces)
    samples = sorted({str(row.get("sample_id")) for row in params})

    for sample_id in samples:
        experiments = sample_experiments(params, sample_id)
        event_sets: dict[str, tuple[list[dict[str, int]], int]] = {}
        range_sets: dict[str, list[tuple[int, int, int]]] = {}
        max_step = 0
        max_tokens = 0

        for experiment in experiments:
            exp_rows = grouped.get(sample_id, {}).get(experiment, [])
            events, total_tokens = extract_update_events(exp_rows)
            _, ranges, _ = canvas_boundaries(exp_rows)
            event_sets[experiment] = (events, total_tokens)
            range_sets[experiment] = ranges
            max_tokens = max(max_tokens, total_tokens)
            if events:
                max_step = max(max_step, max(event["trace_step"] for event in events))

        cols = min(3, max(1, len(experiments)))
        rows_n = math.ceil(len(experiments) / cols)

        fig, axes = plt.subplots(
            rows_n, cols, figsize=(5.2 * cols, 3.8 * rows_n),
            squeeze=False, sharex=True, sharey=True,
        )
        for ax in axes.ravel()[len(experiments):]:
            ax.set_axis_off()

        for ax, experiment in zip(axes.ravel(), experiments):
            events, _ = event_sets[experiment]
            first = [event for event in events if event["update_rank"] == 1]
            if first:
                ax.scatter(
                    [event["global_position"] for event in first],
                    [event["trace_step"] for event in first],
                    color="#2b6cb0", s=18, alpha=0.90, linewidths=0,
                )
            ax.set_xlim(-1, max_tokens)
            ax.set_ylim(-1, max_step + 1)
            ax.set_title(experiment)
            ax.set_xlabel("Global token position")
            ax.set_ylabel("First accepted step")
            draw_token_canvas_separators(ax, range_sets[experiment], max_step + 0.25)
            ax.grid(True, alpha=0.22)

        fig.suptitle(f"{sample_id}: token position vs first accepted step")
        fig.tight_layout(rect=[0, 0, 0.97, 0.94])
        path = out_dir / "trace_distribution" / f"{safe_name(sample_id)}_first_accept.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] wrote {path}")

        fig, axes = plt.subplots(
            rows_n, cols, figsize=(5.2 * cols, 3.8 * rows_n),
            squeeze=False, sharex=True, sharey=True,
        )
        for ax in axes.ravel()[len(experiments):]:
            ax.set_axis_off()

        mappable = None
        max_rank = max(
            1,
            max(
                (event["update_rank"] for exp_events, _ in event_sets.values() for event in exp_events),
                default=1,
            ),
        )
        rank_norm = LogNorm(vmin=1, vmax=max_rank) if max_rank > 1 else Normalize(vmin=0, vmax=1)
        count_cmap = green_count_cmap()

        for ax, experiment in zip(axes.ravel(), experiments):
            events, _ = event_sets[experiment]
            if events:
                ranks = [event["update_rank"] for event in events]
                mappable = ax.scatter(
                    [event["global_position"] for event in events],
                    [event["trace_step"] for event in events],
                    c=ranks if max_rank > 1 else [1.0] * len(ranks),
                    cmap=count_cmap, norm=rank_norm,
                    s=[16 + 3 * min(rank - 1, 8) for rank in ranks],
                    alpha=0.96, linewidths=0,
                )

            ax.set_xlim(-1, max_tokens)
            ax.set_ylim(-1, max_step + 1)
            ax.set_title(experiment)
            ax.set_xlabel("Global token position")
            ax.set_ylabel("Accepted / revision step")
            draw_token_canvas_separators(ax, range_sets[experiment], max_step + 0.25)
            ax.grid(True, alpha=0.22)

            write_events_csv(
                out_dir / "trace_distribution" / "events" /
                safe_name(sample_id) / f"{safe_name(experiment)}.csv",
                events,
            )

        if mappable is not None:
            cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02)
            cbar.set_label("Cumulative acceptance / revision count (log scale)")
            ticks = count_ticks(max_rank)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([str(tick) for tick in ticks])

        fig.suptitle(f"{sample_id}: token position vs every accepted/revision step")
        fig.tight_layout(rect=[0, 0, 0.97, 0.94])
        path = out_dir / "trace_distribution" / f"{safe_name(sample_id)}_all_updates.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] wrote {path}")

def speed_records(
    params: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped = group_rows(traces)
    param_lookup = {
        (str(row.get("sample_id")), str(row.get("experiment_name"))): row
        for row in params
    }
    records: list[dict[str, Any]] = []

    for sample_id, by_experiment in grouped.items():
        for experiment, rows in by_experiment.items():
            rows = sorted(rows, key=trace_order_key)
            accepted = [to_float(row.get("accepted_count")) or 0.0 for row in rows]
            changed = [
                float(len(parse_json_list(row.get("changed_positions"))))
                if str(row.get("changed_positions", "")).strip()
                else float(len(extract_update_events([row])[0]))
                for row in rows
            ]
            latency = to_float(
                param_lookup.get((sample_id, experiment), {}).get("latency_sec")
            )
            _, _, total_tokens = canvas_layout(rows)

            records.append({
                "sample_id": sample_id,
                "experiment": experiment,
                "steps": len(rows),
                "mean_accepted_per_step": safe_mean(accepted) or 0.0,
                "mean_accepted_event_positions_per_step": safe_mean(changed) or 0.0,
                "latency_sec": latency or 0.0,
                "tokens_per_second": (
                    total_tokens / latency
                    if latency is not None and latency > 0 else 0.0
                ),
            })
    return records


def plot_speed_figures(
    params: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    true_grouped = group_rows(traces)
    samples = sorted({str(row.get("sample_id")) for row in params})

    for sample_id in samples:
        experiments = sample_experiments(params, sample_id)

        # One three-row figure per experiment. This keeps each experiment's
        # canvas boundaries exact even when alpha1 and none use different steps.
        for experiment in experiments:
            rows = normalize_canvas_indices(
                true_grouped.get(sample_id, {}).get(experiment, [])
            )
            if not rows:
                continue

            accepted = [to_float(row.get("accepted_count")) or 0.0 for row in rows]
            changed = [
                len(parse_json_list(row.get("changed_positions")))
                if str(row.get("changed_positions", "")).strip()
                else len(parse_json_list(row.get("accepted_positions")))
                for row in rows
            ]

            events, _ = extract_update_events(rows)
            visible_by_step: list[float] = []
            seen: set[int] = set()
            events_by_step: dict[int, list[int]] = {}
            for event in events:
                events_by_step.setdefault(event["trace_step"], []).append(event["global_position"])
            for step_index in range(len(rows)):
                seen.update(events_by_step.get(step_index, []))
                visible_by_step.append(float(len(seen)))

            fig, axes = plt.subplots(3, 1, figsize=(10.5, 11.5), sharex=True)
            axes[0].plot(range(len(accepted)), accepted, marker="o", markersize=3, linewidth=1.3)
            axes[1].plot(range(len(changed)), changed, marker="o", markersize=3, linewidth=1.3)
            axes[2].plot(range(len(visible_by_step)), visible_by_step, marker="o", markersize=3, linewidth=1.3)

            axes[0].set_title("Accepted tokens per decoder step")
            axes[0].set_ylabel("Accepted tokens")
            axes[1].set_title("Accepted positions per step")
            axes[1].set_ylabel("Accepted positions")
            axes[2].set_title("Positions accepted at least once")
            axes[2].set_ylabel("Cumulative accepted positions")
            axes[2].set_xlabel("Chronological trace step")

            for ax in axes:
                ax.grid(True, alpha=0.25)
            draw_step_canvas_separators(axes, canvas_trace_ranges(rows))

            fig.suptitle(f"{sample_id} — {experiment}")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            path = out_dir / "speed" / f"{safe_name(sample_id)}_{safe_name(experiment)}_speed.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] wrote {path}")

    records = speed_records(params, traces)
    by_experiment: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_experiment.setdefault(record["experiment"], []).append(record)

    summary = []
    for experiment in sorted(by_experiment, key=alpha_sort_key):
        group = by_experiment[experiment]
        summary.append({
            "experiment": experiment,
            "mean_steps": safe_mean([float(row["steps"]) for row in group]) or 0.0,
            "mean_accepted_per_step": safe_mean([float(row["mean_accepted_per_step"]) for row in group]) or 0.0,
            "mean_accepted_event_positions_per_step": safe_mean([float(row["mean_accepted_event_positions_per_step"]) for row in group]) or 0.0,
            "mean_latency_sec": safe_mean([float(row["latency_sec"]) for row in group]) or 0.0,
            "mean_tokens_per_second": safe_mean([float(row["tokens_per_second"]) for row in group]) or 0.0,
        })

    summary_path = out_dir / "speed" / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
        if summary:
            writer.writeheader()
            writer.writerows(summary)

    if summary:
        names = [row["experiment"] for row in summary]
        metrics = [
            ("mean_steps", "Mean decoder steps"),
            ("mean_accepted_per_step", "Mean accepted tokens / step"),
            ("mean_accepted_event_positions_per_step", "Mean accepted positions / step"),
            ("mean_latency_sec", "Mean latency (s)"),
            ("mean_tokens_per_second", "Approx. tokens / second"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        for ax in axes.ravel()[len(metrics):]:
            ax.set_axis_off()
        for ax, (key, title) in zip(axes.ravel(), metrics):
            ax.bar(names, [row[key] for row in summary])
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=25)
            ax.grid(True, axis="y", alpha=0.25)
        fig.suptitle("Speed and parallelism summary")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = out_dir / "speed" / "summary.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        print(f"[OK] wrote {path}")

def load_font(size: int, mono: bool = False) -> ImageFont.ImageFont:
    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
            "C:/Windows/Fonts/consola.ttf",
        ]
        if mono else
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def compact_token(token: str, limit: int = 6) -> str:
    token = str(token)
    token = token.replace("\n", "↵").replace("\r", "").replace("\t", "⇥")
    token = token.replace(" ", "·")
    if token == "":
        return "∅"
    if len(token) > limit:
        return token[: limit - 1] + "…"
    return token


def count_ticks(max_count: int) -> list[int]:
    """Readable logarithmic ticks: 1, 2, 4, 8, 16, 32, ..."""
    if max_count <= 0:
        return [1]
    ticks = [1]
    value = 2
    while value < max_count:
        ticks.append(value)
        value *= 2
    if ticks[-1] != max_count:
        ticks.append(max_count)
    return sorted(set(ticks))


def color_lerp(
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    return tuple(
        int(round(a + (b - a) * t))
        for a, b in zip(color_a, color_b)
    )


def green_gradient_color_for_count(
    count: int,
    max_count: int,
) -> tuple[int, int, int]:
    """Wide-span log-scaled gradient.

    Design target:
    - first repeated accepts start near green,
    - then move through cyan / blue,
    - then purple,
    - then near-black for very high revision counts.

    This keeps "green as the starting point" but introduces a much larger color
    span, so high-frequency revision regions stand out much more clearly than a
    green-only palette.
    """
    if count <= 0 or max_count <= 0:
        return STRIP_UNTOUCHED

    if max_count <= 1:
        t = 1.0
    else:
        t = float(LogNorm(vmin=1, vmax=max_count)(count))

    # Slight gamma stretch to make mid/high ranges separate more clearly.
    t = t ** 0.86

    # green -> teal/cyan -> blue -> purple -> near-black purple
    c1 = (145, 225, 110)   # green start
    c2 = (72, 203, 170)    # teal
    c3 = (58, 132, 224)    # blue
    c4 = (104, 72, 196)    # purple
    c5 = (26, 16, 48)      # near-black purple

    if t <= 0.24:
        return color_lerp(c1, c2, t / 0.24)
    if t <= 0.50:
        return color_lerp(c2, c3, (t - 0.24) / 0.26)
    if t <= 0.76:
        return color_lerp(c3, c4, (t - 0.50) / 0.26)
    return color_lerp(c4, c5, (t - 0.76) / 0.24)


def gradient_color_for_count(
    count: int,
    max_count: int,
) -> tuple[int, int, int]:
    return green_gradient_color_for_count(count, max_count)


def green_count_cmap() -> LinearSegmentedColormap:
    """Wide-span count colormap matching the GIF/grid/strip."""
    return LinearSegmentedColormap.from_list(
        "dg_green_count",
        [
            (145 / 255.0, 225 / 255.0, 110 / 255.0),
            (72 / 255.0, 203 / 255.0, 170 / 255.0),
            (58 / 255.0, 132 / 255.0, 224 / 255.0),
            (104 / 255.0, 72 / 255.0, 196 / 255.0),
            (26 / 255.0, 16 / 255.0, 48 / 255.0),
        ],
        N=256,
    )



def build_true_frames(
    rows: list[dict[str, Any]],
    global_max_accept_count: int,
) -> list[dict[str, Any]]:
    """Render the actual noisy output canvas.

    Token-grid semantics:
    - never accepted: noisy token text in a dedicated noise color;
    - accepted for the first time in this frame: green;
    - on the following frame, if still only accepted once: normal black text;
    - if accepted again later (revision / re-accept), use the viridis-log
      gradient by cumulative accept count.

    Bottom strip:
    - gray untouched;
    - viridis/log color = cumulative accepted/revision count;
    - red top mark = accepted now.
    """
    ordered = normalize_canvas_indices(rows)
    if not ordered:
        return []

    lengths, offsets, total_tokens = canvas_layout(ordered)
    current: dict[int, list[str]] = {}
    accept_counts: dict[int, int] = {}
    previous_changed: list[int] = []
    frames: list[dict[str, Any]] = []

    for row in ordered:
        canvas = to_int(row.get("_canvas_index", row.get("canvas_index"))) or 0
        offset = offsets.get(canvas, 0)

        if canvas not in current:
            initial = [
                str(x) for x in parse_json_list(row.get("input_canvas_tokens"))
            ]
            current[canvas] = initial

            flattened = ["" for _ in range(total_tokens)]
            for c, tokens in current.items():
                c_offset = offsets[c]
                flattened[c_offset:c_offset + len(tokens)] = tokens

            frames.append({
                "step": -1,
                "accepted_count": 0,
                "mean_entropy": None,
                "tokens": flattened,
                "changed": [],
                "revisions": [],
                "accept_count_map": dict(accept_counts),
                "max_accept_count": global_max_accept_count,
                "previous_changed": [],
                "total_tokens": total_tokens,
                "canvas_lengths": dict(lengths),
                "canvas_offsets": dict(offsets),
                "started_canvases": sorted(current),
                "active_canvas": canvas,
            })

        output_tokens = [
            str(x) for x in parse_json_list(row.get("output_canvas_tokens"))
        ]
        current[canvas] = output_tokens

        accepted_local = [
            int(x) for x in parse_json_list(row.get("accepted_positions"))
        ]
        ranks = [int(x) for x in parse_json_list(row.get("update_ranks"))]
        if len(ranks) != len(accepted_local):
            ranks = [1] * len(accepted_local)

        accepted_global = [offset + p for p in accepted_local]
        revisions_global = [
            offset + p
            for p, rank in zip(accepted_local, ranks)
            if rank > 1
        ]

        for position in accepted_global:
            accept_counts[position] = accept_counts.get(position, 0) + 1

        flattened = ["" for _ in range(total_tokens)]
        for c, tokens in current.items():
            c_offset = offsets[c]
            flattened[c_offset:c_offset + len(tokens)] = tokens

        frames.append({
            "step": to_int(row.get("generation_step")),
            "accepted_count": len(accepted_local),
            "mean_entropy": to_float(row.get("mean_entropy")),
            "tokens": flattened,
            "changed": accepted_global,
            "revisions": revisions_global,
            "accept_count_map": dict(accept_counts),
            "max_accept_count": global_max_accept_count,
            "previous_changed": list(previous_changed),
            "total_tokens": total_tokens,
            "canvas_lengths": dict(lengths),
            "canvas_offsets": dict(offsets),
            "started_canvases": sorted(current),
            "active_canvas": canvas,
        })

        previous_changed = list(accepted_global)

    return frames



def aligned_frame(frames: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if not frames:
        return {
            "step": None,
            "accepted_count": 0,
            "mean_entropy": None,
            "tokens": [],
            "changed": [],
            "revisions": [],
            "accept_count_map": {},
            "max_accept_count": 0,
            "previous_changed": [],
            "total_tokens": 0,
            "canvas_lengths": {},
            "canvas_offsets": {},
            "started_canvases": [],
            "active_canvas": None,
        }
    return frames[min(index, len(frames) - 1)]


def token_grid_geometry(total_tokens: int) -> tuple[int, int, int, int]:
    if total_tokens <= 256:
        cols = 16
    elif total_tokens <= 512:
        cols = 24
    else:
        cols = 32
    rows = max(1, math.ceil(total_tokens / cols))
    cell_w = 54
    cell_h = 31
    return cols, rows, cell_w, cell_h


def draw_token_panel(
    image: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    frame: dict[str, Any],
) -> None:
    """Draw one experiment with every 256-token canvas in one panel.

    For four canvases, the grids are stacked vertically and separated by a
    labelled boundary.  Completed canvases remain visible while the active
    canvas denoises; future canvases stay blank until their first trace row.
    """
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, fill=PANEL_BG, outline=BORDER, width=2)

    title_font = load_font(22)
    meta_font = load_font(15)
    token_font = load_font(13, mono=True)
    index_font = load_font(9, mono=True)
    canvas_font = load_font(14)

    draw.text((x0 + 14, y0 + 10), title, fill=TEXT, font=title_font)
    entropy = frame.get("mean_entropy")
    entropy_text = "" if entropy is None else f"{entropy:.4f}"
    active_canvas = frame.get("active_canvas")
    canvas_text = "" if active_canvas is None else f"canvas={active_canvas}   "
    meta = (
        f"{canvas_text}local_step={frame.get('step')}   "
        f"accepted={frame.get('accepted_count')}   "
        f"entropy={entropy_text}"
    )
    draw.text((x0 + 14, y0 + 42), meta, fill=MUTED, font=meta_font)

    tokens = [str(x) for x in frame.get("tokens", [])]
    changed = set(int(x) for x in frame.get("changed", []))
    total_tokens = max(len(tokens), int(frame.get("total_tokens", 0)))
    cols, rows, cell_w, cell_h = token_grid_geometry(total_tokens)

    canvas_lengths = {
        int(k): int(v) for k, v in dict(frame.get("canvas_lengths", {})).items()
    }
    canvas_offsets = {
        int(k): int(v) for k, v in dict(frame.get("canvas_offsets", {})).items()
    }
    if not canvas_lengths and total_tokens:
        # Legacy/single-canvas trace.
        canvas_lengths = {0: total_tokens}
        canvas_offsets = {0: 0}

    started_canvases = set(int(x) for x in frame.get("started_canvases", []))
    multi_canvas = len(canvas_lengths) > 1
    left_gutter = 68 if multi_canvas else 0
    grid_x = x0 + 14 + left_gutter
    grid_y = y0 + 70

    count_map = {
        int(k): int(v)
        for k, v in dict(frame.get("accept_count_map", {})).items()
    }
    max_accept_count = int(frame.get("max_accept_count", 0) or 0)
    previous_changed = set(int(x) for x in frame.get("previous_changed", []))

    # Map global positions back to canvas-local positions.
    canvas_ranges = [
        (canvas, canvas_offsets[canvas], canvas_offsets[canvas] + length)
        for canvas, length in sorted(canvas_lengths.items())
    ]

    def position_info(index: int) -> tuple[int, int]:
        for canvas, start_pos, end_pos in canvas_ranges:
            if start_pos <= index < end_pos:
                return canvas, index - start_pos
        return 0, index

    # Canvas labels and boundaries.
    for canvas, start_pos, end_pos in canvas_ranges:
        start_row = start_pos // cols
        label_y = grid_y + start_row * cell_h + 4
        label_fill = STRIP_CURRENT_MARK if canvas == active_canvas else MUTED
        if multi_canvas:
            draw.text(
                (x0 + 16, label_y),
                f"Canvas {canvas}",
                fill=label_fill,
                font=canvas_font,
            )
        if start_pos > 0:
            boundary_y = grid_y + start_row * cell_h - 3
            draw.line(
                (grid_x, boundary_y, grid_x + cols * cell_w - 2, boundary_y),
                fill=STRIP_BORDER,
                width=3,
            )

    for index in range(total_tokens):
        row = index // cols
        col = index % cols
        cx0 = grid_x + col * cell_w
        cy0 = grid_y + row * cell_h
        cx1 = cx0 + cell_w - 2
        cy1 = cy0 + cell_h - 2

        canvas, local_index = position_info(index)
        canvas_started = canvas in started_canvases
        token = tokens[index] if index < len(tokens) else ""
        count = count_map.get(index, 0)

        if not canvas_started:
            fill = MASK_FILL
            text_fill = MUTED
        elif token == MASK_CHAR:
            fill = MASK_FILL
            text_fill = MUTED
        elif count <= 0:
            fill = PANEL_BG
            text_fill = NOISE_TEXT
        elif count == 1 and index in changed:
            fill = FIRST_ACCEPT_FILL
            text_fill = FIRST_ACCEPT_TEXT
        elif count == 1:
            fill = PANEL_BG
            text_fill = FINAL_TEXT
        else:
            fill = gradient_color_for_count(count, max_accept_count)
            text_fill = TEXT

        outline = STRIP_CURRENT_MARK if index in changed else CELL_BORDER
        outline_width = 3 if index in changed else 1
        draw.rectangle(
            (cx0, cy0, cx1, cy1),
            fill=fill,
            outline=outline,
            width=outline_width,
        )

        if index in previous_changed and index not in changed and count == 1:
            draw.rectangle(
                (cx0, cy0, cx1, cy1),
                fill=PANEL_BG,
                outline=CELL_BORDER,
                width=1,
            )

        draw.text((cx0 + 2, cy0 + 1), str(local_index), fill=MUTED, font=index_font)

        # Future canvases should be visibly empty, not filled with ∅ symbols.
        if not canvas_started:
            continue

        shown = compact_token(token)
        bbox = draw.textbbox((0, 0), shown, font=token_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        tx = cx0 + max(2, (cell_w - text_w) // 2)
        ty = cy0 + max(8, (cell_h - text_h) // 2 + 4)
        draw.text((tx, ty), shown, fill=text_fill, font=token_font)

    # Token-position history strip: gradient = cumulative accepted-count.
    strip_x0 = grid_x
    strip_x1 = x1 - 14
    strip_y0 = grid_y + rows * cell_h + 14
    strip_y1 = strip_y0 + 20

    revisions = set(int(x) for x in frame.get("revisions", []))

    draw.text(
        (strip_x0, strip_y0 - 18),
        "accepted-position strip (four canvases concatenated; boundaries marked)",
        fill=MUTED,
        font=meta_font,
    )

    width = max(1, strip_x1 - strip_x0)
    for index in range(total_tokens):
        px0 = strip_x0 + round(index * width / total_tokens)
        px1 = strip_x0 + round((index + 1) * width / total_tokens)
        count = count_map.get(index, 0)
        fill = gradient_color_for_count(count, max_accept_count)
        draw.rectangle(
            (px0, strip_y0, max(px0, px1 - 1), strip_y1),
            fill=fill,
        )
        if index in changed:
            marker_h = 5 if index not in revisions else 8
            draw.rectangle(
                (px0, strip_y0, max(px0, px1 - 1), strip_y0 + marker_h),
                fill=STRIP_CURRENT_MARK,
            )

    # Draw canvas boundaries and labels on the strip.
    for canvas, start_pos, _ in canvas_ranges:
        px = strip_x0 + round(start_pos * width / max(1, total_tokens))
        if start_pos > 0:
            draw.line((px, strip_y0, px, strip_y1), fill=STRIP_BORDER, width=2)
        draw.text((px + 2, strip_y1 + 2), f"C{canvas}", fill=MUTED, font=index_font)

    draw.rectangle(
        (strip_x0, strip_y0, strip_x1, strip_y1),
        outline=STRIP_BORDER,
        width=1,
    )
    draw.text(
        (strip_x0, strip_y1 + 15),
        "brown=noisy | green=first accept | black=stable | cyan/blue/purple=re-accept | red=accepted now",
        fill=MUTED,
        font=meta_font,
    )

    legend_y0 = strip_y1 + 40
    legend_y1 = legend_y0 + 16
    legend_w = min(340, max(180, strip_x1 - strip_x0))
    legend_steps = max(2, legend_w)

    for pixel in range(legend_steps):
        if max_accept_count <= 1:
            count_value = 1
        else:
            ratio = pixel / max(1, legend_steps - 1)
            count_value = max(
                1,
                int(round(math.exp(ratio * math.log(max_accept_count)))),
            )
        color = gradient_color_for_count(count_value, max_accept_count)
        px = strip_x0 + pixel
        draw.line((px, legend_y0, px, legend_y1), fill=color)

    draw.rectangle(
        (strip_x0, legend_y0, strip_x0 + legend_steps - 1, legend_y1),
        outline=STRIP_BORDER,
        width=1,
    )
    draw.text((strip_x0, legend_y0 - 18), "count legend", fill=MUTED, font=meta_font)

    for tick in count_ticks(max_accept_count):
        ratio = 0.0 if max_accept_count <= 1 else math.log(tick) / math.log(max_accept_count)
        px = strip_x0 + round(ratio * (legend_steps - 1))
        draw.line((px, legend_y1, px, legend_y1 + 5), fill=STRIP_BORDER)
        label = str(tick)
        bbox = draw.textbbox((0, 0), label, font=index_font)
        label_w = bbox[2] - bbox[0]
        draw.text((px - label_w / 2, legend_y1 + 6), label, fill=MUTED, font=index_font)

def render_compare_frame(
    sample_id: str,
    experiments: list[str],
    per_experiment_frames: dict[str, list[dict[str, Any]]],
    frame_index: int,
    total_frames: int,
) -> Image.Image:
    total_tokens = max(
        (
            int(frames[-1].get("total_tokens", 0))
            for frames in per_experiment_frames.values()
            if frames
        ),
        default=256,
    )
    cols, rows, cell_w, cell_h = token_grid_geometry(total_tokens)
    canvas_count = max(
        (
            len(dict(frames[-1].get("canvas_lengths", {})))
            for frames in per_experiment_frames.values()
            if frames
        ),
        default=1,
    )
    left_gutter = 68 if canvas_count > 1 else 0
    panel_w = 28 + left_gutter + cols * cell_w
    panel_h = 225 + rows * cell_h

    grid_cols = min(2, max(1, len(experiments)))
    grid_rows = math.ceil(len(experiments) / grid_cols)

    width = 36 + grid_cols * panel_w + (grid_cols - 1) * 18
    height = 92 + grid_rows * panel_h + (grid_rows - 1) * 18 + 24

    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title_font = load_font(28)
    subtitle_font = load_font(16)

    draw.text((22, 16), f"{sample_id} — sampler trace", fill=TEXT, font=title_font)
    draw.text(
        (22, 53),
        f"global frame {frame_index + 1}/{total_frames}; each canvas keeps its own local denoising steps",
        fill=MUTED,
        font=subtitle_font,
    )

    for index, experiment in enumerate(experiments):
        row = index // grid_cols
        col = index % grid_cols
        x0 = 18 + col * (panel_w + 18)
        y0 = 84 + row * (panel_h + 18)
        frame = aligned_frame(per_experiment_frames.get(experiment, []), frame_index)
        draw_token_panel(
            image,
            (x0, y0, x0 + panel_w, y0 + panel_h),
            experiment,
            frame,
        )

    return image


def save_gif_streaming(
    render_frame: Any,
    total_frames: int,
    gif_path: Path,
    final_path: Path,
    fps: float,
    final_hold_seconds: float,
) -> None:
    """Write a GIF without keeping hundreds of 1024-token frames in RAM."""
    if total_frames <= 0:
        return

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(40, int(round(1000 / max(0.1, fps))))
    durations = [duration for _ in range(total_frames)]
    durations[-1] = max(duration, int(round(final_hold_seconds * 1000)))

    first = render_frame(0)

    def remaining_frames():
        for frame_index in range(1, total_frames):
            yield render_frame(frame_index)

    first.save(
        gif_path,
        save_all=True,
        append_images=remaining_frames(),
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )

    final_image = render_frame(total_frames - 1)
    final_image.save(final_path)
    print(f"[OK] wrote {gif_path}")
    print(f"[OK] wrote {final_path}")

def write_gifs(
    params: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    out_dir: Path,
    stride: int,
    fps: float,
    final_hold_seconds: float,
) -> None:
    grouped = group_rows(traces)
    samples = sorted({str(row.get("sample_id")) for row in params})

    for sample_id in samples:
        experiments = sample_experiments(params, sample_id)
        sample_dir = out_dir / "compare" / safe_name(sample_id)

        sample_max_accept_count = 1
        for experiment in experiments:
            events, _ = extract_update_events(
                grouped.get(sample_id, {}).get(experiment, [])
            )
            if events:
                sample_max_accept_count = max(
                    sample_max_accept_count,
                    max(event["update_rank"] for event in events),
                )

        frames_by_experiment = {
            experiment: build_true_frames(
                grouped.get(sample_id, {}).get(experiment, []),
                global_max_accept_count=sample_max_accept_count,
            )[::stride]
            for experiment in experiments
        }
        total = max(
            (len(frames) for frames in frames_by_experiment.values()),
            default=0,
        )
        if not total:
            continue

        def render_frame(frame_index: int) -> Image.Image:
            return render_compare_frame(
                sample_id,
                experiments,
                frames_by_experiment,
                frame_index,
                total,
            )

        save_gif_streaming(
            render_frame,
            total,
            sample_dir / "trace_compare.gif",
            sample_dir / "trace_final.png",
            fps,
            final_hold_seconds,
        )



def resolve_mode(raw: str | None, cfg: dict[str, Any]) -> str:
    mode = (raw or cfg.get("visual", {}).get("mode", "all")).lower()
    aliases = {"charts": "chart", "animation": "dynamic"}
    mode = aliases.get(mode, mode)
    if mode not in {"chart", "dynamic", "all"}:
        fail("mode must be chart, dynamic, or all")
    return mode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DiffusionGemma trace visualizer"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "-m", "--mode",
        choices=["chart", "dynamic", "all", "charts", "animation"],
        default=None,
    )
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--final-hold-seconds", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = Path(cfg["paths"]["output_root"])
    out_dir = Path(cfg["paths"].get("visual_dir", "visual"))
    mode = resolve_mode(args.mode, cfg)
    visual_cfg = cfg.get("visual", {})
    stride = max(1, int(visual_cfg.get("step_stride", 1)))
    fps = float(
        args.fps if args.fps is not None
        else visual_cfg.get("fps", 2)
    )
    final_hold_seconds = float(
        args.final_hold_seconds if args.final_hold_seconds is not None
        else visual_cfg.get("final_hold_seconds", 3)
    )

    for data_dir, relative_group in output_groups(output_root):
        group_out_dir = out_dir / relative_group
        group_out_dir.mkdir(parents=True, exist_ok=True)
        params, traces = load_outputs(data_dir)

        if mode in {"chart", "all"}:
            plot_position_step_figures(params, traces, group_out_dir)
            plot_speed_figures(params, traces, group_out_dir)

        if mode in {"dynamic", "all"}:
            write_gifs(
                params,
                traces,
                group_out_dir,
                stride,
                fps,
                final_hold_seconds,
            )

    print(f"[DONE] visual outputs written to {out_dir}")


if __name__ == "__main__":
    main()
