#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    label: str
    steps: int
    output_name: str


@dataclass
class EntropyTrace:
    sample_id: str
    experiment: ExperimentSpec
    entropy: np.ndarray
    mean_entropy: np.ndarray
    accepted_count: np.ndarray
    accepted_positions: list[list[int]]


def fail(message: str) -> None:
    raise SystemExit(f"[ERROR] {message}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        fail(f"invalid config: {path}")
    return value


def parse_json_list(value: Any) -> list[Any]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON trace field: {text[:120]!r}: {exc}")
    return parsed if isinstance(parsed, list) else []


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        fail(f"missing trace: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def trace_sort_key(row: dict[str, str]) -> tuple[int, int]:
    try:
        canvas = int(row.get("canvas_index", "0") or 0)
    except ValueError:
        canvas = 0
    raw_index = str(row.get("trace_index", "0")).split(".", 1)[0]
    try:
        index = int(raw_index)
    except ValueError:
        index = 0
    return canvas, index


def load_trace(
    output_root: Path,
    canvas_length: int,
    sample_id: str,
    spec: ExperimentSpec,
) -> EntropyTrace:
    path = (
        output_root
        / f"len{canvas_length}"
        / f"step{spec.steps}"
        / spec.output_name
        / sample_id
        / "trace.csv"
    )
    rows = sorted(read_csv(path), key=trace_sort_key)
    canvas_rows = [row for row in rows if trace_sort_key(row)[0] == 0]
    if not canvas_rows:
        fail(f"no canvas-0 rows in {path}")

    entropy_rows: list[list[float]] = []
    accepted_count: list[float] = []
    accepted_positions: list[list[int]] = []
    for row_index, row in enumerate(canvas_rows, start=1):
        values = [float(value) for value in parse_json_list(row.get("position_entropy"))]
        if len(values) != canvas_length:
            fail(
                f"{path}: step {row_index} has {len(values)} position entropies; "
                f"expected {canvas_length}"
            )
        entropy_rows.append(values)
        positions = [int(value) for value in parse_json_list(row.get("accepted_positions"))]
        accepted_positions.append(positions)
        raw_count = str(row.get("accepted_count", "")).strip()
        accepted_count.append(float(raw_count) if raw_count else float(len(positions)))

    entropy = np.asarray(entropy_rows, dtype=np.float64)
    return EntropyTrace(
        sample_id=sample_id,
        experiment=spec,
        entropy=entropy,
        mean_entropy=np.mean(entropy, axis=1),
        accepted_count=np.asarray(accepted_count, dtype=np.float64),
        accepted_positions=accepted_positions,
    )


def write_long_csv(path: Path, traces: list[EntropyTrace]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "experiment", "label", "step_budget", "local_step",
        "mean_entropy", "accepted_count", "accepted_positions",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trace in traces:
            for index in range(len(trace.mean_entropy)):
                writer.writerow({
                    "sample_id": trace.sample_id,
                    "experiment": trace.experiment.name,
                    "label": trace.experiment.label,
                    "step_budget": trace.experiment.steps,
                    "local_step": index + 1,
                    "mean_entropy": f"{trace.mean_entropy[index]:.9g}",
                    "accepted_count": f"{trace.accepted_count[index]:.9g}",
                    "accepted_positions": json.dumps(trace.accepted_positions[index]),
                })


def write_summary_csv(path: Path, traces: list[EntropyTrace]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "experiment", "label", "step_budget", "observed_steps",
        "initial_mean_entropy", "final_mean_entropy", "first_mean_entropy_le_0p1",
        "mean_accepted_count", "max_accepted_count", "entropy_accept_correlation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trace in traces:
            low = np.flatnonzero(trace.mean_entropy <= 0.1)
            if len(trace.mean_entropy) > 1 and np.std(trace.mean_entropy) > 0 and np.std(trace.accepted_count) > 0:
                correlation = float(np.corrcoef(trace.mean_entropy, trace.accepted_count)[0, 1])
            else:
                correlation = math.nan
            writer.writerow({
                "sample_id": trace.sample_id,
                "experiment": trace.experiment.name,
                "label": trace.experiment.label,
                "step_budget": trace.experiment.steps,
                "observed_steps": len(trace.mean_entropy),
                "initial_mean_entropy": f"{trace.mean_entropy[0]:.9g}",
                "final_mean_entropy": f"{trace.mean_entropy[-1]:.9g}",
                "first_mean_entropy_le_0p1": int(low[0]) + 1 if low.size else "",
                "mean_accepted_count": f"{np.mean(trace.accepted_count):.9g}",
                "max_accepted_count": f"{np.max(trace.accepted_count):.9g}",
                "entropy_accept_correlation": (
                    "" if math.isnan(correlation) else f"{correlation:.9g}"
                ),
            })


def plot_heatmaps(
    samples: list[str],
    specs: list[ExperimentSpec],
    trace_lookup: dict[tuple[str, str], EntropyTrace],
    canvas_length: int,
    baseline_steps: int,
    out_path: Path,
) -> None:
    all_values = np.concatenate(
        [trace.entropy.ravel() for trace in trace_lookup.values()]
    )
    finite = all_values[np.isfinite(all_values)]
    if finite.size == 0:
        fail("entropy traces contain no finite values")
    vmin = 0.0
    vmax = float(np.max(finite))
    cmap = plt.get_cmap("viridis").copy()

    fig, axes = plt.subplots(
        len(samples), len(specs),
        figsize=(5.4 * len(specs), 3.7 * len(samples)),
        sharex=True, sharey=False, squeeze=False,
        layout="constrained",
    )
    image = None
    for row_index, sample_id in enumerate(samples):
        for col_index, spec in enumerate(specs):
            ax = axes[row_index, col_index]
            trace = trace_lookup[(sample_id, spec.name)]
            observed_steps = len(trace.entropy)
            image = ax.imshow(
                trace.entropy,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=(-0.5, canvas_length - 0.5, 0.5, observed_steps + 0.5),
            )
            if observed_steps > baseline_steps:
                ax.axhline(
                    baseline_steps + 0.5,
                    color="#d62728",
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.9,
                )
            ax.text(
                canvas_length - 3,
                observed_steps - 0.2,
                f"stop {observed_steps}",
                ha="right",
                va="top",
                fontsize=8,
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
            )
            if row_index == 0:
                ax.set_title(spec.label)
            if col_index == 0:
                ax.set_ylabel(f"{sample_id}\nDenoising step")
            if row_index == len(samples) - 1:
                ax.set_xlabel("Token position")
            ax.set_xticks([0, 64, 128, 192, 255])
            ax.set_ylim(0.5, observed_steps + 0.5)

    if image is not None:
        colorbar = fig.colorbar(
            image, ax=axes.ravel().tolist(), fraction=0.022, pad=0.018,
            shrink=0.82,
        )
        colorbar.set_label("Token entropy (shared scale)")
    fig.suptitle("Self-conditioning and token-level entropy convergence")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_lines(
    samples: list[str],
    specs: list[ExperimentSpec],
    trace_lookup: dict[tuple[str, str], EntropyTrace],
    baseline_steps: int,
    out_path: Path,
    metric: str,
) -> None:
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
    linestyles = ["-", "--", ":", (0, (3, 1, 1, 1)), "-."]
    fig, axes = plt.subplots(
        1, len(samples), figsize=(5.2 * len(samples), 4.4),
        squeeze=False,
    )
    for ax, sample_id in zip(axes.ravel(), samples):
        for index, spec in enumerate(specs):
            trace = trace_lookup[(sample_id, spec.name)]
            values = trace.mean_entropy if metric == "mean_entropy" else trace.accepted_count
            steps = np.arange(1, len(values) + 1)
            ax.plot(
                steps,
                values,
                label=spec.label,
                color=colors[index % len(colors)],
                linestyle=linestyles[index % len(linestyles)],
                linewidth=1.8,
            )
        if max(spec.steps for spec in specs) > baseline_steps:
            ax.axvline(baseline_steps, color="0.45", linestyle=":", linewidth=1.2)
        ax.set_title(sample_id)
        ax.set_xlabel("Denoising step")
        ax.grid(True, alpha=0.25)

    axes[0, 0].set_ylabel(
        "Mean entropy over 256 positions"
        if metric == "mean_entropy"
        else "Accepted token count"
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(specs),
        frameon=False, bbox_to_anchor=(0.5, 0.015),
    )
    fig.suptitle(
        "Mean entropy vs denoising step"
        if metric == "mean_entropy"
        else "Accepted token count vs denoising step",
        y=0.975,
    )
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.18, top=0.82, wspace=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DiffusionGemma entropy convergence")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    entropy_cfg = cfg.get("entropy_visual", {})
    output_root = Path(args.output_root or cfg.get("paths", {}).get("output_root", "outputs"))
    out_dir = Path(args.out_dir or entropy_cfg.get("output_dir", "visual/entropy"))
    canvas_length = int(entropy_cfg.get("canvas_length", 256))
    baseline_steps = int(entropy_cfg.get("baseline_steps", 48))
    samples = [str(value) for value in entropy_cfg.get("samples", [])]
    if not samples:
        fail("entropy_visual.samples must be non-empty")

    specs = [
        ExperimentSpec(
            name=str(item["name"]),
            label=str(item.get("label", item["name"])),
            steps=int(item["steps"]),
            output_name=str(item.get("output_name", item["name"])),
        )
        for item in entropy_cfg.get("experiments", [])
    ]
    if not specs:
        fail("entropy_visual.experiments must be non-empty")

    traces = [
        load_trace(output_root, canvas_length, sample_id, spec)
        for sample_id in samples
        for spec in specs
    ]
    trace_lookup = {
        (trace.sample_id, trace.experiment.name): trace for trace in traces
    }

    write_long_csv(out_dir / "entropy_trace_long.csv", traces)
    write_summary_csv(out_dir / "entropy_summary.csv", traces)
    plot_heatmaps(
        samples,
        specs,
        trace_lookup,
        canvas_length,
        baseline_steps,
        out_dir / "entropy_heatmaps.png",
    )
    plot_lines(
        samples,
        specs,
        trace_lookup,
        baseline_steps,
        out_dir / "mean_entropy_vs_step.png",
        "mean_entropy",
    )
    plot_lines(
        samples,
        specs,
        trace_lookup,
        baseline_steps,
        out_dir / "accepted_tokens_vs_step.png",
        "accepted_count",
    )
    print(f"[DONE] entropy figures written to {out_dir}")


if __name__ == "__main__":
    main()
