from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DiffusionGemma benchmark scores")
    parser.add_argument("--eval-config", default="eval_config.yaml")
    args = parser.parse_args()
    with Path(args.eval_config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    scores_root = Path(cfg.get("scores_root", "eval/scores"))
    figures_root = Path(cfg.get("figures_root", "eval/figures"))
    figures_root.mkdir(parents=True, exist_ok=True)
    rows = read_csv(scores_root / "summary.csv")
    if not rows:
        raise SystemExit("[ERROR] summary.csv is empty")

    experiments = sorted({row["experiment"] for row in rows})
    benchmarks = [name for name in cfg.get("benchmarks", []) if any(r["benchmark"] == name for r in rows)]
    values = {(row["experiment"], row["benchmark"]): float(row["accuracy"]) for row in rows}
    width = 0.8 / max(1, len(experiments))
    x = list(range(len(benchmarks)))
    fig, ax = plt.subplots(figsize=(max(8, len(benchmarks) * 2.2), 5.5))
    for index, experiment in enumerate(experiments):
        offsets = [position - 0.4 + width / 2 + index * width for position in x]
        bars = ax.bar(
            offsets,
            [values.get((experiment, benchmark), 0.0) * 100 for benchmark in benchmarks],
            width,
            label=experiment,
        )
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xticks(x, benchmarks)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy / pass@1 (%)")
    ax.set_title("DiffusionGemma self-conditioning benchmark comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_root / "benchmark_accuracy.png", dpi=180)
    plt.close(fig)

    by_experiment: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_experiment[row["experiment"]].append(float(row["accuracy"]))
    fig, ax = plt.subplots(figsize=(7, 5))
    overall = [sum(by_experiment[name]) / len(by_experiment[name]) * 100 for name in experiments]
    bars = ax.bar(experiments, overall)
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Macro-average score (%)")
    ax.set_title("Overall benchmark score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_root / "overall_score.png", dpi=180)
    plt.close(fig)
    print(f"[DONE] figures: {figures_root}")


if __name__ == "__main__":
    main()
