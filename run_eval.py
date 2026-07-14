from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

import run as dg
from eval_benchmarks import BENCHMARK_SPECS, load_benchmark


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise SystemExit(f"[ERROR] invalid config: {path}")
    return value


def read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(str(json.loads(line)["sample_id"]))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DiffusionGemma official benchmark outputs")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--eval-config", default="eval_config.yaml")
    parser.add_argument("--benchmarks", default=None, help="comma-separated benchmark names")
    parser.add_argument("--limit", type=int, default=None, help="limit each benchmark for a smoke run")
    args = parser.parse_args()

    root = Path(".").resolve()
    cfg = dg.load_config(args.config)
    eval_cfg = load_yaml(args.eval_config)
    dg.configure_huggingface_environment(cfg, root)
    if bool(eval_cfg.get("disable_xet_for_benchmarks", True)):
        # Must be set before importing transformers/datasets/huggingface_hub.
        # Public benchmark parquet downloads occasionally fail with Xet CAS
        # 401 on RunPod; regular Hub HTTP remains resumable and is sufficient
        # for these small datasets.
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    dg.patch_transformers(cfg, root)

    global torch
    import torch
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

    dg.torch = torch
    gen_cfg = cfg["generation"]
    eval_gen = eval_cfg["generation"]
    max_new_tokens = int(eval_gen.get("max_new_tokens", 512))
    seed = int(eval_gen.get("seed", gen_cfg.get("seed", 42)))
    os.environ["DG_TEMPERATURE_SCHEDULE_STEPS"] = str(
        int(gen_cfg.get("temperature_schedule_steps", 48))
    )
    benchmarks = (
        [x.strip() for x in args.benchmarks.split(",") if x.strip()]
        if args.benchmarks else list(eval_cfg["benchmarks"])
    )
    unknown = sorted(set(benchmarks) - set(BENCHMARK_SPECS))
    if unknown:
        raise SystemExit(f"[ERROR] unknown benchmarks: {unknown}")

    print(f"[LOAD] {cfg['model_id']}")
    processor = AutoProcessor.from_pretrained(
        cfg["model_id"], local_files_only=bool(gen_cfg.get("local_files_only", False))
    )
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        cfg["model_id"],
        dtype=gen_cfg.get("dtype", "auto"),
        device_map=gen_cfg.get("device_map", "auto"),
        local_files_only=bool(gen_cfg.get("local_files_only", False)),
    )
    model.eval()
    model._dg_trace_enabled = False

    output_root = root / eval_cfg.get("output_root", "eval/outputs")
    for benchmark in benchmarks:
        items = load_benchmark(benchmark, args.limit)
        print(f"[BENCHMARK] {benchmark}: {len(items)} samples")
        for experiment in eval_gen["experiments"]:
            name = str(experiment["name"])
            alpha_raw = experiment.get("alpha")
            alpha = None if alpha_raw is None or str(alpha_raw).lower() == "none" else float(alpha_raw)
            steps = int(experiment["steps"])
            output_path = output_root / benchmark / name / "predictions.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            completed = read_completed(output_path)
            model._dg_selfcond_mode = "none" if alpha is None else "scaled"
            model._dg_selfcond_alpha = 1.0 if alpha is None else alpha

            with output_path.open("a", encoding="utf-8") as handle:
                for index, item in enumerate(items):
                    if item["sample_id"] in completed:
                        continue
                    encoded = dg.encode_chat(processor, item["prompt"]).to(model.device)
                    prompt_len = encoded["input_ids"].shape[-1]
                    torch.manual_seed(seed + index)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed + index)
                        torch.cuda.synchronize()
                    start = time.time()
                    with torch.inference_mode():
                        output = model.generate(
                            **encoded,
                            max_new_tokens=max_new_tokens,
                            max_denoising_steps=steps,
                            disable_compile=True,
                            return_dict_in_generate=True,
                        )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    generated = processor.decode(
                        output.sequences[0, prompt_len:], skip_special_tokens=True
                    )
                    row = {
                        "benchmark": benchmark,
                        "experiment": name,
                        "alpha": alpha,
                        "max_denoising_steps": steps,
                        "max_new_tokens": max_new_tokens,
                        "seed": seed + index,
                        "latency_sec": time.time() - start,
                        "generated_text": generated,
                        **item,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(f"[OK] {benchmark} {name} {index + 1}/{len(items)} {item['sample_id']}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    print(f"[DONE] benchmark outputs: {output_root}")


if __name__ == "__main__":
    main()
