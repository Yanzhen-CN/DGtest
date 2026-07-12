from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
VENDORED_TRANSFORMERS_SRC = REPO_ROOT / "vendor" / "transformers" / "src"
sys.path.insert(0, str(VENDORED_TRANSFORMERS_SRC))

import torch
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

try:
    import yaml
except ImportError as exc:
    raise SystemExit("[ERROR] missing dependency: pyyaml") from exc


PATCH_MARKER = "DGTEST_DUAL_CANVAS_TRACE_V3"
MASK_CHAR = "□"


def fail(msg: str) -> None:
    raise SystemExit(f"[ERROR] {msg}")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        fail(f"bad config: {path}")
    for key in ("paths", "generation", "sweep", "visual"):
        if key not in cfg:
            fail(f"config missing required field: {key}")
    return cfg


def safe_name(x: Any) -> str:
    s = str(x if x is not None else "none").replace(".", "p")
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "none"


def format_alpha(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p").replace("-", "minus_")


def alpha_name(alpha: float | None) -> str:
    return "none" if alpha is None else f"alpha_{format_alpha(alpha)}"


def alpha_mode(alpha: float | None) -> str:
    return "none" if alpha is None else "scaled"


def parse_alpha_values(raw: Any) -> list[float | None]:
    if raw is None:
        fail("sweep.alphas is required")
    values = [x.strip() for x in raw.split(",") if x.strip()] if isinstance(raw, str) else raw
    if not isinstance(values, (list, tuple)):
        fail("sweep.alphas must be a list or comma-separated string")
    alphas: list[float | None] = []
    for value in values:
        if value is None or str(value).strip().lower() == "none":
            alphas.append(None)
            continue
        try:
            alpha = float(value)
        except Exception as exc:
            raise SystemExit(f"[ERROR] bad alpha value: {value!r}") from exc
        if alpha <= 0:
            fail(f"use 'none' to disable self-conditioning; scaled alpha must be > 0, got {alpha}")
        alphas.append(alpha)
    if not alphas:
        fail("no alpha values selected")
    return alphas


def selected_alphas(arg: str | None, cfg: dict[str, Any]) -> list[float | None]:
    return parse_alpha_values(arg if arg else cfg["sweep"].get("alphas"))


def experiment_for_alpha(alpha: float | None) -> dict[str, Any]:
    return {"name": alpha_name(alpha), "alpha": alpha, "mode": alpha_mode(alpha)}


def experiment_dir(output_root: Path, exp: dict[str, Any]) -> Path:
    return output_root / safe_name(exp["name"])


def replace_once(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        fail(
            f"patch point '{name}' expected once, found {count}. "
            "Usually this means transformers version mismatch or an older patch is present."
        )
    return text.replace(old, new, 1)


def generation_file_path(cfg: dict[str, Any], root: Path) -> Path:
    return root / cfg["paths"]["transformers_generation_file"]


def restore_patch(cfg: dict[str, Any], root: Path) -> None:
    target = generation_file_path(cfg, root)
    backup = target.with_suffix(target.suffix + ".orig")
    if not backup.exists():
        fail(f"backup not found: {backup}")
    target.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[PATCH] restored: {target}")


def patch_transformers(cfg: dict[str, Any], root: Path) -> None:
    target = generation_file_path(cfg, root)
    if not target.exists():
        fail(f"DiffusionGemma generation file not found: {target}")

    backup = target.with_suffix(target.suffix + ".orig")
    text = target.read_text(encoding="utf-8")

    if PATCH_MARKER in text:
        print(f"[PATCH] already patched: {target}")
        return

    if "DGTEST_SELFCOND" in text or "DGTEST_DUAL_CANVAS_TRACE" in text:
        if backup.exists():
            print("[PATCH] older DGTEST patch found; restoring .orig before applying current patch")
            text = backup.read_text(encoding="utf-8")
            target.write_text(text, encoding="utf-8")
        else:
            fail("older DGTEST patch found but .orig backup is missing. Re-checkout vendor/transformers.")

    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"[PATCH] backup written: {backup}")

    text = replace_once(
        text,
        "import copy\nimport math\nimport sys\n",
        "import copy\nimport math\nimport os\nimport sys\n",
        "add_os_import",
    )

    old = """        # 1.c.i Run the decoder, taking the current canvas, the encoder KV cache, and the self-conditioning
        # logits (if available) as inputs.
        decoder_outputs = decoder_forward(
            decoder_input_ids=current_canvas,
            self_conditioning_logits=self_conditioning_logits,
"""
    new = f"""        # {PATCH_MARKER}: self-conditioning alpha sweep.
        dg_selfcond_mode = getattr(self, "_dg_selfcond_mode", os.environ.get("DG_SELFCOND_MODE", "scaled"))
        dg_selfcond_mode = str(dg_selfcond_mode).lower()
        dg_selfcond_alpha = float(getattr(self, "_dg_selfcond_alpha", os.environ.get("DG_SELFCOND_ALPHA", "1.0")))
        if dg_selfcond_mode not in ("scaled", "none"):
            raise ValueError(f"Unknown DG self-conditioning mode: {{dg_selfcond_mode}}")
        decoder_self_conditioning_logits = None if dg_selfcond_mode == "none" else self_conditioning_logits

        # 1.c.i Run the decoder, taking the current canvas, the encoder KV cache, and the self-conditioning
        # logits (if available) as inputs.
        decoder_outputs = decoder_forward(
            decoder_input_ids=current_canvas,
            self_conditioning_logits=decoder_self_conditioning_logits,
"""
    text = replace_once(text, old, new, "decoder_forward_selfcond_input")

    old = """        new_current_canvas = sampler.renoise_canvas(accepted_canvas, cur_step)
        new_current_canvas = new_current_canvas.clone()  # clone needed for compiled sampler

        # 1.c.iv Update the diffusion stopping criteria.
"""
    new = f"""        new_current_canvas = sampler.renoise_canvas(accepted_canvas, cur_step)
        new_current_canvas = new_current_canvas.clone()  # clone needed for compiled sampler

        # {PATCH_MARKER}: record exact sampler states for both real and clean traces.
        if getattr(self, "_dg_trace_enabled", False):
            try:
                with torch.no_grad():
                    token_entropy = torch.distributions.Categorical(logits=processed_logits).entropy()
                    accepted_mask = sampler.accepted_token_mask
                    accepted_count = accepted_mask.sum(dim=-1)
                    renoise_mask = ~accepted_mask
                    trace_row = {{
                        "cur_step": int(cur_step.detach().cpu().item()) if torch.is_tensor(cur_step) else int(cur_step),
                        "mode": dg_selfcond_mode,
                        "alpha": float(dg_selfcond_alpha),
                        "accepted_count": accepted_count.detach().cpu().tolist(),
                        "mean_entropy": token_entropy.mean(dim=-1).detach().cpu().tolist(),
                    }}
                    if getattr(self, "_dg_trace_details", True):
                        trace_row.update({{
                            "accepted_positions": [
                                torch.nonzero(row, as_tuple=False).flatten().detach().cpu().tolist()
                                for row in accepted_mask
                            ],
                            "renoise_positions": [
                                torch.nonzero(row, as_tuple=False).flatten().detach().cpu().tolist()
                                for row in renoise_mask
                            ],
                            "position_entropy": token_entropy.detach().cpu().tolist(),
                            "input_canvas_token_ids": current_canvas.detach().cpu().tolist(),
                            "sampled_canvas_token_ids": denoiser_canvas.detach().cpu().tolist(),
                            "accepted_canvas_token_ids": accepted_canvas.detach().cpu().tolist(),
                            "output_canvas_token_ids": new_current_canvas.detach().cpu().tolist(),
                            "argmax_token_ids": new_argmax_canvas.detach().cpu().tolist(),
                        }})
                    if not hasattr(self, "_dg_trace") or self._dg_trace is None:
                        self._dg_trace = []
                    self._dg_trace.append(trace_row)
            except Exception as exc:
                self._dg_trace_error = repr(exc)

        # 1.c.iv Update the diffusion stopping criteria.
"""
    text = replace_once(text, old, new, "trace_after_renoise")

    old = """        processed_logits = torch.where(
            finished_denoising[:, None, None], self_conditioning_logits, processed_logits
        )
"""
    new = """        if self_conditioning_logits is not None:
            processed_logits = torch.where(
                finished_denoising[:, None, None],
                self_conditioning_logits,
                processed_logits,
            )
"""
    text = replace_once(text, old, new, "finished_branch_none_safe")

    old = """        embeddings_dtype = self.model.decoder.embed_tokens.weight.dtype
        self_conditioning_logits = processed_logits.to(embeddings_dtype)

        return (
"""
    new = f"""        embeddings_dtype = self.model.decoder.embed_tokens.weight.dtype
        # {PATCH_MARKER}: choose next-step self-conditioning.
        if dg_selfcond_mode == "scaled":
            self_conditioning_logits = (processed_logits * dg_selfcond_alpha).to(embeddings_dtype)
        elif dg_selfcond_mode == "none":
            self_conditioning_logits = None
        else:
            raise ValueError(f"Unknown DG self-conditioning mode: {{dg_selfcond_mode}}")

        return (
"""
    text = replace_once(text, old, new, "next_selfcond_update")

    target.write_text(text, encoding="utf-8")
    print(f"[PATCH] patched: {target}")


def read_samples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict) or "id" not in obj or "prompt" not in obj:
                fail(f"{path}:{line_no} must contain id and prompt")
            rows.append(obj)
    if not rows:
        fail(f"no samples found: {path}")
    return rows


def to_py(x: Any) -> Any:
    return x.detach().cpu().tolist() if torch.is_tensor(x) else x


def encode_chat(processor, prompt: str):
    return processor.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def decode_piece(processor, token_id: int) -> str:
    try:
        return processor.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return str(token_id)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def params_csv_row(row: dict[str, Any], gen_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id"),
        "type": row.get("type"),
        "prompt": row.get("prompt"),
        "experiment_name": row.get("experiment_name"),
        "self_conditioning_alpha": row.get("alpha"),
        "mode": row.get("mode"),
        "seed": row.get("seed"),
        "max_new_tokens": gen_cfg.get("max_new_tokens"),
        "max_denoising_steps": gen_cfg.get("max_denoising_steps"),
        "dtype": gen_cfg.get("dtype"),
        "device_map": gen_cfg.get("device_map"),
        "local_files_only": gen_cfg.get("local_files_only"),
        "latency_sec": row.get("latency_sec"),
        "tokens_per_forward": json.dumps(row.get("tokens_per_forward"), ensure_ascii=False),
        "trace_error": row.get("trace_error"),
        "generated_text": row.get("generated_text"),
        "full_text": row.get("full_text"),
    }


def trace_csv_rows(
    trace: list[dict[str, Any]],
    processor,
    sample_id: str,
    experiment_name: str,
    alpha: float | None,
    latency_sec: float,
    generated_text: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    canvas_index = 0
    previous_step: int | None = None

    for trace_index, item in enumerate(trace):
        step = int(item.get("cur_step", 0))
        if previous_step is not None and step <= previous_step:
            canvas_index += 1
        previous_step = step

        positions = (item.get("accepted_positions") or [[]])[0]
        input_canvas = (item.get("input_canvas_token_ids") or [[]])[0]
        sampled_canvas = (item.get("sampled_canvas_token_ids") or [[]])[0]
        accepted_canvas = (item.get("accepted_canvas_token_ids") or [[]])[0]
        output_canvas = (item.get("output_canvas_token_ids") or [[]])[0]
        argmax_canvas = (item.get("argmax_token_ids") or [[]])[0]
        entropies = (item.get("position_entropy") or [[]])[0]
        renoise_positions = (item.get("renoise_positions") or [[]])[0]

        changed_tokens = [
            decode_piece(processor, accepted_canvas[p]) if 0 <= p < len(accepted_canvas) else ""
            for p in positions
        ]
        changed_entropies = [
            entropies[p] if 0 <= p < len(entropies) else None for p in positions
        ]

        def decode_tokens(ids: list[Any]) -> list[str]:
            return [decode_piece(processor, int(token_id)) for token_id in ids]

        rows.append({
            "sample_id": sample_id,
            "experiment_name": experiment_name,
            "self_conditioning_alpha": alpha,
            "mode": item.get("mode"),
            "trace_index": trace_index,
            "canvas_index": canvas_index,
            "generation_step": step,
            "accepted_count": (item.get("accepted_count") or [None])[0],
            "mean_entropy": (item.get("mean_entropy") or [None])[0],
            "accepted_positions": json.dumps(positions, ensure_ascii=False),
            "accepted_tokens": json.dumps(changed_tokens, ensure_ascii=False),
            "input_canvas_token_ids": json.dumps(input_canvas, ensure_ascii=False),
            "input_canvas_tokens": json.dumps(decode_tokens(input_canvas), ensure_ascii=False),
            "sampled_canvas_token_ids": json.dumps(sampled_canvas, ensure_ascii=False),
            "sampled_canvas_tokens": json.dumps(decode_tokens(sampled_canvas), ensure_ascii=False),
            "accepted_canvas_token_ids": json.dumps(accepted_canvas, ensure_ascii=False),
            "accepted_canvas_tokens": json.dumps(decode_tokens(accepted_canvas), ensure_ascii=False),
            "output_canvas_token_ids": json.dumps(output_canvas, ensure_ascii=False),
            "output_canvas_tokens": json.dumps(decode_tokens(output_canvas), ensure_ascii=False),
            "argmax_token_ids": json.dumps(argmax_canvas, ensure_ascii=False),
            "argmax_tokens": json.dumps(decode_tokens(argmax_canvas), ensure_ascii=False),
            "position_entropy": json.dumps(entropies, ensure_ascii=False),
            "accepted_position_entropy": json.dumps(changed_entropies, ensure_ascii=False),
            "renoise_positions": json.dumps(renoise_positions, ensure_ascii=False),
            "latency_sec": latency_sec,
            "final_output": generated_text,
        })
    return rows


def clean_trace_csv_rows(true_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_rows: list[dict[str, Any]] = []
    previous_by_canvas: dict[int, set[int]] = {}
    initialized: set[int] = set()

    for row in true_rows:
        canvas_index = int(row["canvas_index"])
        input_ids = json.loads(row["input_canvas_token_ids"])
        canvas_len = len(input_ids) or 256

        if canvas_index not in initialized:
            clean_rows.append({
                "sample_id": row["sample_id"],
                "experiment_name": row["experiment_name"],
                "self_conditioning_alpha": row["self_conditioning_alpha"],
                "mode": row["mode"],
                "trace_index": f"{row['trace_index']}.initial",
                "canvas_index": canvas_index,
                "generation_step": -1,
                "phase": "initial_all_mask",
                "accepted_count": 0,
                "accepted_positions": "[]",
                "newly_committed_positions": "[]",
                "unaccepted_positions": "[]",
                "clean_canvas_tokens": json.dumps([MASK_CHAR] * canvas_len, ensure_ascii=False),
                "clean_canvas_text": MASK_CHAR * canvas_len,
                "mean_entropy": None,
            })
            initialized.add(canvas_index)
            previous_by_canvas[canvas_index] = set()

        positions = [int(x) for x in json.loads(row["accepted_positions"])]
        tokens = [str(x) for x in json.loads(row["accepted_tokens"])]
        visible = [MASK_CHAR] * canvas_len
        for pos, token in zip(positions, tokens):
            if 0 <= pos < canvas_len:
                visible[pos] = token

        current = {p for p in positions if 0 <= p < canvas_len}
        previous = previous_by_canvas[canvas_index]
        clean_rows.append({
            "sample_id": row["sample_id"],
            "experiment_name": row["experiment_name"],
            "self_conditioning_alpha": row["self_conditioning_alpha"],
            "mode": row["mode"],
            "trace_index": row["trace_index"],
            "canvas_index": canvas_index,
            "generation_step": row["generation_step"],
            "phase": "accepted_state",
            "accepted_count": row["accepted_count"],
            "accepted_positions": json.dumps(sorted(current), ensure_ascii=False),
            "newly_committed_positions": json.dumps(sorted(current - previous), ensure_ascii=False),
            "unaccepted_positions": json.dumps(sorted(previous - current), ensure_ascii=False),
            "clean_canvas_tokens": json.dumps(visible, ensure_ascii=False),
            "clean_canvas_text": "".join(visible),
            "mean_entropy": row["mean_entropy"],
        })
        previous_by_canvas[canvas_index] = current

    return clean_rows


def clear_experiment_outputs(output_root: Path, experiments: list[dict[str, Any]]) -> None:
    for exp in experiments:
        exp_dir = experiment_dir(output_root, exp)
        for filename in ("params.csv", "trace.csv", "clean_trace.csv"):
            path = exp_dir / filename
            if path.exists():
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DiffusionGemma self-conditioning alpha sweep")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--alphas", default=None, help="comma-separated values; example: 1,0.75,0.5,0.25,none")
    parser.add_argument("--samples", default=None, help="override sample_file in config")
    parser.add_argument("--restore-patch", action="store_true")
    parser.add_argument("--patch-only", action="store_true")
    args = parser.parse_args()

    root = Path(".").resolve()
    cfg = load_config(args.config)

    if args.restore_patch:
        restore_patch(cfg, root)
        return

    patch_transformers(cfg, root)
    if args.patch_only:
        return

    gen_cfg = cfg["generation"]
    output_root = root / cfg["paths"]["output_root"]
    result_file = root / cfg["paths"]["result_file"]
    output_root.mkdir(parents=True, exist_ok=True)
    result_file.parent.mkdir(parents=True, exist_ok=True)

    sample_file = root / (args.samples or cfg["paths"]["sample_file"])
    samples = read_samples(sample_file)
    experiments = [experiment_for_alpha(alpha) for alpha in selected_alphas(args.alphas, cfg)]
    clear_experiment_outputs(output_root, experiments)

    seed = int(gen_cfg["seed"])
    print(f"[LOAD] {cfg['model_id']}")
    processor = AutoProcessor.from_pretrained(
        cfg["model_id"],
        local_files_only=bool(gen_cfg.get("local_files_only", False)),
    )
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        cfg["model_id"],
        dtype=gen_cfg.get("dtype", "auto"),
        device_map=gen_cfg.get("device_map", "auto"),
        local_files_only=bool(gen_cfg.get("local_files_only", False)),
    )
    model.eval()
    model._dg_trace_enabled = True
    model._dg_trace_details = bool(gen_cfg.get("trace_details", True))

    params_fields = [
        "sample_id", "type", "prompt", "experiment_name", "self_conditioning_alpha",
        "mode", "seed", "max_new_tokens", "max_denoising_steps", "dtype",
        "device_map", "local_files_only", "latency_sec", "tokens_per_forward",
        "trace_error", "generated_text", "full_text",
    ]
    trace_fields = [
        "sample_id", "experiment_name", "self_conditioning_alpha", "mode",
        "trace_index", "canvas_index", "generation_step", "accepted_count",
        "mean_entropy", "accepted_positions", "accepted_tokens",
        "input_canvas_token_ids", "input_canvas_tokens",
        "sampled_canvas_token_ids", "sampled_canvas_tokens",
        "accepted_canvas_token_ids", "accepted_canvas_tokens",
        "output_canvas_token_ids", "output_canvas_tokens",
        "argmax_token_ids", "argmax_tokens", "position_entropy",
        "accepted_position_entropy", "renoise_positions", "latency_sec", "final_output",
    ]
    clean_trace_fields = [
        "sample_id", "experiment_name", "self_conditioning_alpha", "mode",
        "trace_index", "canvas_index", "generation_step", "phase", "accepted_count",
        "accepted_positions", "newly_committed_positions", "unaccepted_positions",
        "clean_canvas_tokens", "clean_canvas_text", "mean_entropy",
    ]

    per_experiment_params = {exp["name"]: [] for exp in experiments}
    per_experiment_trace = {exp["name"]: [] for exp in experiments}
    per_experiment_clean_trace = {exp["name"]: [] for exp in experiments}

    with result_file.open("w", encoding="utf-8") as fout:
        for sample in samples:
            for exp in experiments:
                alpha = exp["alpha"]
                mode = exp["mode"]
                model._dg_selfcond_mode = mode
                model._dg_selfcond_alpha = 1.0 if alpha is None else float(alpha)
                model._dg_trace = []
                if hasattr(model, "_dg_trace_error"):
                    delattr(model, "_dg_trace_error")

                encoded = encode_chat(processor, str(sample["prompt"]))
                encoded = encoded.to(model.device)
                prompt_len = encoded["input_ids"].shape[-1]

                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                    torch.cuda.synchronize()

                print(f"[RUN] sample={sample['id']} {exp['name']} mode={mode}")
                start = time.time()
                with torch.inference_mode():
                    output = model.generate(
                        **encoded,
                        max_new_tokens=int(gen_cfg["max_new_tokens"]),
                        max_denoising_steps=int(gen_cfg["max_denoising_steps"]),
                        disable_compile=True,
                        return_dict_in_generate=True,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latency = time.time() - start

                seq = output.sequences
                generated_text = processor.decode(seq[0, prompt_len:], skip_special_tokens=False)
                row = {
                    "sample_id": str(sample["id"]),
                    "type": str(sample.get("type", "")),
                    "prompt": str(sample["prompt"]),
                    "experiment_name": exp["name"],
                    "mode": mode,
                    "alpha": alpha,
                    "seed": seed,
                    "latency_sec": latency,
                    "tokens_per_forward": to_py(getattr(output, "tokens_per_forward", None)),
                    "full_text": processor.decode(seq[0], skip_special_tokens=False),
                    "generated_text": generated_text,
                    "trace_error": getattr(model, "_dg_trace_error", None),
                    "trace": model._dg_trace,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()

                true_rows = trace_csv_rows(
                    model._dg_trace,
                    processor=processor,
                    sample_id=str(sample["id"]),
                    experiment_name=exp["name"],
                    alpha=alpha,
                    latency_sec=latency,
                    generated_text=generated_text,
                )
                per_experiment_params[exp["name"]].append(params_csv_row(row, gen_cfg))
                per_experiment_trace[exp["name"]].extend(true_rows)
                per_experiment_clean_trace[exp["name"]].extend(clean_trace_csv_rows(true_rows))

                sample_out = experiment_dir(output_root, exp) / f"{safe_name(sample['id'])}_final_output.txt"
                sample_out.parent.mkdir(parents=True, exist_ok=True)
                sample_out.write_text(generated_text, encoding="utf-8")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    for exp in experiments:
        exp_dir = experiment_dir(output_root, exp)
        write_csv(exp_dir / "params.csv", params_fields, per_experiment_params[exp["name"]])
        write_csv(exp_dir / "trace.csv", trace_fields, per_experiment_trace[exp["name"]])
        write_csv(exp_dir / "clean_trace.csv", clean_trace_fields, per_experiment_clean_trace[exp["name"]])

    print(f"[DONE] aggregate jsonl: {result_file}")
    print(f"[DONE] per-alpha CSVs: {output_root}/alpha_*/params.csv, trace.csv and clean_trace.csv")
    print("[NEXT] python visual.py --config config.yaml --mode dynamic")


if __name__ == "__main__":
    main()
