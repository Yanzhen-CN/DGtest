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

# torch and transformers are imported inside main(), after patch_transformers()
# has updated the vendored source. Importing them here would cache the old,
# unpatched generation module in the current Python process.

try:
    import yaml
except ImportError as exc:
    raise SystemExit("[ERROR] missing dependency: pyyaml") from exc


PATCH_MARKER = "DGTEST_SELFCOND_STEP_SWEEP_V4"

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


def configure_huggingface_environment(cfg: dict[str, Any], root: Path) -> None:
    """Keep the standard Hub cache layout, rooted on persistent storage."""
    gen_cfg = cfg["generation"]
    if bool(gen_cfg.get("disable_xet", True)):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    else:
        os.environ.pop("HF_HUB_DISABLE_XET", None)

    raw_hf_home = gen_cfg.get("hf_home")
    if not raw_hf_home:
        return
    raw_cache_text = str(raw_hf_home)
    # Keep the RunPod config usable for local Windows syntax/import checks.
    if os.name == "nt" and raw_cache_text.startswith("/workspace/"):
        hf_home = root / ".cache" / "huggingface"
    else:
        hf_home = Path(raw_cache_text).expanduser()
    if not hf_home.is_absolute():
        hf_home = root / hf_home
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    # Remove explicit sub-cache overrides so huggingface_hub uses its normal
    # $HF_HOME/hub and $HF_HOME/xet defaults, matching the original setup.
    os.environ.pop("HF_HUB_CACHE", None)
    os.environ.pop("HF_XET_CACHE", None)


def safe_name(x: Any) -> str:
    s = str(x if x is not None else "none").replace(".", "p")
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "none"


def format_alpha(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p").replace("-", "minus_")


def alpha_name(alpha: float | None) -> str:
    return "alphanone" if alpha is None else f"alpha{format_alpha(alpha)}"


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


def parse_step_values(raw: Any) -> list[int]:
    if raw is None:
        fail("generation.denoising_steps is required")
    values = [x.strip() for x in raw.split(",") if x.strip()] if isinstance(raw, str) else raw
    if not isinstance(values, (list, tuple)):
        fail("generation.denoising_steps must be a list or comma-separated string")
    try:
        steps = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"[ERROR] bad denoising step value in: {values!r}") from exc
    if not steps or any(step <= 0 for step in steps):
        fail("all denoising step budgets must be positive integers")
    return list(dict.fromkeys(steps))


def experiment_for_alpha(alpha: float | None) -> dict[str, Any]:
    return {"name": alpha_name(alpha), "alpha": alpha, "mode": alpha_mode(alpha)}


def selected_run_specs(
    alpha_arg: str | None,
    steps_arg: str | None,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resolve exact (alpha, step-budget) pairs.

    CLI overrides retain the historical Cartesian-product behavior. Without
    overrides, ``sweep.experiments`` can express asymmetric budgets such as
    alpha1/alpha0.5 at 48 steps and none at 96 steps.
    """
    configured = cfg.get("sweep", {}).get("experiments")
    if alpha_arg is None and steps_arg is None and configured is not None:
        if not isinstance(configured, list) or not configured:
            fail("sweep.experiments must be a non-empty list")
        specs: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for item in configured:
            if not isinstance(item, dict) or "alpha" not in item or "steps" not in item:
                fail("each sweep.experiments item requires alpha and steps")
            alpha = parse_alpha_values([item["alpha"]])[0]
            steps = int(item["steps"])
            if steps <= 0:
                fail("experiment steps must be a positive integer")
            experiment = experiment_for_alpha(alpha)
            key = (experiment["name"], steps)
            if key not in seen:
                specs.append({"steps": steps, "experiment": experiment})
                seen.add(key)
        return specs

    alphas = selected_alphas(alpha_arg, cfg)
    steps = parse_step_values(
        steps_arg if steps_arg else cfg["generation"].get("denoising_steps")
    )
    return [
        {"steps": budget, "experiment": experiment_for_alpha(alpha)}
        for budget in steps
        for alpha in alphas
    ]


def budget_dir(output_root: Path, canvas_length: int, steps: int) -> Path:
    return output_root / f"len{canvas_length}" / f"step{steps}"


def experiment_dir(output_root: Path, canvas_length: int, steps: int, exp: dict[str, Any]) -> Path:
    return budget_dir(output_root, canvas_length, steps) / safe_name(exp["name"])


def sample_dir(output_root: Path, canvas_length: int, steps: int, exp: dict[str, Any], sample_id: Any) -> Path:
    return experiment_dir(output_root, canvas_length, steps, exp) / safe_name(sample_id)


def replace_once(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        fail(
            f"patch point '{name}' expected once, found {count}. "
            "Usually this means transformers version mismatch or an older patch is present."
        )
    return text.replace(old, new, 1)


def replace_one_of(text: str, variants: list[tuple[str, str]], name: str) -> str:
    """Replace exactly one of the supported upstream source variants."""
    matches = [(old, new) for old, new in variants if text.count(old) == 1]
    if len(matches) != 1:
        counts = [text.count(old) for old, _ in variants]
        fail(
            f"patch point '{name}' expected one supported variant, found counts={counts}. "
            "Usually this means transformers version mismatch or an older patch is present."
        )
    old, new = matches[0]
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

    old = """        self.t_min = t_min
        self.t_max = t_max
        self.max_denoising_steps = max_denoising_steps
"""
    new = f"""        self.t_min = t_min
        self.t_max = t_max
        self.max_denoising_steps = max_denoising_steps
        # {PATCH_MARKER}: decouple temperature decay from the total step budget.
        self.temperature_schedule_steps = int(
            os.environ.get("DG_TEMPERATURE_SCHEDULE_STEPS", max_denoising_steps)
        )
        if self.temperature_schedule_steps <= 0:
            raise ValueError("DG_TEMPERATURE_SCHEDULE_STEPS must be positive")
"""
    text = replace_once(text, old, new, "temperature_schedule_init")

    old = """        temperature = self.t_min + ((self.t_max - self.t_min) * (cur_step / self.max_denoising_steps))
        return scores / temperature
"""
    new = f"""        # {PATCH_MARKER}: the first schedule_steps chronological iterations follow
        # the baseline schedule; any extra budget stays at the minimum temperature.
        elapsed_steps = self.max_denoising_steps - cur_step
        schedule_step = torch.clamp(
            torch.as_tensor(self.temperature_schedule_steps, device=scores.device) - elapsed_steps,
            min=0,
        )
        temperature = self.t_min + (
            (self.t_max - self.t_min) * (schedule_step / self.temperature_schedule_steps)
        )
        return scores / temperature
"""
    text = replace_once(text, old, new, "temperature_schedule_value")

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
                        "entropy_stage": "pre_accept_processed_logits",
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

    text = replace_one_of(
        text,
        [
            # Source layout used by commit ed01d309 and the vendored tree.
            (
                """                processed_logits = torch.where(
                    finished_denoising[:, None, None], self_conditioning_logits, processed_logits
                )
""",
                """                if self_conditioning_logits is not None:
                    processed_logits = torch.where(
                        finished_denoising[:, None, None],
                        self_conditioning_logits,
                        processed_logits,
                    )
""",
            ),
            # Compatibility with the earlier unguarded upstream layout.
            (
                """        processed_logits = torch.where(
            finished_denoising[:, None, None], self_conditioning_logits, processed_logits
        )
""",
                """        if self_conditioning_logits is not None:
            processed_logits = torch.where(
                finished_denoising[:, None, None],
                self_conditioning_logits,
                processed_logits,
            )
""",
            ),
        ],
        "finished_branch_none_safe",
    )

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
    """Serialize the full sampler state and accepted-token events.

    accepted_positions / accepted_tokens:
        token positions and values actually accepted by the sampler.

    changed_positions:
        compatibility alias of accepted_positions. In this project, a token
        "change" means an accepted/committed event, never a re-noise event.

    output_changed_positions:
        diagnostic-only raw canvas value changes between input and output.
        This field includes re-noise and must not be used as commit/revision.
    """
    rows: list[dict[str, Any]] = []
    canvas_index = 0
    previous_step: int | None = None
    acceptance_counts: dict[tuple[int, int], int] = {}

    for trace_index, item in enumerate(trace):
        step = int(item.get("cur_step", 0))
        if previous_step is not None and step > previous_step:
            canvas_index += 1
        previous_step = step

        accepted_positions = [
            int(x) for x in (item.get("accepted_positions") or [[]])[0]
        ]
        input_canvas = list((item.get("input_canvas_token_ids") or [[]])[0])
        sampled_canvas = list((item.get("sampled_canvas_token_ids") or [[]])[0])
        accepted_canvas = list((item.get("accepted_canvas_token_ids") or [[]])[0])
        output_canvas = list((item.get("output_canvas_token_ids") or [[]])[0])
        argmax_canvas = list((item.get("argmax_token_ids") or [[]])[0])
        entropies = list((item.get("position_entropy") or [[]])[0])
        renoise_positions = list((item.get("renoise_positions") or [[]])[0])

        def decode_tokens(ids: list[Any]) -> list[str]:
            return [decode_piece(processor, int(token_id)) for token_id in ids]

        accepted_tokens = [
            decode_piece(processor, accepted_canvas[position])
            if 0 <= position < len(accepted_canvas)
            else ""
            for position in accepted_positions
        ]
        accepted_entropies = [
            entropies[position] if 0 <= position < len(entropies) else None
            for position in accepted_positions
        ]

        update_ranks: list[int] = []
        for position in accepted_positions:
            key = (canvas_index, position)
            rank = acceptance_counts.get(key, 0) + 1
            acceptance_counts[key] = rank
            update_ranks.append(rank)

        output_changed_positions = [
            position
            for position in range(min(len(input_canvas), len(output_canvas)))
            if int(input_canvas[position]) != int(output_canvas[position])
        ]

        rows.append({
            "sample_id": sample_id,
            "experiment_name": experiment_name,
            "self_conditioning_alpha": alpha,
            "mode": item.get("mode"),
            "entropy_stage": item.get("entropy_stage", "pre_accept_processed_logits"),
            "trace_index": trace_index,
            "canvas_index": canvas_index,
            "generation_step": step,
            "accepted_count": len(accepted_positions),
            "changed_count": len(accepted_positions),
            "mean_entropy": (item.get("mean_entropy") or [None])[0],
            "accepted_positions": json.dumps(accepted_positions, ensure_ascii=False),
            "changed_positions": json.dumps(accepted_positions, ensure_ascii=False),
            "update_ranks": json.dumps(update_ranks, ensure_ascii=False),
            "accepted_tokens": json.dumps(accepted_tokens, ensure_ascii=False),
            "output_changed_count": len(output_changed_positions),
            "output_changed_positions": json.dumps(
                output_changed_positions, ensure_ascii=False
            ),
            "input_canvas_token_ids": json.dumps(input_canvas, ensure_ascii=False),
            "input_canvas_tokens": json.dumps(
                decode_tokens(input_canvas), ensure_ascii=False
            ),
            "sampled_canvas_token_ids": json.dumps(sampled_canvas, ensure_ascii=False),
            "sampled_canvas_tokens": json.dumps(
                decode_tokens(sampled_canvas), ensure_ascii=False
            ),
            "accepted_canvas_token_ids": json.dumps(
                accepted_canvas, ensure_ascii=False
            ),
            "accepted_canvas_tokens": json.dumps(
                decode_tokens(accepted_canvas), ensure_ascii=False
            ),
            "output_canvas_token_ids": json.dumps(output_canvas, ensure_ascii=False),
            "output_canvas_tokens": json.dumps(
                decode_tokens(output_canvas), ensure_ascii=False
            ),
            "argmax_token_ids": json.dumps(argmax_canvas, ensure_ascii=False),
            "argmax_tokens": json.dumps(
                decode_tokens(argmax_canvas), ensure_ascii=False
            ),
            "position_entropy": json.dumps(entropies, ensure_ascii=False),
            "accepted_position_entropy": json.dumps(
                accepted_entropies, ensure_ascii=False
            ),
            "renoise_positions": json.dumps(renoise_positions, ensure_ascii=False),
            "latency_sec": latency_sec,
            "final_output": generated_text,
        })

    return rows


def clear_experiment_outputs(
    output_root: Path,
    canvas_length: int,
    run_specs: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> None:
    for steps in sorted({int(spec["steps"]) for spec in run_specs}):
        results_path = budget_dir(output_root, canvas_length, steps) / "results.jsonl"
        if results_path.exists():
            results_path.unlink()
    for spec in run_specs:
        steps = int(spec["steps"])
        exp = spec["experiment"]
        for sample in samples:
            leaf = sample_dir(output_root, canvas_length, steps, exp, sample["id"])
            for filename in ("params.csv", "trace.csv", "final_output.txt"):
                path = leaf / filename
                if path.exists():
                    path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DiffusionGemma self-conditioning alpha sweep")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--alphas", default=None, help="comma-separated values; example: 1,0.5,none")
    parser.add_argument("--steps", default=None, help="comma-separated denoising budgets; example: 48,96")
    parser.add_argument("--samples", default=None, help="override sample_file in config")
    parser.add_argument("--restore-patch", action="store_true")
    parser.add_argument("--patch-only", action="store_true")
    args = parser.parse_args()

    root = Path(".").resolve()
    cfg = load_config(args.config)
    configure_huggingface_environment(cfg, root)

    if args.restore_patch:
        restore_patch(cfg, root)
        return

    patch_transformers(cfg, root)
    if args.patch_only:
        return

    # Import only after the on-disk Transformers patch is complete.
    # This guarantees that model.generate uses the patched implementation
    # in this same process.
    global torch, AutoProcessor, DiffusionGemmaForBlockDiffusion
    import torch
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

    gen_cfg = cfg["generation"]
    output_root = root / cfg["paths"]["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    sample_file = root / (args.samples or cfg["paths"]["sample_file"])
    samples = read_samples(sample_file)
    run_specs = selected_run_specs(args.alphas, args.steps, cfg)
    step_budgets = list(dict.fromkeys(int(spec["steps"]) for spec in run_specs))
    canvas_length = int(gen_cfg["max_new_tokens"])
    schedule_steps = int(gen_cfg.get("temperature_schedule_steps", min(step_budgets)))
    if schedule_steps <= 0:
        fail("generation.temperature_schedule_steps must be positive")
    os.environ["DG_TEMPERATURE_SCHEDULE_STEPS"] = str(schedule_steps)
    clear_experiment_outputs(output_root, canvas_length, run_specs, samples)

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
        "sample_id", "experiment_name", "self_conditioning_alpha", "mode", "entropy_stage",
        "trace_index", "canvas_index", "generation_step", "accepted_count",
        "changed_count", "mean_entropy", "accepted_positions",
        "changed_positions", "update_ranks", "accepted_tokens",
        "output_changed_count", "output_changed_positions",
        "input_canvas_token_ids", "input_canvas_tokens",
        "sampled_canvas_token_ids", "sampled_canvas_tokens",
        "accepted_canvas_token_ids", "accepted_canvas_tokens",
        "output_canvas_token_ids", "output_canvas_tokens",
        "argmax_token_ids", "argmax_tokens", "position_entropy",
        "accepted_position_entropy", "renoise_positions", "latency_sec", "final_output",
    ]

    for steps in step_budgets:
        budget_specs = [spec for spec in run_specs if int(spec["steps"]) == steps]
        result_file = budget_dir(output_root, canvas_length, steps) / "results.jsonl"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        run_gen_cfg = dict(gen_cfg, max_denoising_steps=steps)
        with result_file.open("w", encoding="utf-8") as fout:
            for sample, spec in (
                (sample, spec) for sample in samples for spec in budget_specs
            ):
                exp = spec["experiment"]
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

                print(f"[RUN] len={canvas_length} steps={steps} sample={sample['id']} {exp['name']} mode={mode}")
                start = time.time()
                with torch.inference_mode():
                    output = model.generate(
                        **encoded,
                        max_new_tokens=canvas_length,
                        max_denoising_steps=steps,
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
                leaf = sample_dir(output_root, canvas_length, steps, exp, sample["id"])
                leaf.mkdir(parents=True, exist_ok=True)
                write_csv(leaf / "params.csv", params_fields, [params_csv_row(row, run_gen_cfg)])
                write_csv(leaf / "trace.csv", trace_fields, true_rows)
                (leaf / "final_output.txt").write_text(generated_text, encoding="utf-8")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"[DONE] aggregate jsonl: {result_file}")
    print(f"[DONE] experiment leaves: {output_root}/len{canvas_length}/step*/alpha*/*/")
    print("[NEXT] python visual.py --config config.yaml --mode all")


if __name__ == "__main__":
    main()
