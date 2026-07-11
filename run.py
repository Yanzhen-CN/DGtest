from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the experiment imports the same local Transformers source that it patches.
REPO_ROOT = Path(__file__).resolve().parent
VENDORED_TRANSFORMERS_SRC = REPO_ROOT / "vendor" / "transformers" / "src"
sys.path.insert(0, str(VENDORED_TRANSFORMERS_SRC))

import torch
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

try:
    import yaml
except ImportError as exc:
    raise SystemExit("[ERROR] missing dependency: pyyaml. Install with: pip install pyyaml") from exc


PATCH_MARKER = "DGTEST_SELFCOND_CONFIG_PATCH"


def fail(msg: str) -> None:
    raise SystemExit(f"[ERROR] {msg}")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def replace_once(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        fail(
            f"patch point '{name}' expected once, found {count}. "
            "Usually this means transformers version mismatch or an old patch is present."
        )
    return text.replace(old, new, 1)


def generation_file_path(cfg: dict[str, Any], root: Path) -> Path:
    rel = cfg["paths"]["transformers_generation_file"]
    return root / rel


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

    # If any older DGTEST patch exists, restore original first.
    if "DGTEST_SELFCOND" in text:
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
    new = f"""        # {PATCH_MARKER}: self-conditioning strength sweep.
        # mode:
        #   scaled -> next step receives alpha * processed_logits as self-conditioning
        #   none   -> decoder receives no self-conditioning signal
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

        # {PATCH_MARKER}: minimum trace for output-chain and metric plots.
        if getattr(self, "_dg_trace_enabled", False):
            try:
                with torch.no_grad():
                    token_entropy = torch.distributions.Categorical(logits=processed_logits).entropy()
                    accepted_mask = sampler.accepted_token_mask
                    accepted_count = accepted_mask.sum(dim=-1)

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
                            "argmax_token_ids": new_argmax_canvas.detach().cpu().tolist(),
                            "accepted_canvas_token_ids": accepted_canvas.detach().cpu().tolist(),
                        }})

                    if not hasattr(self, "_dg_trace") or self._dg_trace is None:
                        self._dg_trace = []
                    self._dg_trace.append(trace_row)
            except Exception as exc:
                self._dg_trace_error = repr(exc)

        # 1.c.iv Update the diffusion stopping criteria.
"""
    text = replace_once(text, old, new, "trace_after_renoise")

    old = """                processed_logits = torch.where(
                    finished_denoising[:, None, None], self_conditioning_logits, processed_logits
                )
"""
    new = """                if self_conditioning_logits is not None:
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


def read_samples(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append({"id": str(obj["id"]), "prompt": str(obj["prompt"])})
    return rows


def parse_experiment_ids(arg: str | None, cfg: dict[str, Any]) -> list[str]:
    all_ids = list(cfg["experiments"].keys())
    all_ids = sorted(all_ids, key=lambda x: int(x) if str(x).isdigit() else str(x))
    if not arg or arg.lower() == "all":
        return all_ids
    ids = [x.strip() for x in arg.split(",") if x.strip()]
    missing = [x for x in ids if x not in cfg["experiments"]]
    if missing:
        fail(f"unknown experiment id(s): {missing}. Available: {all_ids}")
    return ids


def to_py(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def encode_chat(processor, prompt: str):
    return processor.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DiffusionGemma self-conditioning experiment")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--experiments", default=None, help="default=all; example: 1,3,5")
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
    out_path = root / cfg["paths"]["output_file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample_file = root / (args.samples or cfg["paths"]["sample_file"])
    samples = read_samples(sample_file)

    exp_ids = parse_experiment_ids(args.experiments, cfg)
    exps = [(eid, cfg["experiments"][eid]) for eid in exp_ids]

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

    with out_path.open("w", encoding="utf-8") as fout:
        for sample in samples:
            for exp_id, exp in exps:
                mode = exp["mode"]
                alpha = exp.get("alpha", None)

                model._dg_selfcond_mode = mode
                model._dg_selfcond_alpha = 1.0 if alpha is None else float(alpha)
                model._dg_trace = []
                if hasattr(model, "_dg_trace_error"):
                    delattr(model, "_dg_trace_error")

                encoded = encode_chat(processor, sample["prompt"])
                encoded = encoded.to(model.device)
                prompt_len = encoded["input_ids"].shape[-1]

                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                    torch.cuda.synchronize()

                print(f"[RUN] sample={sample['id']} exp={exp_id}:{exp['name']}")
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
                row = {
                    "sample_id": sample["id"],
                    "prompt": sample["prompt"],
                    "experiment_id": exp_id,
                    "experiment_name": exp["name"],
                    "mode": mode,
                    "alpha": alpha,
                    "seed": seed,
                    "latency_sec": latency,
                    "tokens_per_forward": to_py(getattr(output, "tokens_per_forward", None)),
                    "full_text": processor.decode(seq[0], skip_special_tokens=False),
                    "generated_text": processor.decode(seq[0, prompt_len:], skip_special_tokens=False),
                    "trace_error": getattr(model, "_dg_trace_error", None),
                    "trace": model._dg_trace,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"[DONE] wrote {out_path}")


if __name__ == "__main__":
    main()
