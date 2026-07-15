# Direct self-conditioning strength sweep

The historical `--alphas` intervention scales the previous-step logits before
their softmax. It therefore changes distribution sharpness and is equivalent to
an inverse-temperature intervention, not a direct self-conditioning strength.

Use `--sc-scales` to scale the gated-MLP branch output immediately before it is
added to the discrete token embedding:

```text
decoder_input = post_norm(token_embedding + scale * self_conditioning_branch)
```

Run a six-point sweep on one 256-token sample:

```bash
python run.py --sc-scales 1,0.8,0.6,0.4,0.2,0 --steps 48 --sample-ids codemerge
```

Results are stored under `outputs/len256/step48/scscale*/codemerge/`. Scale 0
removes the branch contribution and is the endpoint corresponding to `none`.

Plot only this sweep:

```bash
python visual.py --output-root outputs/len256/step48 --experiment-prefix scscale --out-dir visual/scscale --mode chart
```
