# Self-conditioning entropy convergence experiment

The default configuration runs exactly these three settings on `codemerge`,
`mathkkt`, and `rewrite3`, using a 256-token canvas and the same seed:

| Setting | Maximum steps | Temperature schedule |
|---|---:|---|
| `alpha=1.0` | 48 | Baseline 48-step schedule |
| `alpha=0.5` | 48 | Baseline 48-step schedule |
| `none` | 96 | Same schedule for the first 48 steps, then `t_min` |

Run generation on the GPU machine:

```bash
python run.py
```

`sweep.experiments` in `config.yaml` specifies exact alpha/budget pairs. Passing
`--alphas` or `--steps` explicitly retains the older Cartesian-product mode.

Each `trace.csv` row records entropy from the temperature-processed logits used
by the sampler for that step, before acceptance and re-noising affect the next
canvas. It also records the accepted positions and accepted count from the same
step.

Generate the entropy analysis locally:

```bash
python visual_entropy.py
```

Outputs are written under `visual/entropy/`:

- `entropy_heatmaps.png`: 3 tasks x 5 settings with one shared color scale,
  including both 48- and 96-step zero-logits runs.
- `mean_entropy_vs_step.png`: mean entropy over all 256 positions.
- `accepted_tokens_vs_step.png`: per-step accepted token count.
- `entropy_trace_long.csv`: step-level values and accepted-position lists.
- `entropy_summary.csv`: stopping steps and entropy/acceptance summary statistics.

Heatmap steps increase from bottom to top. Each panel ends at its actual adaptive
stopping step instead of being padded to 96, so short convergence trajectories
remain readable. The red dashed line appears only when a trace crosses the
common 48-step comparison boundary.
