# Burgers nu = pi/100: prepared ten-seed evaluation

This campaign is prepared; diffusion sigma is pending development tuning and
training has not started. Compare fixed Hammersley, upstream random resampling,
RAD, PACMANN, and PACMANN with diffusion on paired seeds 4000-4009. Select sigma
using separate development seeds 3000-3002 on `hyperparameter-tuning`, whose
prepared protocol is commit `cb6c6e8`. Freeze that positive coefficient before
evaluation and use the same value for solution and residual outcomes.

The [configuration](../../../config/burgers_nu_0p031415927_10seed.yaml) retains
the network, point budgets, fixed constraints, optimizer schedule, evaluation
grids, and method controls of the
[preceding campaign](../burgers_nu_0p1_10seed/README.md#design). The reference
`data/burgers_dataset/burgers_nu_0p031415927_N1024.npz` is restored unchanged
from dataset commit `52b9359`, with stored viscosity `0.031415926535897934`.
Its SHA-256 is
`aad9aff8eecc32e1fe9e60ad2e4ca5f399dd5e96bf748851f085a2165923fb60`.
Structural, constraint, and loader checks passed; these checks do not
independently certify numerical accuracy.

## Reproduction

After tuning, replace `sigma: null` in the configuration with the selected
value. The runner selects this YAML and all five methods in the order listed
above; an unset sigma prevents the entire selected sequence from starting.
Use the [documented environment](../../../README.md#setup-and-execution) and
execute from the repository root:

```sh
venv/bin/python -u -m scripts.run_experiments.run_burgers_campaign
```

Retain the established timing policy: ten concurrent seeds per method on one
NVIDIA GeForce RTX 5070 (12 GB), float64, and one numerical CPU thread per
worker. Methods run sequentially. Outputs belong under
`output/burgers/burgers_nu_0p031415927_10seed/<method>/seed_<seed>/`.

Use final-model metrics on all ten seeds and retain failures. Report per-seed
outcomes, means and sample SD, paired differences, and supplementary trimmed
means dropping one minimum and one maximum per metric. Solution relative L2
uses percent, paired solution differences use percentage points, and residual
RMS retains its original units. Do not retune on evaluation seeds or select a
separate sigma for residual error. Add outcomes here after completion and
consolidate comparisons in [results](../../../docs/results.md).
See [study context](../../../docs/research_context.md) and
[artifact meanings](../../../docs/output_format.md).
