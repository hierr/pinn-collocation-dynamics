# Burgers ν = 0.1 diffusion tuning

**Prepared; no scientific runs started.** This development sweep selects a
positive diffusion coefficient before a separate frozen ten-seed evaluation.
It extends the per-viscosity study to a more diffusive Burgers regime.

## Protocol

Compare ordinary PACMANN (`sigma = 0`) with positive sigmas `1e-5`, `1e-4`,
`1e-3`, and `1e-2`, each on paired development seeds 3000–3002: 15 initial runs.
The five [YAML configurations](../../../../config/burgers/nu_0p1/) named
`burgers_nu_0p1_sigma_*.yaml` retain the previous sweep's
network, training schedule, sampling settings, point budgets, and evaluation
grids. Only viscosity, reference path, and campaign names change.

The reference is `data/burgers_dataset/burgers_nu_0p1_N1024.npz`, transferred
unchanged from `main` dataset commit `dbeb289`. Its stored viscosity is **0.1**.
Structural, PDE-parameter, constraint, and loader checks establish input
consistency; they do not independently certify numerical accuracy at this
viscosity. The original-reference validation at `0.01/pi` does not certify this
reference.

Select the positive sigma with the lowest arithmetic mean final solution
relative L2 across all three development seeds; exact ties favor the smaller
positive sigma. Zero is a comparison baseline only, even if it outperforms all
positive settings. If the winner is at a positive search boundary, extend by
two values at the next two decades on that side, with at most two expansion
rounds. Then freeze the best tested positive sigma and disclose any remaining
boundary limitation. The prepared runner executes only the initial grid.
Evaluate residual error at the solution-selected sigma, without separate tuning.
Retain all planned seeds and numerical failures; incomplete sweeps do not select
a coefficient. This campaign does not launch a ten-seed evaluation.

## Reproduction

Use the [documented environment](../../../../README.md#setup-and-execution).
`CONFIG_PATHS` in `scripts/run_experiments/run_burgers_sigma_tuning.py` selects
the five ν = 0.1 configurations. `ACTION = "run"` executes three seeds
concurrently per candidate, with candidates run sequentially. The planned
device is one NVIDIA GeForce RTX 5070 (12 GB), with float64 and one numerical
CPU thread per worker. Execute from the repository root:

```sh
venv/bin/python -m scripts.run_experiments.run_burgers_sigma_tuning
```

Set `ACTION = "summarize"` to read saved artifacts without starting training.
Occupied seed destinations are rejected. For a new repetition, select copies
of the YAML files with unused campaign names, retaining scientific settings.
Add per-seed outcomes and aggregates after verification.

The baseline runs belong under `pacmann_adam/seed_<seed>/` here; each positive
candidate uses `pacmann_diffusion/seed_<seed>/` in its sibling campaign directory.
See [study context](../../../../docs/research_context.md#current-development-and-evaluation-plan),
[selection policy](../../../../docs/research_context.md#accepted-controls-and-evaluation-policy),
[results](../../../../docs/results.md), and the
[artifact dictionary](../../../../docs/output_format.md).
