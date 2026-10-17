# Burgers ν = 0.01: frozen ten-seed evaluation

**Prepared; training has not started.** The
[configuration](../../../config/burgers_nu_0p01_10seed.yaml) freezes five methods
and ten paired evaluation seeds, for 50 planned runs.

## Design

Evaluate fixed Hammersley, upstream random resampling, RAD, PACMANN, and PACMANN
with diffusion at **ν = 0.01**, using
`data/burgers_dataset/burgers_nu_0p01_N1024.npz`. Diffusion uses **sigma = 1e-3**,
selected on development seeds 3000–3002 in `hyperparameter-tuning` commit
`2baa9dd`. The positive search winner was interior, so no expansion was needed.
See the [development result](../../../docs/results.md#diffusion-development-selection-at-nu-001).

Use evaluation seeds 4000–4009 for every method, excluding the tuning seeds.
These seed identifiers also occur in the earlier viscosity campaign; retain
them all, including unfavorable outcomes. Network, point budgets, fixed
constraints, optimizer stages, baseline sampling settings, evaluation grids,
and reporting rules match the
[shared comparison design](../burgers_nu_0p0031830989_10seed/README.md#design-and-comparisons).
Only viscosity, reference dataset, and the separately selected diffusion sigma
change. Do not retune on evaluation results.

The primary contrast is diffusion versus PACMANN for final solution relative L2.
Assess the same contrast for independent residual RMS, without a separate sigma
selection. Compare adaptive methods against fixed Hammersley and random
resampling as in the shared design. Report all per-seed outcomes, means, sample
SD, paired differences and improvement counts; supplementary trimmed means
remove one minimum and one maximum separately for solution L2 and residual RMS.
Boundary and initial-condition RMSE are constraint diagnostics. Report actual
optimizer work and timing, retaining numerical failures and all planned seeds.

This campaign evaluates held-out seeds at one viscosity. Comparison with the
earlier viscosity does not by itself establish a sharpness-dependent benefit;
explicit sharpness and feature-region evaluation remain pending. The generated
reference passed structural and loader checks, without independent numerical
accuracy certification at this viscosity. Baseline samplers retain upstream
settings while diffusion is locally tuned; disclose that tuning policy.

## Reproduction

Use the [documented environment](../../../README.md#setup-and-execution).
The runner selects `config/burgers_nu_0p01_10seed.yaml` and the five methods in
the order listed above. Execute from the repository root:

```sh
venv/bin/python -m scripts.run_experiments.run_burgers_campaign
```

Each method launches ten concurrent seed processes, each with one numerical CPU
thread, sharing one GPU. The next method starts after all workers finish
training and final evaluation. Use the same execution conditions as the earlier
campaign: NVIDIA GeForce RTX 5070 (12 GB), float64. Timing includes compilation
and training, excludes final evaluation, and reflects shared-GPU concurrency.
It is not comparable to the three-way-concurrent development timings.

Set the runner's `METHODS` to a subset to stop after selected method batches.
The runner rejects occupied seed destinations and stops after a failed batch's
other workers finish. Preserve artifacts and follow the
[recovery policy](../../../AGENTS.md#execution-recovery). For a full repetition,
copy the YAML with a new campaign name and select that copy in the runner;
keep scientific settings fixed.

Outputs will appear under `<method>/seed_<seed>/` alongside this README.
Add per-seed results and aggregates here after verification, and consolidate
comparisons in [results](../../../docs/results.md). See
[study context](../../../docs/research_context.md#reference-validation-and-current-evidence)
and the [artifact dictionary](../../../docs/output_format.md).
