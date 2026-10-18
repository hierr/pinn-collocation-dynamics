# Burgers ν = 0.1: ten-seed evaluation

This frozen-setting campaign compares fixed Hammersley, upstream random
resampling, RAD, PACMANN, and PACMANN with diffusion on paired seeds 4000–4009
at viscosity 0.1. The diffusion coefficient is **σ = 1e-5**, selected using
separate development seeds 3000–3002. The tuning completed 21 runs, including
one approved lower-boundary expansion, and selected the positive coefficient
with the lowest mean final solution relative L2. The selected value was interior
among the tested positive settings; ordinary PACMANN at σ = 0 remained a
comparison baseline only. The selection and retained tuning artifacts are in
[`hyperparameter-tuning` commit `1968b35`](https://github.com/hierr/pinn-collocation-dynamics/commit/1968b350a5d44adc9180782406357c2c8763be92),
trained from source commit
[`0948ad9`](https://github.com/hierr/pinn-collocation-dynamics/commit/0948ad936fb32502c19f0b174572866799ca1285).

## Design

All five methods use the same network, point budgets, fixed boundary and initial
constraints, optimizer schedule, and evaluation grids as the preceding frozen
comparison. Only viscosity, generated reference dataset, and the development-
selected diffusion coefficient differ. The [frozen configuration](../../../config/burgers_nu_0p1_10seed.yaml)
uses the N1024 reference at
`data/burgers_dataset/burgers_nu_0p1_N1024.npz`; its stored viscosity is 0.1
and its SHA-256 is
`78fbdc52d18e6538668d127174d5a94ed5999ef8c295a5fa9414592f14f9a833`.
This reference passed structural and loader checks, but it has no independent
numerical-accuracy certification at this viscosity.

Use final-model errors on all ten seeds. Retain unfavorable outcomes and report
per-seed values, mean and sample SD, paired differences, and improvement counts.
Supplementary trimmed means remove one minimum and one maximum independently
for solution L2 and residual RMS. Report solution relative L2 in percent and
paired solution differences in percentage points; report residual RMS in its
original units. Boundary and initial-condition RMSE are separate constraint
diagnostics. Do not retune sigma or select a different value for residual error
using these evaluation seeds.

## Reproduction

Use the [documented environment](../../../README.md#setup-and-execution). The
launcher selects `config/burgers_nu_0p1_10seed.yaml` and methods in this order:
fixed Hammersley, random resampling, RAD, PACMANN, and PACMANN with diffusion.
From the repository root, execute:

```sh
venv/bin/python -u -m scripts.run_experiments.run_burgers_campaign
```

The launcher runs ten seeds concurrently per method, with one numerical CPU
thread per worker, sharing one NVIDIA GeForce RTX 5070 (12 GB) using float64.
Methods run sequentially; each next batch starts after the current method has
finished training and evaluation. Training times include compilation and
training, exclude final evaluation, and reflect ten-way GPU concurrency. They
are not directly comparable with the three-way tuning timings.

Runs are retained under
`output/burgers/burgers_nu_0p1_10seed/<method>/seed_<seed>/`. After completion,
summarize every planned seed in this README and the study-level comparison in
[results](../../../docs/results.md). See the [study context](../../../docs/research_context.md)
and [artifact dictionary](../../../docs/output_format.md).

The comparison measures these methods and this per-viscosity tuning policy. It
does not independently certify the generated reference or establish a general
sharpness-dependent effect; explicit sharpness and feature-region evaluation
remain future work.
