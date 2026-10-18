# Burgers ν = 0.01 diffusion tuning

Scientific development: select a positive diffusion coefficient before the
separate ten-seed evaluation. All 15 planned runs completed without numerical
failures, using paired seeds 3000–3002. Sigma zero is ordinary PACMANN and is a
comparison baseline, not an eligible diffusion setting.

## Protocol and reproduction

The five [YAML configurations](../../../../config/burgers/nu_0p01/) named
`burgers_nu_0p01_sigma_*.yaml` fix viscosity at **0.01** and use
`data/burgers_dataset/burgers_nu_0p01_N1024.npz`. Source commit:
`30693a705207adc07f2c13cf0e522c83ef311c45`; each run retains its configuration,
source revision, dependency versions, and dataset fingerprint.
The new reference passed structural and loader checks; numerical accuracy at
this viscosity was not independently certified.

From the repository root, with the [documented environment](../../../../README.md#setup-and-execution):

```sh
venv/bin/python -m scripts.run_experiments.run_burgers_sigma_tuning
```

Set the runner's `CONFIG_PATHS` to the five `burgers_nu_0p01_sigma_*.yaml`
configurations above; its default selection may point to a later viscosity. Set
`ACTION = "summarize"` to read retained results without training. For a new
execution, select copies of these YAML files with unused campaign names; the
runner rejects occupied destinations. Keep all scientific settings unchanged.

Three seeds ran concurrently on one NVIDIA GeForce RTX 5070 (12 GB), with
candidates executed sequentially. Training used float64. Solution relative L2
uses a common 1,024-by-199 uniform grid with spatial barycentric interpolation
of the reference; residual RMS uses an independent 1,024-by-200 interior midpoint
grid. Timing includes compilation and optimizer stages, excludes analysis,
and reflects three-way concurrency; it is not comparable to serial-run timing.

Select the positive sigma with the lowest arithmetic mean final solution
relative L2; exact ties favor the smaller sigma. The approved boundary expansion
rule is in [research context](../../../../docs/research_context.md#accepted-controls-and-evaluation-policy).
Residual error is assessed at the solution-selected sigma, without separate tuning.

## Final outcomes

Solution relative L2 is expressed as a percentage. SD is the sample standard
deviation across the three development seeds. Per-seed residual RMS is unnormalized.

| Sigma | Seed 3000 L2 (%) | Seed 3001 L2 (%) | Seed 3002 L2 (%) | Mean ± SD (%) | Mean training seconds/run |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0175369 | 0.0293492 | 0.0400651 | 0.0289837 ± 0.0112685 | 3009.9 |
| 1e-5 | 0.0280652 | 0.0247387 | 0.0401162 | 0.0309733 ± 0.00809073 | 3068.1 |
| 1e-4 | 0.0276861 | 0.0507238 | 0.0246761 | 0.0343620 ± 0.0142494 | 3012.2 |
| **1e-3** | 0.0236522 | 0.0221034 | 0.0331182 | **0.0262913 ± 0.00596281** | 3152.1 |
| 1e-2 | 0.0419175 | 0.0460069 | 0.0394651 | 0.0424632 ± 0.00330482 | 3076.4 |

| Sigma | Seed 3000 RMS | Seed 3001 RMS | Seed 3002 RMS | Mean ± SD |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00177791 | 0.00193902 | 0.00189910 | 0.00187201 ± 0.0000839024 |
| 1e-5 | 0.00194138 | 0.00179157 | 0.00223119 | 0.00198805 ± 0.000223493 |
| 1e-4 | 0.00163619 | 0.00240786 | 0.00183988 | 0.00196131 ± 0.000399910 |
| **1e-3** | 0.00230759 | 0.00151849 | 0.00136503 | **0.00173037 ± 0.000505739** |
| 1e-2 | 0.00210300 | 0.00310105 | 0.00249675 | 0.00256693 ± 0.000502712 |

Select **sigma = 1e-3**. It is an interior candidate, so no expansion is needed.
Compared with ordinary PACMANN, mean solution error decreases by 0.00269247
percentage points (9.29% relative), and mean residual RMS decreases by
0.000141638 (7.57% relative). Each outcome improves in two of three paired
seeds. These are selection-set observations, not evidence of a held-out benefit
or a general sharpness effect. The ten-seed evaluation is not part of this sweep.

Run directories are under [pacmann_adam](pacmann_adam/) here and
`pacmann_diffusion/` in the four sibling sigma campaigns. Final weights were
re-evaluated independently on CPU and reproduced all five saved evaluation
metrics within floating-point tolerance. Matched initial PDE populations,
fixed constraints, and no point movement during L-BFGS were checked.
DeepXDE's batched L-BFGS can exceed configured iteration limits; actual work is
retained in each `training.json` rather than inferred from nominal budgets.

See [consolidated results](../../../../docs/results.md),
[study context](../../../../docs/research_context.md), and
[artifact dictionary](../../../../docs/output_format.md).
