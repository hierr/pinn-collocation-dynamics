# Experimental results

This document consolidates scientific comparisons for quick access. Link each
table or figure to its campaign README, retained artifacts, and relevant study phase.
Identify the metric, units, planned/completed seed counts, variability, and
whether a comparison concerns tuning or frozen-setting evaluation. Include brief
interpretation and limitations needed to understand each comparison; broader
study decisions belong in [research context](research_context.md).

## Current evidence

The [controlled N1024 Burgers sigma sweep](../output/burgers_sigma_tuning_n1024_baseline/README.md)
completed all 15 planned runs with no failures. It used three paired development
seeds per setting on the generated `N1024` reference at `nu = 0.01/pi`. This is
hyperparameter tuning, not the frozen-setting evaluation.
Solution relative L2 uses the common 1,024-by-199 solution grid; residual RMS
uses the independent 1,024-by-200 interior midpoint grid.

| Sigma | Final solution relative L2, mean ± sample SD (%) | Mean paired change vs. sigma 0 (percentage points) | Seeds improved | Mean training time per run (s) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.140387 ± 0.0970043 | — | — | 928.8 |
| 1e-5 | 0.0951455 ± 0.0907051 | -0.0452411 | 2/3 | 925.1 |
| 1e-4 | **0.0796273 ± 0.0464911** | **-0.0607594** | 2/3 | 927.8 |
| 1e-3 | 0.129885 ± 0.0412965 | -0.0105021 | 1/3 | 920.1 |
| 1e-2 | 0.18753 ± 0.0879229 | +0.0471435 | 1/3 | 928.1 |

Sigma `1e-4` is the development-selected candidate because it had the lowest
mean final solution relative L2. Paired changes are diffusion minus the
same-seed sigma-0 errors; negative values indicate lower error. The sample
SD describes variation across the three seeds.
Training times include compilation and optimizer stages, and exclude
post-processing.

Residual error was evaluated with the same three seeds at the solution-selected
sigma, without choosing a separate residual-optimal value.

| Sigma | Final PDE-residual RMS, mean ± sample SD | Mean paired change vs. sigma 0 | Seeds improved |
| ---: | ---: | ---: | ---: |
| 0 | 0.00432176 ± 0.000872872 | — | — |
| 1e-4 | **0.00364778 ± 0.000266738** | **-0.000673973** | 2/3 |

At this one viscosity, both mean solution error and mean residual RMS were lower
at the selected candidate than at sigma 0, with improvement in two of three
paired seeds for each outcome. The small development sample and single
viscosity do not establish a general diffusion benefit or a sharpness effect.
The candidate remains to be assessed on new seeds in the frozen-setting
evaluation. DeepXDE's batched L-BFGS stopping exceeded the configured iteration
limit in some stages, so these runs share optimizer settings but have different
realized work; see the [campaign limitation](../output/burgers_sigma_tuning_n1024_baseline/README.md#optimizer-work).
Per-seed values and full run records are in the campaign artifacts.
See the [study design and next phase](research_context.md#current-development-and-evaluation-plan).
