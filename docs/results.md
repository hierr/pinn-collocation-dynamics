# Experimental results

This document consolidates scientific comparisons for quick access. Link each
table or figure to its campaign README, retained artifacts, and relevant study phase.
Identify the metric, units, planned/completed seed counts, variability, and
whether a comparison concerns tuning or frozen-setting evaluation. Include brief
interpretation and limitations needed to understand each comparison; broader
study decisions belong in [research context](research_context.md).

## Earlier development at ν = 0.01/π

The [controlled N1024 Burgers sigma sweep](../output/burgers/nu_0p0031830989/sigma_0/README.md)
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
realized work; see the [campaign limitation](../output/burgers/nu_0p0031830989/sigma_0/README.md#optimizer-work).
Per-seed values and full run records are in the campaign artifacts.
See the [study design and next phase](research_context.md#current-development-and-evaluation-plan).

## Diffusion tuning at ν = 0.01

The [ν = 0.01 sweep](../output/burgers/nu_0p01/sigma_0/README.md)
completed all 15 planned runs with no numerical failures: ordinary PACMANN and
four positive diffusion coefficients, each on development seeds 3000–3002.
It uses the generated N1024 reference at **0.01**, not `0.01/pi`.
The evaluation grids match the earlier sweep. Sigma zero is a comparison
baseline only; select among positive candidates by mean final solution relative L2.

| Sigma | Final solution relative L2, mean ± sample SD (%) | Mean paired change vs. sigma 0 (percentage points) | Seeds improved |
| ---: | ---: | ---: | ---: |
| 0 | 0.0289837 ± 0.0112685 | — | — |
| 1e-5 | 0.0309733 ± 0.00809073 | +0.00198960 | 1/3 |
| 1e-4 | 0.0343620 ± 0.0142494 | +0.00537824 | 1/3 |
| **1e-3** | **0.0262913 ± 0.00596281** | **-0.00269247** | 2/3 |
| 1e-2 | 0.0424632 ± 0.00330482 | +0.0134794 | 1/3 |

Select **sigma = 1e-3**; its interior position requires no search expansion.
Residual error is evaluated at this solution-selected setting.

| Sigma | Final PDE-residual RMS, mean ± sample SD | Mean paired change vs. sigma 0 | Seeds improved |
| ---: | ---: | ---: | ---: |
| 0 | 0.00187201 ± 0.0000839024 | — | — |
| 1e-3 | 0.00173037 ± 0.000505739 | -0.000141638 | 2/3 |

Mean solution error is 9.29% lower and mean residual RMS is 7.57% lower than
ordinary PACMANN on these development seeds. The small selection sample does
not establish a held-out diffusion benefit or a sharpness effect. The reference
passed structural checks, without independent accuracy certification at this
viscosity. Per-seed outcomes, execution conditions, and optimizer-work limitations
are in the campaign README and artifacts. No ten-seed evaluation at this
viscosity has started.

## Diffusion tuning at nu = 0.1

The [nu = 0.1 development sweep](../output/burgers/nu_0p1/sigma_0/README.md)
completed all 21 planned runs without numerical failures. It compared ordinary
PACMANN with six positive settings, including one approved lower-boundary
expansion, using paired development seeds 3000-3002. The reference at stored
viscosity 0.1 passed structural and loader checks but has no independent
numerical-accuracy certification at this viscosity. Solution relative L2 is
reported in percent; paired changes are percentage points.

| Sigma | Final solution relative L2, mean ± sample SD (%) | Mean paired change vs. sigma 0 (percentage points) | Seeds improved | Mean training time per run (s) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0177093 ± 0.00715926 | - | - | 9344.3 |
| 1e-7 | 0.0122791 ± 0.00109882 | -0.00543022 | 2/3 | 8951.1 |
| 1e-6 | 0.0129587 ± 0.00294386 | -0.00475061 | 2/3 | 9373.4 |
| **1e-5** | **0.00961106 ± 0.00186134** | **-0.00809823** | **3/3** | 9417.0 |
| 1e-4 | 0.0168219 ± 0.00633167 | -0.000887435 | 2/3 | 8736.4 |
| 1e-3 | 0.0150394 ± 0.0111366 | -0.00266984 | 2/3 | 9192.9 |
| 1e-2 | 0.0288924 ± 0.0084914 | +0.0111831 | 1/3 | 9310.8 |

The positive candidate with the lowest mean final solution relative L2 was
selected, with exact ties assigned to the smaller positive value; sigma zero
was a comparison baseline only. The selected sigma = 1e-5 is interior among
the tested positive settings. Mean solution error was 45.73% lower than the
baseline on these development seeds. Residual error was assessed at the same
selected setting, without separate tuning.

| Sigma | Final PDE-residual RMS, mean ± sample SD | Mean paired change vs. sigma 0 | Seeds improved |
| ---: | ---: | ---: | ---: |
| 0 | 0.00115176 ± 0.000400647 | - | - |
| **1e-5** | **0.000773829 ± 0.000180949** | **-0.000377936** | **3/3** |

These three-seed development results do not establish held-out performance or
a general diffusion benefit across viscosities. Per-seed outcomes, search
history, execution conditions, and provenance are in the campaign README and
tracked artifacts. Training times reflect three-way GPU concurrency; they are
not directly comparable with ten-way evaluation timing.

## Diffusion tuning at pi/100

The [π/100 tuning campaign](../output/burgers/nu_0p031415927/sigma_0/README.md)
completed all 15 planned runs without numerical failures. It used paired seeds
3000–3002 and the restored N1024 reference, stored at viscosity
0.031415926535897934. The reference passed structural and loader checks, but
its numerical accuracy is not independently certified. All runs record source
revision `cb6c6e8`; all 15 final checkpoints independently reproduced the five
saved error metrics on CPU.

The positive sigma with the lowest mean final solution relative L2 was selected.
Exact ties favor the smaller positive value; sigma zero is a comparison baseline
only. The interior positive winner was **sigma = 1e-4**, so no search expansion
was required. Solution L2 is reported as percent and paired changes as
percentage points.

| Sigma | Final solution relative L2, mean ± sample SD (%) | Mean paired change vs. sigma 0 (percentage points) | Improved / 3 | Mean training seconds per run |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0135490 ± 0.00717480 | — | — | 3239.2 |
| 1e-5 | 0.0157212 ± 0.00568930 | +0.00217220 | 1/3 | 3165.9 |
| **1e-4** | **0.0142509 ± 0.00391486** | **+0.000701916** | **1/3** | 3296.5 |
| 1e-3 | 0.0169627 ± 0.00251858 | +0.00341373 | 1/3 | 3252.4 |
| 1e-2 | 0.0327593 ± 0.0102919 | +0.0192103 | 0/3 | 3163.9 |

The sigma-zero baseline had a lower mean solution error than every positive
candidate. The accepted positive-only rule still selects sigma 1e-4 for
evaluation. At that setting, mean residual RMS was 0.000968064 ± 0.000112133,
versus 0.000959376 ± 0.000249007 for sigma zero; the paired residual difference
was +0.00000868754 ± 0.000154842, with lower residual in one of three seeds.
Residual was assessed at the solution-selected sigma and did not select a
different setting.

Each run used 35,000 Adam steps. Mean ± sample SD observed L-BFGS iterations and
function evaluations per run were:

| Sigma | L-BFGS iterations | L-BFGS function evaluations |
| ---: | ---: | ---: |
| 0 | 16891 ± 656.2 | 27411.7 ± 2907.3 |
| 1e-5 | 16463 ± 482.6 | 26983.7 ± 1371.1 |
| **1e-4** | **17101.7 ± 486.1** | **28792 ± 2102.0** |
| 1e-3 | 16828 ± 581.7 | 27838.3 ± 1712.9 |
| 1e-2 | 16368 ± 83.1 | 27014 ± 2628.0 |

Training times include compilation and training, exclude post-processing, and
reflect three concurrent seeds on one RTX 5070 using float64. The small
development sample does not establish held-out performance, a general
diffusion benefit, or a sharpness effect. The selected setting is frozen for
the separate ten-seed evaluation on new seeds after transfer to `main`.

## Diffusion tuning at ν = 0.001

The [ν = 0.001 tuning campaign](../output/burgers/nu_0p001/sigma_0/README.md)
completed all 21 planned runs without reported execution failures: ordinary
PACMANN and six positive coefficients on paired development seeds 3000–3002.
The initial boundary winner, `sigma = 1e-5`, triggered the approved expansion
to `1e-7` and `1e-6`. All outcomes are retained.

The N4096 working reference has N2048–N4096 solution disagreement of 0.00964855%
relative L2; this is refinement evidence, not an absolute error bound. The
baseline spatial-grid check supported the common 1,024-by-199 solution grid
and 1,024-by-200 residual midpoint grid. All 21 restored final checkpoints
reproduced their saved predictions and five error metrics. Reference and
evaluation limitations are recorded in the campaign README.

| Sigma | Final solution relative L2, mean ± sample SD (%) | Mean paired change vs. sigma 0 (percentage points) | Improved / 3 |
| ---: | ---: | ---: | ---: |
| 0 | 41.75898 ± 28.78297 | — | — |
| 1e-7 | 59.98023 ± 17.72189 | +18.22125 | 0/3 |
| **1e-6** | **44.16674 ± 36.08290** | **+2.40776** | **1/3** |
| 1e-5 | 54.57271 ± 20.69209 | +12.81373 | 1/3 |
| 1e-4 | 70.95560 ± 18.66465 | +29.19662 | 0/3 |
| 1e-3 | 62.27386 ± 17.67351 | +20.51488 | 1/3 |
| 1e-2 | 58.15774 ± 14.10763 | +16.39876 | 0/3 |

Freeze **sigma = 1e-6** under the predeclared positive-only rule. It is interior
among tested positive values, so no further expansion is required. Sigma zero
had lower mean solution error than every positive candidate. At the selected
setting, individual solution errors were 78.27303%, 47.83910%, and 6.38810%;
the improvement is confined to seed 3002 and does not establish a diffusion
benefit. These are large and highly variable errors under the fixed training
budget, despite successful execution and reproducible scoring.

Residual error was assessed at the same solution-selected setting:

| Sigma | Final PDE-residual RMS, mean ± sample SD | Mean paired change vs. sigma 0 | Improved / 3 |
| ---: | ---: | ---: | ---: |
| 0 | 2.931608 ± 2.791394 | — | — |
| **1e-6** | **2.949592 ± 1.737246** | **+0.017983** | **1/3** |

Mean residual RMS was also slightly higher at the selected coefficient, with
lower residual only in seed 3002. The three-seed selection sample does not
establish held-out performance or a sharpness effect. Full per-seed outcomes,
optimizer work, and concurrency conditions are in the campaign README and
artifacts. The separate ten-seed evaluation has not started.
